import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, PretrainedConfig

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


REVISION = "7fcdab4072fab14c6d3696ee21c088b8086b94d5"
REPO_CACHE = Path("/workspace/.cache/huggingface/hub/models--Vxtzq--Crowd-v1/snapshots")
MODEL_DIR = REPO_CACHE / REVISION
VOCAB_SIZE = 151669
DIM = 1536
N_LAYERS = 24
N_HEADS = 16
N_KV_HEADS = 4
HEAD_DIM = 96
MAX_SEQ_LEN = 2048
MLP_HIDDEN = 2560
ENG_NUM_BUCKETS = 227865
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def precompute_freqs(dim, seq_len):
    inv = 1.0 / (
        10000.0
        ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    freqs = torch.einsum(
        "i,j->ij", torch.arange(seq_len, dtype=torch.float32), inv
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh = N_HEADS
        self.nkv = N_KV_HEADS
        self.nrep = N_HEADS // N_KV_HEADS
        self.wq = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x, cos, sin, attention_mask=None):
        batch, seq_len, channels = x.shape
        q = self.wq(x).view(batch, seq_len, self.nh, HEAD_DIM).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.nkv, HEAD_DIM).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.nkv, HEAD_DIM).transpose(1, 2)
        cos, sin = cos[:, :, :seq_len, :], sin[:, :, :seq_len, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        k = k.repeat_interleave(self.nrep, dim=1)
        v = v.repeat_interleave(self.nrep, dim=1)
        if attention_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            causal = torch.ones(
                (seq_len, seq_len), dtype=torch.bool, device=x.device
            ).tril()[None, None, :, :]
            keys_valid = attention_mask[:, None, None, :].to(torch.bool)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal & keys_valid, is_causal=False
            )
        return self.wo(out.transpose(1, 2).contiguous().view(batch, seq_len, channels))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w2 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w3 = nn.Linear(MLP_HIDDEN, DIM, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(DIM)
        self.attn = GroupedQueryAttention()
        self.ln_2 = nn.LayerNorm(DIM)
        self.mlp = SwiGLU()

    def forward(self, x, cos, sin, attention_mask=None):
        x = x + self.attn(self.ln_1(x), cos, sin, attention_mask)
        return x + self.mlp(self.ln_2(x))


class EngramMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(ENG_NUM_BUCKETS, DIM, sparse=True).to("cpu")

    def forward(self, idx, attention_mask=None):
        idx_cpu = idx.detach().cpu()
        prev = torch.cat((torch.zeros_like(idx_cpu[:, :1]), idx_cpu[:, :-1]), dim=1)
        if attention_mask is not None:
            mask_cpu = attention_mask.detach().cpu().to(torch.bool)
            prev_valid = torch.cat(
                (torch.zeros_like(mask_cpu[:, :1]), mask_cpu[:, :-1]), dim=1
            )
            prev = prev.masked_fill(~prev_valid, 0)
        hashed = (prev * 1000003 + idx_cpu) % ENG_NUM_BUCKETS
        unique, inverse = torch.unique(hashed.reshape(-1), return_inverse=True)
        gathered = self.table(unique)
        out = gathered[inverse].view(idx.shape[0], idx.shape[1], DIM)
        return out.to(idx.device)


class CrowdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.config.model_type = "llama"
        self.config.vocab_size = VOCAB_SIZE
        self.config.max_position_embeddings = MAX_SEQ_LEN
        self.config.is_encoder_decoder = False
        self.config.torch_dtype = torch.float32
        self.wte = nn.Embedding(VOCAB_SIZE, DIM)
        self.engram = EngramMemory()
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.wte.weight
        cos, sin = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN)
        self.register_buffer("freqs_cos", cos)
        self.register_buffer("freqs_sin", sin)

    @property
    def device(self):
        return self.wte.weight.device

    def tie_weights(self):
        self.lm_head.weight = self.wte.weight

    def load_flat_weights(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        model_flat = load_file(str(model_dir / "model_bf16.safetensors"))["weights"]
        offset = 0
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.startswith("engram."):
                    continue
                size = param.numel()
                param.copy_(model_flat[offset : offset + size].view_as(param))
                offset += size
        if offset != model_flat.numel():
            raise ValueError(f"Loaded {offset} model parameters, expected {model_flat.numel()}")

        engram_flat = load_file(str(model_dir / "engram_bf16.safetensors"))["weights"]
        with torch.no_grad():
            self.engram.table.weight.copy_(engram_flat.view(ENG_NUM_BUCKETS, DIM))
        print(f"Loaded model parameters: {offset:,}; Engram parameters: {engram_flat.numel():,}", flush=True)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None and not bool(attention_mask.all()):
            padding_mask = attention_mask
        else:
            padding_mask = None
        x = self.wte(input_ids)
        x = x + self.engram(input_ids, padding_mask)
        for block in self.blocks:
            x = block(x, self.freqs_cos, self.freqs_sin, padding_mask)
        x = self.ln_f(x)
        return SimpleNamespace(logits=self.lm_head(x))


class CrowdHFLM(HFLM):
    def _model_call(self, inps, attn_mask=None, labels=None):
        # lm-eval's causal path does not pass an attention mask into the model.
        # Recover the left-padding mask here; this also lets Engram ignore pad IDs.
        attention_mask = inps.ne(self.tokenizer.pad_token_id)
        with torch.no_grad():
            return self.model(input_ids=inps, attention_mask=attention_mask).logits


def main():
    total_start = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this timed run")
    device = torch.device("cuda:0")
    print(f"Revision: {REVISION}", flush=True)
    print(f"Device: {torch.cuda.get_device_name(device)}", flush=True)

    setup_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if len(tokenizer) != VOCAB_SIZE:
        raise ValueError(f"Tokenizer length {len(tokenizer)} != expected {VOCAB_SIZE}")
    model = CrowdModel()
    model.load_flat_weights()
    model.to(device)
    model.engram.table.to("cpu")
    model.eval()
    torch.cuda.synchronize()
    print(f"Tokenizer/model setup: {time.perf_counter() - setup_start:.1f}s", flush=True)

    with torch.inference_mode():
        sample = tokenizer("A short language modeling check.", return_tensors="pt")["input_ids"].to(device)
        output = model(sample).logits
        print(f"Forward check: shape={tuple(output.shape)} device={output.device}", flush=True)

    lm = CrowdHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        max_length=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        add_bos_token=True,
        logits_cache=True,
    )
    setup_seconds = time.perf_counter() - setup_start
    all_results = {}
    task_times = {}
    for task in ["wikitext", "piqa", "arc_easy", "winogrande"]:
        torch.cuda.synchronize()
        task_start = time.perf_counter()
        print(f"Starting {task}", flush=True)
        result = simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=0,
            batch_size=BATCH_SIZE,
            bootstrap_iters=0,
            log_samples=False,
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
        )
        torch.cuda.synchronize()
        task_times[task] = time.perf_counter() - task_start
        result_map = result["results"] if isinstance(result, dict) else result.results
        all_results[task] = result_map[task]
        metrics = result_map[task]
        print(f"Finished {task} in {task_times[task]:.1f}s: {metrics}", flush=True)

    total_elapsed = time.perf_counter() - total_start
    report = {
        "repo": "Vxtzq/Crowd-v1",
        "revision": REVISION,
        "global_step": 0,
        "tokenizer": "Qwen/Qwen3-0.6B",
        "device": torch.cuda.get_device_name(device),
        "batch_size": BATCH_SIZE,
        "num_fewshot": 0,
        "task_seconds": task_times,
        "setup_seconds": setup_seconds,
        "total_seconds": total_elapsed,
        "results": all_results,
    }
    output_path = Path(f"/workspace/crowd_v1_step0_lm_eval_b{BATCH_SIZE}_results.json")
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Total elapsed: {total_elapsed:.1f}s", flush=True)
    print(f"Results saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
