import csv
import math
import os
import re
import shutil
import time
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from lm_eval.api.instance import Instance
from transformers import AutoTokenizer

from lm_eval.tasks.wikitext.preprocess_wikitext import wikitext_detokenizer
from run_crowd_v1_step0_eval import (
    BATCH_SIZE,
    CrowdHFLM,
    CrowdModel,
    MAX_SEQ_LEN,
    MODEL_DIR as STEP0_MODEL_DIR,
    REVISION as STEP0_REVISION,
    VOCAB_SIZE,
)


REPO_ID = "Vxtzq/Crowd-v1"
HISTORY_CSV = Path("/workspace/crowd_v1_checkpoint_history.csv")
RESULT_CSV = Path("/workspace/crowd_v1_step5_wikitext2_token_ppl.csv")
TEMP_ROOT = Path("/tmp/crowd_v1_step5_eval")
XET_CACHE = Path(os.environ.get("HF_XET_CACHE", "/tmp/crowd_v1_step5_xet"))
DEVICE = torch.device("cuda:0")


def get_checkpoints():
    with HISTORY_CSV.open(newline="", encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    start = next(i for i, row in enumerate(history) if row["commit_hash"] == STEP0_REVISION)
    by_step = {}
    for row in history[start:]:
        step = int(row["global_step"])
        if step % 5 == 0 and step not in by_step:
            by_step[step] = row
    expected = list(range(0, 371, 5))
    missing = [step for step in expected if step not in by_step]
    if missing:
        raise RuntimeError(f"Missing checkpoints: {missing}")
    return [by_step[step] for step in expected]


def load_checkpoint_files(row):
    step = int(row["global_step"])
    revision = row["commit_hash"]
    if revision == STEP0_REVISION:
        return STEP0_MODEL_DIR, 0.0, None

    cache_dir = TEMP_ROOT / f"step-{step:03d}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="model_bf16.safetensors",
        revision=revision,
        cache_dir=str(cache_dir),
    )
    engram_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="engram_bf16.safetensors",
        revision=revision,
        cache_dir=str(cache_dir),
    )
    return Path(model_path).parent, time.perf_counter() - start, cache_dir


def read_documents():
    dataset = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1",
        split="test",
    )
    docs = []
    for i, raw_doc in enumerate(dataset):
        text = wikitext_detokenizer(raw_doc)
        docs.append(
            {
                "id": i,
                "text": text,
                "raw_page": raw_doc["page"],
                "words": len(re.split(r"\s+", raw_doc["page"])),
                "bytes": len(raw_doc["page"].encode("utf-8")),
            }
        )
    return docs


def evaluate_revision(model, tokenizer, docs, row):
    torch.cuda.synchronize()
    lm = CrowdHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        max_length=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        add_bos_token=True,
        logits_cache=False,
    )
    requests = [
        Instance(
            request_type="loglikelihood_rolling",
            doc={"text": doc["text"]},
            arguments=(doc["text"],),
            idx=i,
            metadata=("wikitext", i, 1),
        )
        for i, doc in enumerate(docs)
    ]
    start = time.perf_counter()
    loglikelihoods = lm.loglikelihood_rolling(requests, disable_tqdm=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_loglikelihood = sum(loglikelihoods)
    token_count = sum(len(lm.tok_encode(doc["text"])) for doc in docs)
    nll = -total_loglikelihood
    token_ppl = math.exp(nll / token_count)
    word_ppl = math.exp(nll / sum(doc["words"] for doc in docs))
    byte_ppl = math.exp(nll / sum(doc["bytes"] for doc in docs))
    return {
        "token_ppl": token_ppl,
        "tokens_scored": token_count,
        "total_loglikelihood": total_loglikelihood,
        "word_ppl_check": word_ppl,
        "byte_ppl_check": byte_ppl,
        "eval_seconds": elapsed,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    checkpoints = get_checkpoints()
    max_checkpoints = int(os.environ.get("MAX_CHECKPOINTS", "0"))
    if max_checkpoints > 0:
        checkpoints = checkpoints[:max_checkpoints]
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    completed = set()
    if RESULT_CSV.exists():
        with RESULT_CSV.open(newline="", encoding="utf-8") as f:
            completed = {int(row["global_step"]) for row in csv.DictReader(f)}
    fieldnames = [
        "global_step",
        "trained_tokens",
        "token_ppl",
        "tokens_scored",
        "total_loglikelihood",
        "commit_hash",
        "commit_date_utc",
        "token_status_commit_hash",
        "eval_seconds",
        "download_seconds",
        "weight_load_seconds",
        "word_ppl_check",
        "byte_ppl_check",
    ]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if len(tokenizer) != VOCAB_SIZE:
        raise ValueError(f"Tokenizer length {len(tokenizer)} != expected {VOCAB_SIZE}")
    docs = read_documents()
    print(f"WikiText-2 test docs: {len(docs)}", flush=True)

    model = CrowdModel()
    model.load_flat_weights(STEP0_MODEL_DIR)
    model.to(DEVICE)
    model.engram.table.to("cpu")
    model.eval()
    output_mode = "a" if RESULT_CSV.exists() else "w"
    with RESULT_CSV.open(output_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if output_mode == "w":
            writer.writeheader()
            f.flush()

        for index, row in enumerate(checkpoints):
            step = int(row["global_step"])
            if step in completed:
                print(f"Skipping completed step {step}", flush=True)
                continue

            step_start = time.perf_counter()
            revision = row["commit_hash"]
            print(f"[{index + 1}/{len(checkpoints)}] step={step} revision={revision}", flush=True)
            model_dir, download_seconds, cache_dir = load_checkpoint_files(row)
            load_start = time.perf_counter()
            model.load_flat_weights(model_dir)
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - load_start
            model.eval()
            metrics = evaluate_revision(model, tokenizer, docs, row)

            if step == 0:
                print(
                    "step 0 check: "
                    f"word_ppl={metrics['word_ppl_check']:.6g}, "
                    f"byte_ppl={metrics['byte_ppl_check']:.8f}",
                    flush=True,
                )

            result = {
                **{key: row[key] for key in ("global_step", "trained_tokens", "commit_hash", "commit_date_utc", "token_status_commit_hash")},
                **metrics,
                "download_seconds": download_seconds,
                "weight_load_seconds": load_seconds,
            }
            writer.writerow(result)
            f.flush()
            os.fsync(f.fileno())
            completed.add(step)
            print(
                f"done step={step} token_ppl={metrics['token_ppl']:.6f} "
                f"eval={metrics['eval_seconds']:.1f}s download={download_seconds:.1f}s "
                f"load={load_seconds:.1f}s total={time.perf_counter() - step_start:.1f}s",
                flush=True,
            )

            if cache_dir is not None:
                shutil.rmtree(cache_dir, ignore_errors=True)
            if XET_CACHE.exists():
                shutil.rmtree(XET_CACHE, ignore_errors=True)
            XET_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"Saved table: {RESULT_CSV}", flush=True)


if __name__ == "__main__":
    main()
