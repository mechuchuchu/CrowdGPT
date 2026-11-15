Crowd-v1 WikiText-2 token-PPL sweep

Included
- eval_crowd_v1_step5_token_ppl.py: evaluates the selected checkpoints.
- run_crowd_v1_step0_eval.py: Crowd-v1 model definition and flat-weight loader.
- crowd_v1_checkpoint_history.csv: checkpoint metadata used to select commits and trained-token counts.
- crowd_v1_step5_wikitext2_token_ppl.csv: completed results (75 checkpoints, global steps 0 through 370 every 5 steps).

Evaluation setup
- Checkpoints: latest global-step-0 restart, revision 7fcdab4072fab14c6d3696ee21c088b8086b94d5 onward.
- Dataset: EleutherAI/wikitext_document_level, wikitext-2-raw-v1, test split.
- Tokenizer: Qwen/Qwen3-0.6B (151,669 tokens including added tokens).
- Batch size: 8.
- token_ppl = exp(-sum(token log-likelihoods) / number of scored tokenizer tokens).
- The weights are fetched from Hugging Face by commit revision; model weights are not bundled.

Run
Extract these files directly into /workspace, then run:
  HF_XET_CACHE=/tmp/crowd_v1_step5_xet python /workspace/eval_crowd_v1_step5_token_ppl.py

The script resumes from rows already present in the result CSV. Move the existing result CSV out of the way to run the full sweep from scratch. It expects a CUDA-enabled PyTorch environment with lm-eval, accelerate, transformers, datasets, huggingface-hub, and safetensors installed.
