# Fine-Tuning Suite and Tests

Fine tuning Qwen 3.5 family models using Unsloth. Example code for fine-tuning LLMs locally using Unsloth's Python's API in `notebooks/` directory. We recommend starting with the `notebooks/fine-tune.ipynb` notebook for the basics of the fine-tuning process. 

Result plots can be found in `results/plots/` and analysis can be founds in the `notes/` directory: [[notes/qwen3.5-eb-guardian-results.md]].

All tests were run on one NVIDIA A100 GPU.

## Full Project Structure

Here's what the project structure looks like with results for fine-tuning a guardian model for EchoBot from Qwen 3.5 9B:

```
local-llm-fine-tuning
├── README.md
├── adapters
│   └── eb-qwen3.5-9b-lora-522
│       ├── README.md
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── chat_template.jinja
│       ├── processor_config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── checkpoints
│   └── eb-qwen3.5-9b-lora-522
│       ├── README.md
│       └── checkpoint-32
│           ├── README.md
│           ├── adapter_config.json
│           ├── adapter_model.safetensors
│           ├── chat_template.jinja
│           ├── optimizer.pt
│           ├── processor_config.json
│           ├── rng_state.pth
│           ├── scheduler.pt
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── trainer_state.json
│           └── training_args.bin
├── data
│   ├── echobot
│   │   ├── test.jsonl
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── overfit
│       └── train.jsonl
├── notebooks
│   ├── compare_results.ipynb
│   ├── download_models.ipynb
│   ├── fine-tune.ipynb
│   ├── qwen3.5-0.8b.ipynb
│   ├── qwen3.5-27b.ipynb
│   └── qwen3.5-9b.ipynb
├── notes
│   └── qwen3.5-eb-guardian-results.md
├── pyproject.toml
├── results
│   ├── eb-qwen3.5-9b-lora-522.json
│   ├── plots
│   │   ├── batch_sz_test.png
│   │   ├── qwen3.5-base-metrics.png
│   │   └── qwen3.5-fine-tuned-metrics.png
├── src
│   └── evaluation
│   │   ├── __init__.py
│   │   ├── binary_classifier.py
│   │   └── plots.py
└── uv.lock
```

## Serving a Fine-Tuned Model

You can serve a fine-tuned model using vLLM with the following command:
```bash
vllm serve <path_to_base_model> \
    --served-model-name qwen3.5-9b \
    --port 8001 \
    --max-model-len 4096 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-lora \
    --lora-modules <served_lora_model_name>=<path_to_adapters>
```

`path_to_base_model` refers to the model used for the fine-tuning process. For models downloaded using Hugging Face hub, point to a specific snapshot like `/home/user/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B/snapshots/005429cee5cb648998cf2b70eebdd83175989c9a`. 

`served_lora_model_name` is the name you'll use to query the fine-tuned model through LiteLLM or plain `curl` requests. 

`path_to_adapter` is the filepath to the fine-tuned weights Unsloth produces when saving models like below:
```python
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
```

For more information on how DavisAI serves and queries local models, check out the documentation on [[vLLM]] and [[LiteLLM]].
