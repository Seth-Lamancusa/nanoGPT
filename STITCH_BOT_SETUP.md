# nanoGPT Setup for Stitch Bot

This guide explains how to set up nanoGPT for use with the Stitch bots service.

## Overview

The Stitch bots service includes a nanoGPT bot wrapper that connects to a self-hosted nanoGPT inference server. This repository contains the training and inference code needed to run that server.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the core dependencies:
```bash
pip install torch numpy transformers tiktoken fastapi uvicorn pydantic
```

### 2. Start Inference Server with GPT-2 Weights

The easiest way to get started is to use pretrained GPT-2 weights from HuggingFace:

```bash
python inference_server.py --init_from=gpt2 --port=8000
```

This will:
- Download GPT-2 (124M parameters) weights automatically
- Start an HTTP server on port 8000
- Expose a `/generate` endpoint compatible with the Stitch bot wrapper

### 3. Configure Stitch Bot

In your `stitch-bots/.env` file, add:

```bash
NANOGPT_API_URL=http://localhost:8000
NANOGPT_MAX_TOKENS=256
NANOGPT_TEMPERATURE=0.8
NANOGPT_TOP_K=200
```

Optional: Set an API key for authentication:
```bash
NANOGPT_API_KEY=your-secret-key-here
```

Then start the inference server with the key:
```bash
python inference_server.py --init_from=gpt2 --port=8000 --api_key=your-secret-key-here
```

## Model Options

### Using Different GPT-2 Sizes

- `gpt2` (124M) - Smallest, fastest
- `gpt2-medium` (350M) - Medium size
- `gpt2-large` (774M) - Large
- `gpt2-xl` (1.6B) - Extra large

Example:
```bash
python inference_server.py --init_from=gpt2-medium --port=8000
```

### Using a Trained Checkpoint

If you've trained your own model:

```bash
python inference_server.py --init_from=resume --out_dir=out-shakespeare-char --port=8000
```

## Training

### Quick Training Example (Shakespeare Character-Level)

Train a small model on Shakespeare:

```bash
# Prepare data
python data/shakespeare_char/prepare.py

# Train (adjust for your GPU/CPU)
python train.py config/train_shakespeare_char.py

# Start inference server with trained model
python inference_server.py --init_from=resume --out_dir=out-shakespeare-char --port=8000
```

### Training with GPT-2 Weights (Fine-tuning)

Fine-tune GPT-2 on custom data:

```bash
# Prepare your dataset (see data/shakespeare/prepare.py for example)
python data/shakespeare/prepare.py

# Fine-tune
python train.py config/finetune_shakespeare.py

# Use fine-tuned model
python inference_server.py --init_from=resume --out_dir=out-shakespeare --port=8000
```

### Training from Scratch (Reproducing GPT-2)

For serious training on OpenWebText:

```bash
# Prepare OpenWebText dataset (this downloads ~40GB)
python data/openwebtext/prepare.py

# Train (requires 8x A100 GPUs, takes ~4 days)
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

## API Endpoints

### POST /generate

Generate text completion.

**Request:**
```json
{
  "prompt": "Hello, how are you?",
  "max_tokens": 256,
  "temperature": 0.8,
  "top_k": 200
}
```

**Response:**
```json
{
  "text": "I'm doing well, thank you for asking..."
}
```

### GET /

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "gpt2",
  "device": "cuda",
  "dtype": "bfloat16"
}
```

## Device Options

- `--device=cuda` - Use GPU (default if CUDA available)
- `--device=cpu` - Use CPU
- `--device=mps` - Use Apple Silicon GPU (Mac)

Example:
```bash
python inference_server.py --init_from=gpt2 --device=cpu --port=8000
```

## Performance Tips

1. **Use GPU**: Much faster than CPU
2. **Compile model**: Add `--compile` flag for PyTorch 2.0 compilation (faster)
3. **Use bfloat16**: Automatic on CUDA GPUs that support it
4. **Smaller models**: `gpt2` is fastest, `gpt2-xl` is slowest

Example with compilation:
```bash
python inference_server.py --init_from=gpt2 --compile --port=8000
```

## Troubleshooting

### Out of Memory

- Use a smaller model (`gpt2` instead of `gpt2-xl`)
- Reduce `max_tokens` in requests
- Use CPU instead of GPU

### Model Loading Errors

- Ensure `transformers` is installed: `pip install transformers`
- Check internet connection (needed to download GPT-2 weights)
- Verify checkpoint path if using `--init_from=resume`

### Slow Generation

- Use GPU if available
- Enable model compilation with `--compile`
- Use smaller model or reduce `max_tokens`

## Integration with Stitch Bots

The inference server is designed to work seamlessly with the Stitch bots service. The bot wrapper (`stitch-bots/src/wrappers/ngpt/client.py`) automatically formats requests and handles responses.

See `stitch-bots/README.md` for more information about the bot service.

