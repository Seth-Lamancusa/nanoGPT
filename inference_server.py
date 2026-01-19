#!/usr/bin/env python3
"""
nanoGPT Inference Server

A simple HTTP API server for nanoGPT text generation.
Compatible with the stitch-bots nanoGPT wrapper.

Usage:
    python inference_server.py --init_from=gpt2 --port=8000
    python inference_server.py --init_from=resume --out_dir=out-shakespeare-char --port=8000
"""

import os
import pickle
import argparse
from contextlib import nullcontext
from typing import Optional, Dict, Any

import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default configuration
init_from = 'gpt2'  # 'resume' or 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
out_dir = 'out'  # ignored if init_from is not 'resume'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False
port = 8000
host = '0.0.0.0'
api_key: Optional[str] = None  # Set via NANOGPT_API_KEY env var

# Parse command line args
parser = argparse.ArgumentParser(description='nanoGPT Inference Server')
parser.add_argument('--init_from', type=str, default=init_from, help='Model to load: resume, gpt2, gpt2-medium, etc.')
parser.add_argument('--out_dir', type=str, default=out_dir, help='Checkpoint directory (if init_from=resume)')
parser.add_argument('--device', type=str, default=device, help='Device: cuda, cpu, mps')
parser.add_argument('--dtype', type=str, default=dtype, help='Data type: float32, float16, bfloat16')
parser.add_argument('--compile', action='store_true', help='Compile model with torch.compile')
parser.add_argument('--port', type=int, default=port, help='Server port')
parser.add_argument('--host', type=str, default=host, help='Server host')
parser.add_argument('--api_key', type=str, default=None, help='Optional API key for authentication')
args = parser.parse_args()

init_from = args.init_from
out_dir = args.out_dir
device = args.device
dtype = args.dtype
compile_model = args.compile
port = args.port
host = args.host
api_key = args.api_key or os.getenv('NANOGPT_API_KEY')

# -----------------------------------------------------------------------------
# Initialize model
print(f"Loading model from: {init_from}")
print(f"Device: {device}, dtype: {dtype}")

torch.manual_seed(1337)
if 'cuda' in device:
    torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.eval()
model.to(device)
if compile_model:
    print("Compiling model...")
    model = torch.compile(model)

# Load tokenizer
load_meta = False
if init_from == 'resume':
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("Using GPT-2 tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

print(f"Model loaded successfully!")

# -----------------------------------------------------------------------------
# FastAPI app
app = FastAPI(title="nanoGPT Inference Server")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: Optional[int] = None


class GenerateResponse(BaseModel):
    text: str


def verify_api_key(request_api_key: Optional[str]) -> bool:
    """Verify API key if one is configured."""
    if api_key is None:
        return True  # No API key required
    return request_api_key == api_key


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": init_from,
        "device": device,
        "dtype": dtype
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, authorization: Optional[str] = None):
    """
    Generate text completion.
    
    Compatible with stitch-bots nanoGPT wrapper.
    """
    # Verify API key if configured
    if api_key:
        # Extract Bearer token from Authorization header
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.replace("Bearer ", "").strip()
        if token != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Encode prompt
    try:
        start_ids = encode(request.prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error encoding prompt: {e}")
    
    # Generate
    try:
        with torch.no_grad():
            with ctx:
                # Use top_k if provided, otherwise use default (200)
                top_k = request.top_k if request.top_k is not None else 200
                y = model.generate(
                    x,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=top_k
                )
                generated_text = decode(y[0].tolist())
                # Remove the original prompt from the output
                if generated_text.startswith(request.prompt):
                    generated_text = generated_text[len(request.prompt):].strip()
                
                return GenerateResponse(text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")


if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting nanoGPT inference server on {host}:{port}")
    print(f"   Model: {init_from}")
    print(f"   Device: {device}")
    print(f"   API endpoint: http://{host}:{port}/generate")
    if api_key:
        print(f"   API key: {'*' * len(api_key)}")
    print()
    uvicorn.run(app, host=host, port=port)

