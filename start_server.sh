#!/bin/bash
# Quick start script for nanoGPT inference server

# Default values
MODEL=${1:-gpt2}
PORT=${2:-8000}
DEVICE=${3:-cuda}

echo "Starting nanoGPT inference server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Device: $DEVICE"
echo ""

python inference_server.py --init_from="$MODEL" --port="$PORT" --device="$DEVICE"

