#!/usr/bin/env python3
"""
Quick test script for the inference server.
Run this after starting inference_server.py
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint."""
    response = requests.get(f"{API_URL}/")
    print("Health check:", json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_generate():
    """Test text generation."""
    payload = {
        "prompt": "The meaning of life is",
        "max_tokens": 50,
        "temperature": 0.8,
        "top_k": 200
    }
    response = requests.post(f"{API_URL}/generate", json=payload)
    if response.status_code == 200:
        result = response.json()
        print("\nGenerated text:")
        print(result["text"])
        return True
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    print("Testing nanoGPT inference server...")
    print(f"Server URL: {API_URL}\n")
    
    if test_health():
        print("\n✓ Health check passed")
    else:
        print("\n✗ Health check failed")
        exit(1)
    
    if test_generate():
        print("\n✓ Generation test passed")
    else:
        print("\n✗ Generation test failed")
        exit(1)
    
    print("\n✅ All tests passed!")

