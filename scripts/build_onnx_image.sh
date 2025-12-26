#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-chatbot-onnx:local}
DOCKERFILE=${DOCKERFILE:-Dockerfile.onnx}

echo "Building ONNX Docker image $IMAGE using $DOCKERFILE"
docker build -f "$DOCKERFILE" -t "$IMAGE" .

echo "Run container locally (background)"
docker run -d --rm -p 8080:8080 --name chatbot_onnx "$IMAGE"

echo "Waiting for server to come up (sleep 3s)"
sleep 3

echo "Run quick measure"
python scripts/measure_latency.py

echo "Stopping container"
docker stop chatbot_onnx || true