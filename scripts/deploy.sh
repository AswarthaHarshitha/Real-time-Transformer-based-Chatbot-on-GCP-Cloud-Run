#!/usr/bin/env bash
# Deploy helper: build image, push to Artifact Registry, deploy to Cloud Run
set -euo pipefail

PROJECT_ID=${PROJECT_ID:-your-gcp-project}
REGION=${REGION:-us-central1}
REPO=${REPO:-chatbot-repo}
IMAGE_NAME=${IMAGE_NAME:-chatbot}
TAG=${TAG:-latest}
SERVICE_NAME=${SERVICE_NAME:-realtime-chatbot}

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

echo "Building container..."
docker build -t ${IMAGE_URI} .

echo "Pushing to Artifact Registry..."
docker push ${IMAGE_URI}

echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_URI} \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --memory=1Gi \
  --concurrency=1 \
  --max-instances=10

echo "Deployment complete"
