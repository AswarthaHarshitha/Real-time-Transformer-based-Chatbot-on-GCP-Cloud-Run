#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:-your-gcp-project}
REGION=${REGION:-us-central1}
BUCKET=${BUCKET:-your-gcs-bucket}
MODEL_DIR=${MODEL_DIR:-outputs/distilgpt2-dialogue}
DISPLAY_NAME=${DISPLAY_NAME:-distilgpt2-dialogue}

echo "Uploading model to gs://$BUCKET/models/$DISPLAY_NAME"
gsutil cp -r "$MODEL_DIR" "gs://$BUCKET/models/$DISPLAY_NAME"

echo "Uploading model to Vertex AI"
MODEL_ID=$(gcloud ai models upload --region=$REGION --display-name="$DISPLAY_NAME" --artifact-uri="gs://$BUCKET/models/$DISPLAY_NAME" --format="value(name)")
echo "Model ID: $MODEL_ID"

echo "Creating endpoint"
ENDPOINT_ID=$(gcloud ai endpoints create --region=$REGION --display-name="${DISPLAY_NAME}-endpoint" --format="value(name)")
echo "Endpoint ID: $ENDPOINT_ID"

echo "Deploying model to endpoint with 1x T4 (modify flags as needed)"
gcloud ai endpoints deploy-model "$ENDPOINT_ID" --model="$MODEL_ID" --region="$REGION" --machine-type=n1-standard-8 --accelerator=count=1,type=nvidia-tesla-t4 --format="value(deployedModel)"

echo "Deployed. Endpoint: $ENDPOINT_ID"
