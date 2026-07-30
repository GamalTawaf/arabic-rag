#!/bin/sh
set -e

## Load the corpus into a database no laptop can reach.
##
## docker/entrypoint.sh applies the schema, but nothing puts the five documents in
## data/corpus/ into it, and with ipv4_enabled = false (terraform/sql.tf) there is
## no route from here to run `python -m ingestion ingest` locally. The image that
## serves /ask already carries the corpus, the ingestion CLI and the baked bge-m3
## weights, so run it there as a one-off job:
##
##   PROJECT=$(gcloud config get-value project)
##   REGION=me-central1
##   gcloud run jobs create arabic-rag-load-corpus \
##     --region "$REGION" \
##     --image "$REGION-docker.pkg.dev/$PROJECT/arabic-rag/arabic-rag:latest" \
##     --command /app/scripts/load-corpus.sh \
##     --service-account "arabic-rag-run@$PROJECT.iam.gserviceaccount.com" \
##     --set-secrets DATABASE_URL=arabic-rag-database-url:latest \
##     --set-cloudsql-instances "$PROJECT:$REGION:arabic-rag-pg" \
##     --network arabic-rag-vpc --subnet arabic-rag-run \
##     --vpc-egress private-ranges-only \
##     --cpu 4 --memory 8Gi --task-timeout 30m --max-retries 1 \
##     --execute-now --wait
##
##   gcloud run jobs delete arabic-rag-load-corpus --region "$REGION"
##
## Same service account, same secret, same VPC and same Cloud SQL socket as the
## service (terraform/run.tf) — a job with narrower access fails at connect time,
## and one with wider access is a second thing to audit. The 8Gi/4cpu is the
## embedder's requirement, not the ingest's: backfill loads bge-m3 on CPU.
##
## Deliberately not a terraform resource. A resource whose whole purpose is to run
## one command once is a resource someone maintains forever, and `terraform
## destroy` then has one more thing to trip over. Deleting the job after it
## succeeds costs one command and leaves nothing behind.
##
## Safe to re-run: chunk ids are a hash of the text and every row is upserted
## ON CONFLICT DO UPDATE, so a second run rewrites the same rows.

alembic upgrade head                      # no-op when the service already migrated
python -m ingestion ingest                # data/corpus/ -> chunks
python -m ingestion backfill --model bge  # the vector column /ask queries
python -m ingestion stats                 # coverage 1.0, or the corpus is not loaded
