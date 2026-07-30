output "service_url" {
  description = "Cloud Run HTTPS URL. POST /ask, GET /stats, GET /health."
  value       = google_cloud_run_v2_service.rag.uri
}

output "database_connection_name" {
  description = "Cloud SQL connection name (project:region:instance) — the argument to `cloud-sql-proxy` and the path under /cloudsql inside the container."
  value       = google_sql_database_instance.pg.connection_name
}

output "ingest_api_key" {
  description = "The x-api-key POST /ingest requires. Read with: terraform output -raw ingest_api_key"
  value       = random_password.ingest_key.result
  sensitive   = true
}

output "ingest_topic" {
  description = "Pub/Sub topic that drives asynchronous ingestion. Publish with: gcloud pubsub topics publish <topic> --message '{...}'"
  value       = google_pubsub_topic.ingest.id
}

output "ingest_dead_letter_topic" {
  description = "Where a message lands after 5 failed deliveries. Read it via the -dlq-pull subscription."
  value       = google_pubsub_topic.ingest_dlq.id
}

output "image_repository" {
  description = "Artifact Registry image path Cloud Run pulls. Build and push here before the first apply."
  value       = local.image
}

output "destroy_reminder" {
  description = "The point of this whole directory."
  value       = <<-EOT

    ┌──────────────────────────────────────────────────────────────────────┐
    │  THIS STACK IS EPHEMERAL. DESTROY IT WHEN THE DEMO ENDS.             │
    │                                                                      │
    │      cd terraform && terraform destroy                               │
    │                                                                      │
    │  Left up, Cloud SQL alone bills ~$0.017/hour = ~$0.41/day =          │
    │  ~$12.60/month whether or not anyone asks it a question, and a live  │
    │  Cloud Run instance adds ~$0.32/hour while it is warm. Destroyed,    │
    │  the stack costs ~$0.50/month: the container image in Artifact       │
    │  Registry, which is worth keeping so the next demo is a `terraform   │
    │  apply` and not a 5 GB docker push.                                  │
    │                                                                      │
    │  Destroy does NOT delete: the enabled APIs (deliberate, see          │
    │  apis.tf) or anything outside this state file. It DOES delete the    │
    │  database and every chunk in it — the corpus is committed under      │
    │  data/corpus/ and `python -m ingestion ingest` rebuilds it.          │
    └──────────────────────────────────────────────────────────────────────┘
  EOT
}

output "workload_identity_provider" {
  description = "Value for the deploy workflow's google-github-actions/auth step."
  value       = var.enable_github_oidc ? google_iam_workload_identity_pool_provider.github[0].name : null
}

output "deployer_service_account" {
  description = "Service account GitHub Actions impersonates. No infra rights."
  value       = var.enable_github_oidc ? google_service_account.deployer[0].email : null
}
