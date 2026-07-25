# Every variable has a default except project_id, so `terraform plan` runs with
# one input and a reader can see the intended shape without a tfvars file.

variable "project_id" {
  description = "GCP project id. No default on purpose — naming someone else's project is not a sensible default."
  type        = string
}

variable "region" {
  description = <<-EOT
    Deployment region.

    me-central1 is Doha, Qatar — the same country as the corpus (Qatari Labour
    Law No. 14 of 2004) and the market this artifact targets, and it carries
    Cloud Run, Cloud SQL, Artifact Registry and Secret Manager. The alternatives
    were me-central2 (Dammam, Saudi Arabia) and me-west1 (Tel Aviv); Doha wins on
    data residency for a Qatari legal corpus and on latency to a Doha demo.

    Note: me-central1 is not a Google "tier 1" region, so list prices are higher
    than the us-central1 figures the cost comments in this stack are computed
    from. Those comments say so where it matters.
  EOT
  type        = string
  default     = "me-central1"
}

variable "service_name" {
  description = "Cloud Run service name; also the Artifact Registry repository id."
  type        = string
  default     = "arabic-rag"
}

variable "image_tag" {
  description = <<-EOT
    Tag of the container image in Artifact Registry.

    "latest" is a deliberate demo-grade default: this stack is applied minutes
    after a `docker push` and destroyed the same day, so an immutable digest buys
    nothing here. For anything that outlives an afternoon, pass a git sha.
  EOT
  type        = string
  default     = "latest"
}

variable "db_tier" {
  description = <<-EOT
    Cloud SQL machine type. db-f1-micro (shared vCPU, 0.6 GB RAM) is the smallest
    tier Cloud SQL offers and is genuinely enough for this corpus: 233 chunks x
    1024 dims is ~1 MB of vectors, and the HNSW index over it is smaller still.

    Upgrade path, in order: db-g1-small (1.7 GB) if the working set grows past a
    few thousand chunks, then db-custom-1-3840 (1 dedicated vCPU, 3.75 GB) which
    is where Cloud SQL's dedicated-core pricing starts. Shared-core tiers require
    edition = ENTERPRISE (set in sql.tf); ENTERPRISE_PLUS rejects them.
  EOT
  type        = string
  default     = "db-f1-micro"
}

variable "db_disk_gb" {
  description = "Cloud SQL SSD size in GB. 10 is the Cloud SQL minimum and ~100x the corpus."
  type        = number
  default     = 10
}

variable "max_instances" {
  description = "Cloud Run ceiling. 2 for a demo: enough to survive one stuck request, low enough that a loose script cannot fan out."
  type        = number
  default     = 2
}

variable "container_cpu" {
  description = <<-EOT
    Cloud Run vCPU. 4 because the service loads BAAI/bge-m3 (~2.2 GB) for query
    embedding and BAAI/bge-reranker-v2-m3 (~2.2 GB) for reranking, and both run
    on CPU here — there is no MPS and no GPU on Cloud Run. The measured 914 ms
    p95 rerank stage in docs/latency-budget.md was on Apple Silicon MPS; on CPU
    it will be worse, by an amount nobody in this repo has measured.
  EOT
  type        = string
  default     = "4"
}

variable "container_memory" {
  description = "Cloud Run memory. 8Gi holds both transformer models plus the Python process with room to spare."
  type        = string
  default     = "8Gi"
}

variable "rerank_enabled" {
  description = <<-EOT
    Sets RERANK_ENABLED on the service. Leave true only if the pushed image
    installs requirements-models.txt — the committed Dockerfile installs
    requirements.txt only, which has no torch, so an image built from it as-is
    cannot embed or rerank. See README.md, "Build and push the image".
  EOT
  type        = bool
  default     = true
}

variable "daily_spend_cap_usd" {
  description = "DAILY_SPEND_CAP_USD for the app's own kill-switch. Independent of any GCP budget alert."
  type        = number
  default     = 5.0
}

variable "allow_unauthenticated" {
  description = <<-EOT
    Grant roles/run.invoker to allUsers so a demo URL can be opened in a browser.

    True by default because that is what this stack is for. What stands between
    an open URL and a bill: the app's own daily USD cap (daily_spend_cap_usd,
    503 past it), Cloud Run max_instances, and the fact that the stack is meant
    to be destroyed the same day. There is no auth on /ask. Set false and use
    `gcloud run services proxy` if the URL will be up longer than a demo.
  EOT
  type        = bool
  default     = true
}

variable "otel_exporter_otlp_endpoint" {
  description = <<-EOT
    Value for OTEL_EXPORTER_OTLP_ENDPOINT. Empty disables app-level span export
    entirely (app/observability/tracing.py installs no provider without it), so
    tracing costs nothing by default.

    Empty is the default because the app's OTLP/gRPC exporter cannot authenticate
    to Google directly — see the Cloud Trace note in README.md. Cloud Run's own
    request spans still reach Cloud Trace regardless of this value.
  EOT
  type        = string
  default     = ""
}
