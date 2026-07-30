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

variable "generation_providers" {
  description = <<-EOT
    Sets PROVIDERS: the generation adapters in failover order, first is primary.
    "huggingface" is the deployed default because its key is the one this project
    actually has; the Anthropic and Gemini secrets exist and hold placeholders,
    so listing them here without replacing those placeholders buys a 401 per
    request instead of a failover.
  EOT
  type        = string
  default     = "huggingface"
}

variable "hf_model" {
  description = <<-EOT
    Sets HF_MODEL: the model id the Hugging Face router is asked for. Must be one
    the router serves and one that answers in Arabic — Qwen2.5-72B-Instruct and
    CohereLabs/aya-expanse-32b both do. The price used for the spend cap is
    HF_PRICE_*_USD_PER_MILLION in app/config.py, not a published table; see the
    note there before trusting the cost figures on /stats.
  EOT
  type        = string
  default     = "Qwen/Qwen2.5-72B-Instruct"
}

variable "rerank_enabled" {
  description = <<-EOT
    Sets RERANK_ENABLED on the service. Deployed false: the image has torch, but
    Cloud Run has no GPU and the cross-encoder measured 3972 ms p95 there against
    a 1200 ms allocation (docs/latency-budget.md), and the weights are not baked
    into the image. True costs a 2.2 GB download on the first request of every
    cold instance plus ~4 s per question, and buys recall@10 0.946 -> 0.965.
  EOT
  type        = bool
  default     = false
}

variable "daily_spend_cap_usd" {
  description = <<-EOT
    DAILY_SPEND_CAP_USD for the app's own kill-switch. Independent of any GCP
    budget alert.

    1.0 for a public URL, not 5.0: the cap is per-process, so the real ceiling is
    this number times max_instances. At 1.0 x 2 the worst an open /ask can cost
    in a day is $2 of generation. Raise it for a private demo where nobody is
    fuzzing the endpoint.
  EOT
  type        = number
  default     = 1.0
}

variable "ask_rate_limit_per_minute" {
  description = <<-EOT
    ASK_RATE_LIMIT_PER_MINUTE — sliding-window requests per client IP on /ask.
    0 disables the limiter.

    10, well below the app's own default of 60, because a human evaluating the
    demo asks a handful of questions and a script does not. Bounds how fast the
    spend cap above can be reached, and it is per-instance and per-IP, so it
    slows abuse rather than preventing it. See the README's security posture.
  EOT
  type        = number
  default     = 10
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

variable "enable_github_oidc" {
  description = <<-EOT
    Create the Workload Identity pool, provider and deployer service account that
    let GitHub Actions push images and roll Cloud Run revisions without a
    service-account key (github_oidc.tf). False makes that file inert.
  EOT
  type        = bool
  default     = true
}

variable "github_repo" {
  description = <<-EOT
    "owner/repo" allowed to federate in. Appears in both the provider's
    attribute_condition and the impersonation binding, because either one alone
    being wrong is a hole: without the condition any repository on GitHub can mint
    a token for the pool.
  EOT
  type        = string
  default     = "GamalTawaf/arabic-rag"
}

variable "run_subnet_cidr" {
  description = <<-EOT
    Range for the subnet Cloud Run egresses from (network.tf). Direct VPC egress
    takes one address per instance, so this bounds scaling: /24 leaves room, the
    /28 minimum does not.
  EOT
  type        = string
  default     = "10.8.0.0/24"
}

variable "db_public_ip" {
  description = <<-EOT
    Give Cloud SQL a public endpoint in addition to its private address.

    False, which is the point of the VPC: the database has no internet-facing
    address at all. There is still no authorized_networks block, so even flipping
    this to true admits nobody by itself — it exists because a private-only
    instance cannot be reached from a laptop, and `alembic upgrade head` plus
    `python -m ingestion ingest` have to run from somewhere. Flip it, run them
    through the Auth Proxy, flip it back, and the window during which a public
    address exists is minutes rather than the life of the stack.
  EOT
  type        = bool
  default     = false
}
