# Managed Service for Prometheus, as a Cloud Run sidecar.
#
# The service already exports Prometheus text at /metrics (app/main.py mounts the
# OTel PrometheusMetricReader there). Locally, docker-compose.observability.yml
# points a prometheus container at it. On Cloud Run nothing was scraping it: an
# instance that scales to zero cannot be polled from outside, so the metrics
# existed and nobody read them.
#
# This is Google's sidecar collector (`cloud-run-gmp-sidecar`). It runs inside the
# same instance, scrapes localhost:8000/metrics, and writes to Cloud Monitoring,
# where the series are queryable with PromQL. Scraping from inside is the only
# shape that works with min_instances = 0 — the collector lives and dies with the
# instance it is measuring.
#
# What it costs: nothing extra in Cloud Run terms, because Cloud Run bills the
# instance, not the container, and the collector shares the 4 vCPU / 8 GiB the app
# already has. Cloud Monitoring bills ingested samples, so the scrape interval is
# the cost knob — 30 s over a handful of series is cents, 5 s is six times that
# for a graph nobody watches that closely.
#
# What it does NOT do: make anything public. Cloud Monitoring is IAM-gated, so
# these metrics are visible to project viewers only. A public dashboard is a
# separate problem — see the note at the end of terraform/README.md.
#
# Set enable_prometheus_sidecar = false and this file plus the collector container
# in run.tf disappear.

locals {
  gmp_count = var.enable_prometheus_sidecar ? 1 : 0

  # RunMonitoring: a Cloud Run-flavoured subset of the PodMonitoring CRD. The
  # sidecar reads it from /etc/rungmp/config.yaml. Without a config it scrapes
  # port 8080, which is not where this app listens — 8000 is (Dockerfile EXPOSE).
  #
  # targetLabels.metadata attaches the Cloud Run service and revision to every
  # series, which is what makes "did latency change after that deploy?" a query
  # rather than a guess.
  gmp_config = <<-EOT
    apiVersion: monitoring.googleapis.com/v1beta
    kind: RunMonitoring
    metadata:
      name: ${var.service_name}
    spec:
      endpoints:
      - port: 8000
        path: /metrics
        interval: ${var.gmp_scrape_interval}
      targetLabels:
        metadata:
        - service
        - revision
  EOT
}

# The config has to arrive as a mounted secret — that is the only delivery
# mechanism the sidecar supports. It holds no secret value; this is Secret Manager
# used as a file mount, which is worth knowing when you see it in the console.
resource "google_secret_manager_secret" "gmp_config" {
  count = local.gmp_count

  secret_id = "${var.service_name}-gmp-config"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "gmp_config" {
  count = local.gmp_count

  secret      = google_secret_manager_secret.gmp_config[0].id
  secret_data = local.gmp_config
}

# Read the config file. Bound to this one secret, like every other grant in iam.tf.
resource "google_secret_manager_secret_iam_member" "runtime_gmp_config" {
  count = local.gmp_count

  secret_id = google_secret_manager_secret.gmp_config[0].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

# Write the scraped series. Project-scoped because Cloud Monitoring has no
# finer resource to bind a writer to — the same reason cloudtrace.agent is
# project-scoped in iam.tf.
resource "google_project_iam_member" "runtime_metric_writer" {
  count = local.gmp_count

  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# The collector writes its own logs as the runtime service account. Cloud Run's
# app logs do not need this — the platform writes those — so this grant exists
# only for the sidecar, and goes away with it.
resource "google_project_iam_member" "runtime_log_writer" {
  count = local.gmp_count

  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}
