# Schema and corpus load, run from inside the VPC.
#
# The database has no public address (sql.tf), so nothing outside this network can
# run `alembic upgrade head` — which is the point. This job is the path in: the
# same image the service runs, the same runtime service account, the same Cloud
# SQL socket and the same secrets, executed on demand:
#
#   gcloud run jobs execute arabic-rag-migrate --region me-central1 --wait
#
# It is idempotent by construction. Alembic skips revisions already applied,
# ingestion upserts on a chunk id that is a pure function of the text, and
# backfill only embeds rows whose vector is missing — so re-running it after a
# failed half is safe, and that is what makes it usable as the only way in.

resource "google_cloud_run_v2_job" "migrate" {
  name     = "${var.service_name}-migrate"
  location = var.region

  deletion_protection = false

  template {
    template {
      service_account       = google_service_account.runtime.email
      execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

      # 30 minutes. The backfill embeds 233 chunks on CPU with no GPU and no MPS;
      # it is ~65 s on a laptop and nobody has measured it here, so the timeout is
      # slack rather than a target.
      timeout = "1800s"

      # One retry. A cold start that loses the race with IAM propagation is worth
      # retrying; a migration that fails on its own SQL will fail identically.
      max_retries = 1

      vpc_access {
        egress = "PRIVATE_RANGES_ONLY"

        network_interfaces {
          network    = google_compute_network.vpc.id
          subnetwork = google_compute_subnetwork.run.id
        }
      }

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.pg.connection_name]
        }
      }

      containers {
        image = local.image

        # Chained with && on purpose: an ingest against a schema that failed to
        # migrate would write nothing useful, and a backfill over no rows would
        # exit 0 and report success.
        command = ["/bin/sh", "-c"]
        args = [
          "alembic upgrade head && python -m ingestion ingest && python -m ingestion backfill --model bge",
        ]

        resources {
          limits = {
            # Same as the service: the backfill loads the same 2.3 GB embedder.
            cpu    = var.container_cpu
            memory = var.container_memory
          }
        }

        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }

        dynamic "env" {
          for_each = local.plain_env
          content {
            name  = env.key
            value = env.value
          }
        }

        dynamic "env" {
          for_each = local.app_secrets
          content {
            name = env.value
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.app[env.key].secret_id
                version = "latest"
              }
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_version.database_url,
    google_secret_manager_secret_version.llm_placeholder,
    google_project_iam_member.runtime_cloudsql_client,
    google_secret_manager_secret_iam_member.runtime_secret_accessor,
  ]
}
