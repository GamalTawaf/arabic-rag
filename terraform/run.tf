# The Cloud Run service.

locals {
  # Plain (non-secret) environment. Names are app/config.py fields uppercased —
  # pydantic-settings does the mapping. Everything sensitive goes through
  # secret_key_ref below instead.
  plain_env = merge(
    {
      ENV = "production"
      # Which generation adapters, in failover order. Only the ones whose key
      # secret holds a real value will answer; the rest fail over past.
      PROVIDERS                 = var.generation_providers
      HF_MODEL                  = var.hf_model
      RERANK_ENABLED            = var.rerank_enabled ? "true" : "false"
      DAILY_SPEND_CAP_USD       = tostring(var.daily_spend_cap_usd)
      ASK_RATE_LIMIT_PER_MINUTE = tostring(var.ask_rate_limit_per_minute)

      # Set even when no exporter is configured: it is what a span ends up
      # labelled as if one ever is, and it costs nothing to be right in advance.
      OTEL_SERVICE_NAME = var.service_name

      # Cloud Logging parses stdout/stderr as JSON when it is JSON, and reads
      # `severity` as the level. GCP_PROJECT is what lets each line name its
      # trace, so a log entry in the console links to the span tree it came from.
      LOG_JSON    = "true"
      GCP_PROJECT = var.project_id
    },
    var.otel_exporter_otlp_endpoint == "" ? {} : {
      OTEL_EXPORTER_OTLP_ENDPOINT = var.otel_exporter_otlp_endpoint
    },
  )
}

resource "google_cloud_run_v2_service" "rag" {
  name     = var.service_name
  location = var.region

  # Ephemeral: destroy must not need a flag flipped first.
  deletion_protection = false

  # Public ingress is required, and not only for the demo URL — Pub/Sub push
  # arrives from the internet, not from inside the VPC, so INGRESS_TRAFFIC_ALL is
  # the only setting under which the /ingest subscription in pubsub.tf can
  # deliver at all. Authorization is IAM's job (iam.tf), not the ingress rule's.
  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    # Not the default compute service account. See iam.tf.
    service_account = google_service_account.runtime.email

    # gen2: full Linux syscall compatibility and mmap, which is what torch and
    # the sentence-transformers model loader assume. gen1's emulated sandbox is
    # where "works on my laptop, ImportErrors on Cloud Run" comes from.
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

    # 300 s. The measured p95 for the retrieval half of /ask is ~975 ms on Apple
    # Silicon MPS (docs/latency-budget.md); on Cloud Run's CPU the reranker will
    # be slower by an unmeasured factor, and the first request of a cold instance
    # additionally downloads and loads ~4.4 GB of model weights. 300 s is slack
    # for that, not a target.
    timeout = "300s"

    # Low on purpose: two transformer models per instance means concurrency is
    # bounded by memory and CPU, not by the event loop.
    max_instance_request_concurrency = 4

    scaling {
      # The whole cost story. Nothing is running, and nothing is billed, between
      # the last request and the next one.
      min_instance_count = 0
      max_instance_count = var.max_instances
    }

    # Direct VPC egress, not a Serverless VPC Access connector: the connector is
    # a pair of billed e2-micro instances that would cost more than everything
    # else in this stack combined and sit idle between demos. Direct egress gives
    # each instance an address in the run subnet and costs nothing extra.
    #
    # PRIVATE_RANGES_ONLY, not ALL_TRAFFIC: RFC1918 destinations go through the
    # VPC so the database is reachable, and everything public — the Hugging Face
    # router, Hugging Face model downloads if the image ever needs one — keeps
    # taking the default internet path. ALL_TRAFFIC would need Cloud NAT to reach
    # any of that, which is another billed resource to remember to destroy.
    vpc_access {
      egress = "PRIVATE_RANGES_ONLY"

      # Names, not `.id`. The API stores the short name, so a full resource path
      # here means every subsequent plan shows a change that never converges.
      network_interfaces {
        network    = google_compute_network.vpc.name
        subnetwork = google_compute_subnetwork.run.name
      }
    }

    # Cloud SQL Auth Proxy, mounted as a unix socket. No password on the wire, no
    # IP allowlist, IAM-authenticated via roles/cloudsql.client. DATABASE_URL
    # (secrets.tf) points at /cloudsql/<connection_name>.
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.pg.connection_name]
      }
    }

    # Named because the collector sidecar (prometheus.tf) declares a startup
    # dependency on it by name. A single-container service does not need a name;
    # a multi-container one does.
    containers {
      name  = "app"
      image = local.image

      ports {
        container_port = 8000 # matches EXPOSE / uvicorn in config/Dockerfile
      }

      resources {
        limits = {
          cpu    = var.container_cpu
          memory = var.container_memory
        }

        # CPU throttling off: CPU is allocated for the instance's whole lifetime,
        # not only while a request is in flight.
        #
        # The cost of this choice, said out loud, because this repo is partly
        # about cost: at us-central1 list prices, always-allocated CPU bills
        # ~$0.000018/vCPU-second and ~$0.000002/GiB-second, so a live 4 vCPU /
        # 8 GiB instance is ~$0.32/hour — and Cloud Run keeps an idle instance
        # around for roughly 15 minutes after the last request before scaling to
        # zero. A one-hour demo with sporadic traffic therefore costs on the
        # order of $0.15-$0.30, several times the database underneath it.
        #
        # It is still the right setting here: this container lazily loads ~4.4 GB
        # of transformer weights on first use (app/deps.py builds nothing at
        # import), and with cpu_idle = true every millisecond outside an active
        # request runs at roughly 5% of a core, which turns model loading and
        # torch's threadpool warm-up into a cold-start cliff.
        #
        # Flip to cpu_idle = true if the demo is long and idle, or if the image
        # ever drops the local models for API embeddings — then you pay only for
        # request-seconds and this line is pure waste.
        cpu_idle = false

        # Full CPU during startup regardless of the above. Free.
        startup_cpu_boost = true
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

      # Secrets by reference. The container reads them as env vars, but the
      # values live in Secret Manager and never appear in the service config.
      # "latest" is resolved at instance start, so replacing a secret version
      # takes effect on the next cold start without a redeploy.
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

      # /health is deliberately the cheapest route in the service: it touches no
      # model and no database (app/api/health.py; /health/db is the one that
      # talks to Postgres). So the probe measures "uvicorn is up", which is
      # exactly what a startup probe should measure — gating readiness on model
      # load would make every cold start look like a failed deploy.
      #
      # 5 s initial + 24 x 5 s = up to 125 s to come up, raised from 65 s when
      # migrations moved into the entrypoint: the port is now opened only after
      # `alembic upgrade head` returns, so image pull, the first Cloud SQL socket
      # connect and any DDL share this budget. A revision that would have come up
      # in 70 s used to be reported as failed to start.
      startup_probe {
        initial_delay_seconds = 5
        period_seconds        = 5
        timeout_seconds       = 3
        failure_threshold     = 24

        http_get {
          path = "/health"
          port = 8000
        }
      }
    }

    # The Managed Prometheus collector (prometheus.tf). No ports block: only the
    # ingress container gets one, and this one talks outward to Cloud Monitoring.
    # No resources block either — Google's own example sets none, and Cloud Run
    # splits the instance's allocation, so pinning a fraction here would take CPU
    # from the embedder to guarantee it to a scraper that idles.
    dynamic "containers" {
      for_each = var.enable_prometheus_sidecar ? [1] : []
      content {
        name  = "collector"
        image = var.gmp_sidecar_image

        # Start after the app and stop before it, so the last scrape happens while
        # there is still something to scrape.
        depends_on = ["app"]

        volume_mounts {
          name       = "gmp-config"
          mount_path = "/etc/rungmp"
        }
      }
    }

    dynamic "volumes" {
      for_each = var.enable_prometheus_sidecar ? [1] : []
      content {
        name = "gmp-config"
        secret {
          secret = google_secret_manager_secret.gmp_config[0].secret_id
          items {
            # The resolved version, not "latest". "latest" is a constant in the
            # revision template, so editing gmp_scrape_interval wrote a new secret
            # version and changed nothing Cloud Run can see: no new revision, and
            # running instances never re-resolve a mounted secret file. With
            # min_instances > 0 the old interval would survive indefinitely while
            # `terraform apply` reported success.
            version = google_secret_manager_secret_version.gmp_config[0].version
            path    = "config.yaml" # the sidecar reads /etc/rungmp/config.yaml
          }
        }
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_version.database_url,
    google_secret_manager_secret_version.llm_placeholder,
    google_project_iam_member.runtime_cloudsql_client,
    google_secret_manager_secret_iam_member.runtime_secret_accessor,
  ]
}
