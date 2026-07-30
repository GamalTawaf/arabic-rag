# Cloud SQL for PostgreSQL 17 — the pgvector store.
#
# ============================================================================
# WHAT THIS COSTS WHILE IT EXISTS  (the number that justifies "destroy after")
# ============================================================================
#
#   db-f1-micro, ENTERPRISE, zonal, no HA, no backups   ~$0.0150 / hour
#   10 GB PD_SSD  (10 GB x $0.17/GB-month / 730 h)      ~$0.0023 / hour
#                                                       ----------------
#                                                       ~$0.017 / hour
#                                                       ~$0.41 / day
#                                                       ~$12.60 / month
#
# It bills for existing, not for being used: an idle Cloud SQL instance costs the
# same as a busy one. Cloud Run at min_instances = 0 costs nothing when nobody is
# asking, Artifact Registry is ~$0.0007/hour, Pub/Sub at demo volume is inside
# the free tier, Secret Manager is under a cent an hour. So this instance is
# ~90% of the steady-state cost of the whole stack, and `terraform destroy`
# takes the stack from ~$12.60/month to $0.50/month (the image) — which is the
# entire reason this directory exists.
#
# Those are us-central1 list prices, arithmetic done here rather than measured:
# there is no billing account attached to this repo and nothing here has ever
# been applied. me-central1 (Doha) is not a tier-1 region and runs higher —
# budget maybe 15-25% more. Verify against cloud.google.com/sql/pricing before
# quoting these to anyone who is paying.
# ============================================================================

resource "random_password" "db" {
  length = 32
  # Alphanumeric only: this password is interpolated into a postgresql:// URL in
  # secrets.tf, and percent-encoding a generated password is a bug waiting for a
  # demo to happen.
  special = false
}

resource "google_sql_database_instance" "pg" {
  name             = "${var.service_name}-pg"
  database_version = "POSTGRES_17"
  region           = var.region

  # Ephemeral by design. The default is `true`, which turns `terraform destroy`
  # into an error message — exactly wrong for a stack whose selling point is
  # that it goes away. Both flags are needed: the provider-side one below and
  # the API-side one in settings.
  deletion_protection = false

  settings {
    tier              = var.db_tier
    edition           = "ENTERPRISE" # shared-core tiers are rejected by ENTERPRISE_PLUS
    availability_type = "ZONAL"      # REGIONAL is ~2x the price to survive a zone outage a demo will not see

    disk_type       = "PD_SSD"
    disk_size       = var.db_disk_gb
    disk_autoresize = false # a runaway ingest should fail loudly, not grow the bill

    deletion_protection_enabled = false

    backup_configuration {
      # No backups. The corpus is committed under data/corpus/ and reloaded by
      # `python -m ingestion ingest` in about a minute; paying to protect a
      # derived artifact that a script regenerates is theatre.
      enabled = false
    }

    ip_configuration {
      # No public endpoint. With a private IP the question "who can open a socket
      # to this database" is answered by routing rather than by an
      # authorized-networks list that a later edit can widen: the only path in is
      # from inside the VPC (network.tf), which is why Cloud Run gets Direct VPC
      # egress in run.tf.
      #
      # This replaces the earlier public-IP-with-nothing-allowed shape. That was
      # defensible on reachability grounds — an empty authorized_networks list
      # admits nobody — but it depended on a list staying empty forever, and
      # defence in depth should not rest on a future edit being careful.
      #
      # The price, and it is a real one: the peering in network.tf makes destroy
      # messier, and a laptop can no longer reach the database at all. Migrations
      # and the corpus load happen from inside the VPC, or through a temporary
      # public endpoint — see db_public_ip below and README.md.
      ipv4_enabled = var.db_public_ip

      private_network = google_compute_network.vpc.id

      # Lets the instance be reached over Private Service Connect paths from
      # Google-managed services (the Cloud Run Cloud SQL connector among them)
      # without a public address.
      enable_private_path_for_google_cloud_services = true

      # Still required even with no public IP: private does not mean plaintext,
      # and anything inside the VPC is a peer, not a trusted one.
      ssl_mode = "ENCRYPTED_ONLY"
    }

    insights_config {
      # Query Insights is free and is the only way to see whether the HNSW scan
      # or the FTS scan is the slow half once this runs on something other than
      # a laptop.
      query_insights_enabled = true
    }
  }

  # The peering must exist before an instance can be given a private address.
  depends_on = [
    google_project_service.apis,
    google_service_networking_connection.psa,
  ]
}

resource "google_sql_database" "rag" {
  name     = "rag_db"
  instance = google_sql_database_instance.pg.name
}

resource "google_sql_user" "app" {
  name     = "rag_user"
  instance = google_sql_database_instance.pg.name
  password = random_password.db.result

  # Cloud SQL grants API-created Postgres users the cloudsqlsuperuser role, which
  # is what lets `alembic upgrade head` run the `CREATE EXTENSION IF NOT EXISTS
  # vector` in alembic/versions/0001_initial.py. That migration is how pgvector
  # gets enabled — there is no Terraform resource and no database flag for it;
  # the extension ships with Cloud SQL Postgres and has to be created inside the
  # database by someone holding this role. See README.md, "What you must do by
  # hand".
}
