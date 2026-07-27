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
      # trade-off: public IP with an EMPTY authorized_networks list, not a private
      # IP behind Private Service Access.
      #
      # This is a judgement call and it is the one place this stack departs from
      # the obvious "production" answer, so here is the reasoning in full:
      #
      # Reachability is already closed. With no authorized networks, no address
      # on the internet can open a socket to this instance. The only ingress is
      # the Cloud SQL Auth Proxy, which Cloud Run mounts as a unix socket
      # (run.tf) and which authenticates with IAM (roles/cloudsql.client, granted
      # to exactly one service account in iam.tf) over mutual TLS. SSL is
      # required below, so even a future authorized network could not downgrade.
      #
      # What private IP would add: defence in depth if a Cloud SQL authorized-
      # network entry were ever added carelessly.
      #
      # What it would cost: a VPC, a reserved /16 for Private Service Access, and
      # a google_service_networking_connection. That last resource is a known
      # destroy hazard — the peering it creates cannot be deleted while the
      # producer connection exists, and the VPC cannot be deleted while the
      # peering exists, so `terraform destroy` on this shape routinely ends in a
      # hand-cleanup. In a stack whose whole promise is a clean destroy, and
      # which I cannot test because there are no credentials here, shipping a
      # destroy path I believe is broken is worse than shipping a public IP with
      # nothing allowed through it.
      #
      # Upgrade path if this ever outlives a demo: add a VPC + PSA range +
      # service networking connection, set ipv4_enabled = false and
      # private_network, add Direct VPC egress to the Cloud Run template, and
      # budget an extra ~10 minutes on apply and a manual peering delete on
      # destroy.
      #
      # There is deliberately no `authorized_networks` block below. In this
      # provider it is a repeatable block, not a list argument, so "none" is
      # written by saying nothing — which is also the only way to say it that a
      # future edit cannot weaken by appending one more CIDR to an existing list.
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }

    insights_config {
      # Query Insights is free and is the only way to see whether the HNSW scan
      # or the FTS scan is the slow half once this runs on something other than
      # a laptop.
      query_insights_enabled = true
    }
  }

  depends_on = [google_project_service.apis]
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
