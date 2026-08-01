# Secret Manager. The Cloud Run service reads all five through
# `value_source.secret_key_ref` (run.tf), never as plaintext env values, so the
# key text never appears in the service YAML, in `gcloud run services describe`,
# or in a Cloud Console screenshot taken during a demo.
#
# Cost: ~$0.06 per active secret version per month, plus $0.03 per 10k accesses.
# Five secrets is under $0.01/hour. Not the reason to destroy this stack.

locals {
  # secret id -> the env var the container reads it as. Names match app/config.py
  # (pydantic-settings uppercases the field name).
  app_secrets = {
    "database-url"      = "DATABASE_URL"
    "anthropic-api-key" = "ANTHROPIC_API_KEY"
    "google-api-key"    = "GOOGLE_API_KEY"
    "hf-api-key"        = "HF_API_KEY"
    "ingest-api-key"    = "INGEST_API_KEY"
  }

  # Only the LLM keys get a placeholder version. database-url's real value is
  # computed below, because Terraform is the thing that knows the password.
  llm_secrets = toset(["anthropic-api-key", "google-api-key", "hf-api-key"])
}

resource "google_secret_manager_secret" "app" {
  for_each = local.app_secrets

  secret_id = "${var.service_name}-${each.key}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# The database URL, in the unix-socket form asyncpg wants against the Cloud SQL
# Auth Proxy socket Cloud Run mounts at /cloudsql. Terraform generates the
# password, so Terraform is the only thing that can write this version.
#
# This puts the password in terraform.tfstate in cleartext. That is how
# Terraform works, not a bug being papered over: state is gitignored (see
# .gitignore), it is local (see versions.tf), and the whole database it protects
# is deleted at destroy. If that trade is unacceptable, drop this resource, set
# `password_wo` by hand, and add the version with gcloud.
resource "google_secret_manager_secret_version" "database_url" {
  secret      = google_secret_manager_secret.app["database-url"].id
  secret_data = "postgresql+asyncpg://${google_sql_user.app.name}:${random_password.db.result}@/${google_sql_database.rag.name}?host=/cloudsql/${google_sql_database_instance.pg.connection_name}"
}

# Cloud Run refuses to deploy a revision that references a secret with no
# versions, so the LLM keys need *something* here before the first apply.
#
# Consequence, stated plainly: until you replace these, ANTHROPIC_API_KEY holds
# a non-empty string, so app/generation sees a configured provider and /ask
# returns whatever Anthropic says to a bad key (a 401, which the failover policy
# deliberately does not retry) instead of the local 503 that names the missing
# variable. Replace them before demoing:
#
#   printf '%s' "$ANTHROPIC_API_KEY" | gcloud secrets versions add \
#     arabic-rag-anthropic-api-key --data-file=-
#
# `version = "latest"` in run.tf is resolved when an instance starts, so the next
# cold start picks the new value up without a redeploy.
# The /ingest key. Generated rather than placeholdered, because a placeholder
# committed to this repo would be a published password: app/config.py enforces
# `x-api-key` whenever INGEST_API_KEY is non-empty, so a known value is worse
# than no value — it looks locked and is not.
#
# Terraform owns it for the same reason it owns the database password: something
# has to write the first version, and only Terraform is running. Read it back
# with `terraform output -raw ingest_api_key`. Same tfstate-in-cleartext trade as
# random_password.db above, and the same mitigation — gitignored, local, and the
# stack it guards is destroyed with the demo.
resource "random_password" "ingest_key" {
  length = 32
  # Alphanumeric only: this value is sent as an HTTP header, and a header whose
  # correctness depends on shell quoting is a support ticket.
  special = false
}

resource "google_secret_manager_secret_version" "ingest_api_key" {
  secret      = google_secret_manager_secret.app["ingest-api-key"].id
  secret_data = random_password.ingest_key.result
}

resource "google_secret_manager_secret_version" "llm_placeholder" {
  for_each = local.llm_secrets

  secret      = google_secret_manager_secret.app[each.key].id
  secret_data = "placeholder-replace-with-gcloud-secrets-versions-add"

  lifecycle {
    # Never fight the human. Once a real version exists, Terraform must not
    # reintroduce the placeholder as the latest version on the next apply.
    ignore_changes = [secret_data]
  }
}
