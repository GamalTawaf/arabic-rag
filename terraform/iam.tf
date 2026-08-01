# Service accounts.
#
# Two of them, and neither is the default compute service account. The default
# compute SA carries roles/editor on the whole project: anything running as it
# can delete the database it reads from. Cloud Run only uses it when you decline
# to say otherwise, so this file says otherwise.

# --- runtime: the identity the container runs as ---------------------------

resource "google_service_account" "runtime" {
  account_id   = "${var.service_name}-run"
  display_name = "arabic-rag Cloud Run runtime"
  description  = "Runs the arabic-rag container. Four roles, each scoped to one resource where the API allows it."

  depends_on = [google_project_service.apis]
}

# roles/cloudsql.client — connect through the Cloud SQL Auth Proxy that Cloud Run
# mounts as a unix socket (see run.tf). Project-scoped because Cloud SQL does not
# expose instance-level IAM for the client role; the instance itself is the only
# one in the stack.
resource "google_project_iam_member" "runtime_cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# roles/cloudtrace.agent — write spans. Project-scoped because Cloud Trace has no
# finer resource to bind to.
resource "google_project_iam_member" "runtime_trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# roles/pubsub.subscriber, bound to the one subscription rather than the project.
#
# Honest note: push delivery does not need this. Pub/Sub pushes to Cloud Run
# using its own OIDC identity (see pubsub.tf), so the runtime SA never calls the
# Pub/Sub API on the push path. It is here because the moment the ingestion
# worker becomes a pull loop — the shape you want if ingestion outgrows a 600 s
# ack deadline — this is the exact role it needs, and binding it to one
# subscription costs nothing and grants nothing else.
resource "google_pubsub_subscription_iam_member" "runtime_subscriber" {
  subscription = google_pubsub_subscription.ingest_push.name
  role         = "roles/pubsub.subscriber"
  member       = "serviceAccount:${google_service_account.runtime.email}"
}

# roles/secretmanager.secretAccessor, bound per secret rather than per project.
# Same role the brief asks for, three secrets wide instead of every secret in the
# project.
resource "google_secret_manager_secret_iam_member" "runtime_secret_accessor" {
  for_each = google_secret_manager_secret.app

  secret_id = each.value.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

# --- pubsub-invoker: the identity Pub/Sub presents at /ingest ---------------

resource "google_service_account" "pubsub_invoker" {
  account_id   = "${var.service_name}-pubsub"
  display_name = "arabic-rag Pub/Sub push invoker"
  description  = "Subject of the OIDC token Pub/Sub attaches to push requests. Holds run.invoker on one service and nothing else."

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "pubsub_invoker" {
  project  = google_cloud_run_v2_service.rag.project
  location = google_cloud_run_v2_service.rag.location
  name     = google_cloud_run_v2_service.rag.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.pubsub_invoker.email}"
}

# Pub/Sub's own service agent must be allowed to mint OIDC tokens *as* the
# invoker SA. Bound to that one service account, not to the project — the
# project-wide version of this grant is the usual way a Pub/Sub topic quietly
# becomes an impersonation primitive.
#
# The service agent's address is derived, not created: it appears the first time
# the Pub/Sub API is used in the project. google_project_service_identity would
# force it into existence, but that resource is google-beta only and adding a
# second provider for one address is not worth it.
resource "google_service_account_iam_member" "pubsub_agent_token_creator" {
  service_account_id = google_service_account.pubsub_invoker.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-pubsub.iam.gserviceaccount.com"

  depends_on = [google_project_service.apis]
}

# --- public access ----------------------------------------------------------

# Gated on a variable so turning the demo URL off is a one-line change, not a
# refactor. See variables.tf for what does and does not protect this URL.
resource "google_cloud_run_v2_service_iam_member" "public" {
  count = var.allow_unauthenticated ? 1 : 0

  project  = google_cloud_run_v2_service.rag.project
  location = google_cloud_run_v2_service.rag.location
  name     = google_cloud_run_v2_service.rag.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
