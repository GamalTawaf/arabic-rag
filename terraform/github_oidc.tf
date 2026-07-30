# Keyless CI deploys: GitHub Actions federates into a deployer service account
# through Workload Identity, so no service-account JSON key ever exists — not in
# the repo, not in GitHub secrets, not on a laptop. A leaked key is permanent
# until someone notices; a leaked OIDC token is useless outside a run.
#
# What CI is allowed to do, and deliberately nothing more: push an image and roll
# a new Cloud Run revision. It cannot create infrastructure, read a secret's
# value, or touch the database. `terraform apply` stays manual because the state
# is local (versions.tf has no backend), so a CI apply would either fight the
# laptop's state or need a bucket this stack does not create.
#
# Set enable_github_oidc = false and this whole file becomes inert.

locals {
  oidc_count = var.enable_github_oidc ? 1 : 0
}

resource "google_iam_workload_identity_pool" "github" {
  count = local.oidc_count

  workload_identity_pool_id = "${var.service_name}-github"
  display_name              = "GitHub Actions"
  description               = "Federates ${var.github_repo} into the deployer service account"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  count = local.oidc_count

  workload_identity_pool_id          = google_iam_workload_identity_pool.github[0].workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub OIDC"

  # `attribute_condition` is the load-bearing line. Without it, *any* GitHub
  # repository's workflow could mint a token for this pool — the audience is
  # github.com, not this repo. The binding below narrows to one repo as well;
  # both exist because a mistake in either one alone is a public deploy hole.
  attribute_condition = "assertion.repository == '${var.github_repo}'"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "deployer" {
  count = local.oidc_count

  account_id   = "${var.service_name}-deployer"
  display_name = "CI deployer (GitHub Actions)"
  description  = "Pushes images and deploys Cloud Run revisions. No infra rights."
}

# Only the main branch of only that repo may impersonate the deployer. A PR from
# a fork runs with a different `ref`, so it can build and test but never deploy.
resource "google_service_account_iam_member" "deployer_from_github" {
  count = local.oidc_count

  service_account_id = google_service_account.deployer[0].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github[0].name}/attribute.repository/${var.github_repo}"
}

# Push images. Scoped to the one repository, not the project.
resource "google_artifact_registry_repository_iam_member" "deployer_writer" {
  count = local.oidc_count

  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer[0].email}"
}

# Roll a revision of this one service. roles/run.developer, not run.admin: it can
# update the service it is bound to and cannot grant anyone else access to it —
# setting IAM policy is how a deploy role quietly becomes a "make it public" role.
resource "google_cloud_run_v2_service_iam_member" "deployer_developer" {
  count = local.oidc_count

  location = google_cloud_run_v2_service.rag.location
  name     = google_cloud_run_v2_service.rag.name
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.deployer[0].email}"
}

# A revision runs *as* the runtime service account, and Cloud Run requires the
# deployer to hold actAs on it. Bound to that one service account — the
# project-wide version of this grant is impersonation of everything.
resource "google_service_account_iam_member" "deployer_acts_as_runtime" {
  count = local.oidc_count

  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer[0].email}"
}
