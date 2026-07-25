# Docker repository for the service image.
#
# Cost while it exists: Artifact Registry bills storage at roughly $0.10 per GB
# per month above a 0.5 GB free allowance (us list price, unverified from this
# machine). This image is not small — a torch-carrying build is ~5 GB — so call
# it ~$0.50/month, ~$0.0007/hour. That is the one piece of this stack worth
# leaving up between demos, because re-pushing 5 GB costs more of your afternoon
# than $0.50 costs of your money.

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.service_name
  description   = "Container images for the arabic-rag service (ephemeral demo stack)."
  format        = "DOCKER"

  # Keep the last few tags; every push before those is deleted. Without this the
  # repository is the only resource in the stack whose cost grows monotonically.
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-recent-versions"
    action = "KEEP"

    most_recent_versions {
      keep_count = 5
    }
  }

  depends_on = [google_project_service.apis]
}

locals {
  # The image Cloud Run pulls. Terraform never builds or pushes it — see
  # README.md, "What you must do by hand".
  image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/${var.service_name}:${var.image_tag}"
}
