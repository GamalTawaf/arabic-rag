# Service APIs, and the project lookup everything else derives service-agent
# identities from.

data "google_project" "this" {
  project_id = var.project_id
}

locals {
  services = [
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "pubsub.googleapis.com",
    "cloudtrace.googleapis.com", # the "Cloud Trace enabled" requirement, literally
    "iam.googleapis.com",
    "compute.googleapis.com",           # the VPC and the subnet Cloud Run egresses into
    "servicenetworking.googleapis.com", # the Private Service Access peering
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.services)

  project = var.project_id
  service = each.value

  # Do not switch APIs off on destroy. Disabling an API is project-wide and
  # instant; if anything else in the project uses Pub/Sub or Cloud SQL, a
  # `terraform destroy` here would break it. Destroy should remove *this
  # stack's* resources and nothing else. Disabled APIs also cost nothing.
  disable_on_destroy = false
}
