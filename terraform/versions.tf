# Provider and Terraform version pins.
#
# Pinned, not floated: this stack has never been applied (no GCP credentials in
# this repo — see README.md), so the only thing standing between it and a
# surprise is a version boundary someone can reproduce.

terraform {
  required_version = ">= 1.9.0, < 2.0.0"

  required_providers {
    google = {
      source = "hashicorp/google"
      # 7.41.0 is the latest release at time of writing. `~> 7.41` allows
      # 7.41.x -> 7.x patch/minor, not the 8.0 major, where Google routinely
      # changes resource defaults (deletion_protection did exactly that in 6.0).
      version = "~> 7.41"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # ponytail: local state on purpose. This stack lives for the length of a demo;
  # a GCS remote-state bucket would be the one resource that survives the
  # `terraform destroy` this whole design exists to make cheap. Upgrade path if
  # more than one person ever applies it: uncomment and create the bucket by hand.
  #
  # backend "gcs" {
  #   bucket = "arabic-rag-tfstate"
  #   prefix = "ephemeral"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
