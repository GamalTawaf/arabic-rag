# The VPC that makes the database unreachable from the internet.
#
# Cloud SQL with a private IP has no public endpoint at all, so "who can open a
# socket to it" stops being an authorized-networks list and becomes a routing
# fact. Cloud Run reaches it through Direct VPC egress into the subnet below
# (run.tf); nothing else has a route.
#
# The cost, stated up front because it is the reason this was not the original
# shape: `google_service_networking_connection` creates a VPC peering that cannot
# be deleted while the producer connection exists, and the network cannot be
# deleted while the peering exists. `terraform destroy` can therefore end with a
# peering to remove by hand. See terraform/README.md, "Destroying a private-IP
# stack", before running destroy in front of anybody.

resource "google_compute_network" "vpc" {
  name = "${var.service_name}-vpc"
  # One subnet, created below, in one region. Auto mode would create a subnet in
  # every region on earth — 30-odd ranges to reason about for a stack that runs
  # in exactly one.
  auto_create_subnetworks = false

  depends_on = [google_project_service.apis]
}

# Direct VPC egress hands every Cloud Run instance an address out of this range.
# /24 rather than the /28 minimum: the ceiling is instances, and a range that
# fits max_instances today is a range that blocks scaling tomorrow.
resource "google_compute_subnetwork" "run" {
  name          = "${var.service_name}-run"
  region        = var.region
  network       = google_compute_network.vpc.id
  ip_cidr_range = var.run_subnet_cidr

  # Private Google Access: the container reaches Secret Manager, Artifact
  # Registry and Cloud Trace over Google's network. Without it, PRIVATE_RANGES_ONLY
  # egress still works (those calls take the public path) but this keeps them
  # internal, and it is free.
  private_ip_google_access = true
}

# The range Google's side of the peering allocates the database's address from.
# Reserved, not routed: nothing of ours is deployed into it.
resource "google_compute_global_address" "psa" {
  name          = "${var.service_name}-psa"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "psa" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.psa.name]

  # The documented way to make `terraform destroy` able to remove the peering
  # instead of leaving it behind. It is not a guarantee — a producer connection
  # that still exists will refuse — but without it destroy fails every time.
  deletion_policy = "ABANDON"
}
