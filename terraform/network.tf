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
# /24 rather than the /26 Direct VPC egress asks for (a /28 is the *connector's*
# floor, not this one's): the ceiling is instances, and a range that
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
  name = "${var.service_name}-psa"
  # Pinned, not auto-allocated. Left to Google, the allocator can return a /16
  # that covers run_subnet_cidr (10.8.0.0/24) — and nothing orders these two
  # creations, so an overlap surfaces as a failed apply against a half-built VPC.
  address       = "10.100.0.0"
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
  # ABANDON: Terraform removes this from state without calling the API, so the
  # peering survives — which then blocks deleting the VPC itself. That makes the
  # `gcloud compute networks peerings delete` step in README.md a MANDATORY part of
  # every teardown, not a fallback. The alternative (letting Terraform delete the
  # connection) fails while the SQL instance still exists, which is worse: destroy
  # stops with the database intact and still billing.
  deletion_policy = "ABANDON"
}
