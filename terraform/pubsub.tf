# Asynchronous ingestion: a topic, a push subscription that calls the service,
# and a dead-letter topic for the messages that never succeed.
#
# Cost: Pub/Sub's first 10 GiB of throughput per month is free, and this stack
# moves kilobytes. Treat it as $0.

resource "google_pubsub_topic" "ingest" {
  name = "${var.service_name}-ingest"

  # Nothing subscribed for a week means nobody is coming.
  message_retention_duration = "604800s"

  depends_on = [google_project_service.apis]
}

resource "google_pubsub_topic" "ingest_dlq" {
  name = "${var.service_name}-ingest-dlq"

  message_retention_duration = "604800s"

  depends_on = [google_project_service.apis]
}

resource "google_pubsub_subscription" "ingest_push" {
  name  = "${var.service_name}-ingest-push"
  topic = google_pubsub_topic.ingest.id

  # 600 s is Pub/Sub's maximum. Ingestion is fetch -> normalize -> chunk ->
  # embed -> load; the embed step runs a transformer over every chunk, and on
  # Cloud Run CPU that is not fast. If a real corpus ever pushes past 600 s, the
  # answer is not a bigger deadline — it is a pull worker (see the
  # roles/pubsub.subscriber note in iam.tf).
  ack_deadline_seconds = 600

  message_retention_duration = "86400s"

  expiration_policy {
    ttl = "" # never auto-delete the subscription
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  # 5 is Pub/Sub's minimum. It is the right number here rather than a compromise:
  # /ingest/pubsub already acks permanently-bad messages itself (200 + an ERROR
  # log + a rejected counter), so the only thing that can reach five failures is
  # a sustained 5xx — the database being down, or ingestion timing out. Five
  # attempts with the backoff below is roughly 20 minutes of trying before the
  # message is set aside somewhere a human can read it.
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.ingest_dlq.id
    max_delivery_attempts = 5
  }

  push_config {
    # /ingest/pubsub, not /ingest.
    #
    # Both routes run the same pipeline (app/api/ingest.py), but they differ in
    # what they accept and in what a status code means. /ingest takes a bare
    # IngestDocument and answers 422 to anything it cannot parse; a Pub/Sub push
    # envelope posted there would be 422 on every single delivery — a poison
    # message that retries until it dead-letters, forever, for every message.
    # /ingest/pubsub unwraps the envelope, and answers 200 to a permanently
    # malformed message precisely so that loop cannot start.
    push_endpoint = "${google_cloud_run_v2_service.rag.uri}/ingest/pubsub"

    # Signed, verifiable identity on every push. The Cloud Run service checks
    # the token; only the invoker SA (iam.tf) can produce one, so an open
    # ingress does not mean an open ingest path.
    #
    # audience = the service URL, which is what Cloud Run's own IAM check
    # expects as the `aud` claim.
    oidc_token {
      service_account_email = google_service_account.pubsub_invoker.email
      audience              = google_cloud_run_v2_service.rag.uri
    }
  }

  depends_on = [
    google_cloud_run_v2_service_iam_member.pubsub_invoker,
    google_service_account_iam_member.pubsub_agent_token_creator,
  ]
}

# A dead-letter *topic* with no subscription is a black hole: Pub/Sub drops
# messages published to a topic nobody listens to. This pull subscription is what
# makes the DLQ inspectable —
#
#   gcloud pubsub subscriptions pull arabic-rag-ingest-dlq-pull --auto-ack --limit=10
#
resource "google_pubsub_subscription" "ingest_dlq_pull" {
  name  = "${var.service_name}-ingest-dlq-pull"
  topic = google_pubsub_topic.ingest_dlq.id

  ack_deadline_seconds       = 60
  message_retention_duration = "604800s"

  expiration_policy {
    ttl = ""
  }
}

# Pub/Sub's service agent moves the failed message from the subscription to the
# dead-letter topic on its own behalf, so it needs publish on the DLQ and
# subscribe on the source subscription. Without both, dead-lettering silently
# does not happen and the message redelivers forever.
resource "google_pubsub_topic_iam_member" "dlq_publisher" {
  topic  = google_pubsub_topic.ingest_dlq.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_pubsub_subscription_iam_member" "dlq_subscriber" {
  subscription = google_pubsub_subscription.ingest_push.name
  role         = "roles/pubsub.subscriber"
  member       = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}
