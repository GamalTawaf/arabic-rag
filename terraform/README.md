# terraform/ — the ephemeral GCP stack

Cloud Run + Cloud SQL + Pub/Sub + Artifact Registry + Secret Manager, sized for a
demo and designed to be deleted the same day. Applied before someone watches it,
destroyed after, so steady-state cost is zero.

---

## Read this first: this has never been applied

**No `terraform apply` has ever been run against this configuration.** This
repository has no GCP credentials, no billing account, and no project attached —
by design, since the whole project is built to run at ~$0 on a laptop. Everything
below is *validated-but-unproven infrastructure*:

| | |
|---|---|
| Syntax and schema | **verified** — see [Validation](#validation) for the exact commands and output |
| Provider/resource arguments | **verified** against the real `hashicorp/google` 7.41.0 schema |
| Plan against a project | never run |
| Apply | never run |
| Destroy | never run |
| Any cost figure below | list-price arithmetic, never a billing statement |

So: nothing here has been deployed. Treat the first `terraform apply` as an
experiment, not a rollout, and read [What I would expect to break
first](#what-i-would-expect-to-break-first) before running it in front of
anybody. A reviewer can tell the difference between "this works" and "this
type-checks", and claiming the first would be worth less than admitting the
second.

---

## What it creates

| Resource | File | Notes |
|---|---|---|
| Cloud Run service (gen2) | `run.tf` | `min_instances = 0`, `cpu_idle = false`, startup probe on `/health`, Cloud SQL socket volume |
| Cloud SQL Postgres 17 | `sql.tf` | `db-f1-micro`, zonal, no backups, no authorized networks, SSL required |
| Artifact Registry (Docker) | `artifact_registry.tf` | keeps the 5 most recent versions |
| Pub/Sub topic + push subscription + DLQ | `pubsub.tf` | pushes to `/ingest/pubsub` with an OIDC token |
| Secret Manager × 5 | `secrets.tf` | `database-url`, `anthropic-api-key`, `google-api-key`, `hf-api-key`, `ingest-api-key` |
| Two service accounts | `iam.tf` | runtime + Pub/Sub invoker. **Not** the default compute SA |
| 7 project APIs | `apis.tf` | not disabled on destroy — see the comment there |

Least privilege, concretely. The runtime service account holds exactly four
roles, and two of them are bound to a single resource rather than the project:

| Role | Scope | Why |
|---|---|---|
| `roles/cloudsql.client` | project | Cloud SQL exposes no instance-level binding for it |
| `roles/cloudtrace.agent` | project | Cloud Trace has no finer resource |
| `roles/pubsub.subscriber` | **one subscription** | not needed for push; it is the role a future pull worker needs |
| `roles/secretmanager.secretAccessor` | **each secret, individually** | not "every secret in the project" |

The Pub/Sub invoker SA holds `roles/run.invoker` on one service and nothing else,
and the Pub/Sub service agent can mint tokens as it — bound to that one service
account, not project-wide, because the project-wide version of that grant is how
a Pub/Sub topic quietly becomes an impersonation primitive.

### Region

`me-central1` — **Doha, Qatar**. The corpus is Qatari Labour Law No. 14 of 2004
and the audience is the Doha market, so data residency and demo latency point at
the same place. The alternatives considered were `me-central2` (Dammam, Saudi
Arabia) and `me-west1` (Tel Aviv). me-central1 carries Cloud Run, Cloud SQL,
Artifact Registry and Secret Manager. It is not a "tier 1" region, so list prices
run above the us-central1 figures the cost arithmetic below is computed from.

---

## What it costs while it is up

The number that justifies destroying it:

```
Cloud SQL  db-f1-micro, ENTERPRISE, zonal, no HA, no backups   ~$0.0150 /hour
Cloud SQL  10 GB PD_SSD  (10 × $0.17/GB-month ÷ 730 h)         ~$0.0023 /hour
Artifact Registry, ~5 GB image ($0.10/GB-month)                ~$0.0007 /hour
Pub/Sub, Secret Manager, Cloud Trace at demo volume            ~$0
Cloud Run at min_instances = 0, nobody asking                    $0
                                                               --------------
IDLE TOTAL                                                     ~$0.018 /hour
                                                               ~$0.43  /day
                                                               ~$13    /month
```

Plus, only while an instance is warm:

```
Cloud Run 4 vCPU + 8 GiB, CPU always allocated                 ~$0.32  /hour
  (4 × $0.000018/vCPU-s + 8 × $0.000002/GiB-s, and Cloud Run
   keeps an idle instance ~15 min after the last request)
```

So an hour of sporadic demo traffic is on the order of **$0.15–$0.30**, several
times the database underneath it, and a month of leaving it up is **~$13** —
almost all of it Cloud SQL, which bills for existing rather than for being used.
`terraform destroy` takes that to **~$0.50/month**: the container image, which is
worth keeping so the next demo is an `apply` and not a 5 GB `docker push`.

These are us-central1 list prices as I understand them, with the arithmetic shown
so you can substitute real ones. **I could not verify them from this machine** —
no billing account, no console, and I did not fetch the pricing pages. Check
`cloud.google.com/sql/pricing` and `cloud.google.com/run/pricing` before quoting
any of this to someone who is paying.

---

## What you must do by hand

Terraform provisions. It does not build, push, migrate, or load data. Four things
are yours:

### 1. Build and push the image

The `Dockerfile` installs CPU torch and `sentence-transformers` alongside
`requirements.txt`, because the service embeds queries with `BAAI/bge-m3` and
reranks with `BAAI/bge-reranker-v2-m3` (`app/deps.py`, `SERVICE_MODEL_KEY = "bge"`).
Setting `rerank_enabled = false` would not remove the need — the *embedder* uses
torch too. Verified locally: the image builds at 2.06 GB and `import app.main`
succeeds inside it.

What the image does **not** carry is the ~4.4 GB of model weights; they download
on first use, so expect a slow first request after each scale-to-zero cold start.
The `Dockerfile` comment names the build step that trades image size for cold-start
latency if that matters more.

Then:

```bash
PROJECT=your-gcp-project-id
REGION=me-central1
IMAGE="$REGION-docker.pkg.dev/$PROJECT/arabic-rag/arabic-rag:latest"

gcloud auth configure-docker "$REGION-docker.pkg.dev"
docker build --platform linux/amd64 -t "$IMAGE" .   # linux/amd64 matters on Apple Silicon
docker push "$IMAGE"
```

Chicken-and-egg: Artifact Registry is created by this stack, so the very first
run is `terraform apply -target=google_artifact_registry_repository.images`,
then push, then a full `terraform apply`. Cloud Run will not create a service
whose image does not exist.

### 2. Replace the LLM API key secrets

Terraform creates `arabic-rag-anthropic-api-key`, `arabic-rag-google-api-key` and
`arabic-rag-hf-api-key` with a placeholder version, because Cloud Run refuses to
deploy a revision that references a secret with no versions. Until you replace
them, the app sees a non-empty key, calls the provider with it, and returns that
provider's 401 rather than the local 503 that names the missing variable.

`generation_providers` defaults to `huggingface`, so **the only one that has to be
real is the HF key**:

```bash
printf '%s' "$HF_API_KEY" | \
  gcloud secrets versions add arabic-rag-hf-api-key --data-file=-

# only if you also list them in generation_providers
printf '%s' "$ANTHROPIC_API_KEY" | \
  gcloud secrets versions add arabic-rag-anthropic-api-key --data-file=-
printf '%s' "$GOOGLE_API_KEY" | \
  gcloud secrets versions add arabic-rag-google-api-key --data-file=-
```

The service reads `version = "latest"`, resolved when an instance starts, so the
next cold start picks it up. Force one with `gcloud run services update
arabic-rag --region "$REGION" --update-labels rev=$(date +%s)` if you do not want
to wait for scale-to-zero.

`arabic-rag-database-url` is the exception: Terraform generates the password, so
Terraform writes that one. **The generated password is therefore in
`terraform.tfstate` in cleartext.** State is gitignored and local; the database it
protects is deleted at destroy.

### 3. Migrate and load the corpus

`CREATE EXTENSION vector` is not a Terraform resource and not a Cloud SQL database
flag — pgvector ships with Cloud SQL Postgres 17 and has to be created *inside*
the database. `alembic/versions/0001_initial.py` does exactly that, and
`google_sql_user.app` is created through the Cloud SQL API, which grants it
`cloudsqlsuperuser`, which is the privilege that makes it work.

From your laptop, through the Auth Proxy:

```bash
cloud-sql-proxy "$(terraform output -raw database_connection_name)" --port 5432 &

export DATABASE_URL="postgresql+asyncpg://rag_user:PASSWORD@localhost:5432/rag_db"
#   PASSWORD: gcloud secrets versions access latest --secret arabic-rag-database-url
#   (that secret holds the full unix-socket URL; take the password out of it)

alembic upgrade head                     # creates the vector extension + schema
python -m ingestion ingest               # 233 chunks from the committed corpus
python -m ingestion backfill --model bge # the column the service queries
python -m ingestion stats                # confirm coverage 1.0
```

Or publish the same documents to the ingest topic and let the service do it:

```bash
gcloud pubsub topics publish "$(terraform output -raw ingest_topic)" \
  --message "$(jq -c '{doc_id, title, text}' some-document.json)"
```

### 4. Destroy it

See below. It is the whole point.

---

## Apply / destroy workflow

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # then set project_id
terraform init

# first time only — Artifact Registry must exist before the image does
terraform apply -target=google_artifact_registry_repository.images
#   ... build and push the image (step 1 above) ...

terraform apply
terraform output service_url
#   ... run the migration and ingest (step 3) ...
#   ... demo ...
terraform destroy
```

Expect the first `apply` to take roughly 10–15 minutes; Cloud SQL instance
creation is almost all of it. `destroy` is quicker.

What `destroy` removes: the Cloud Run service, the Cloud SQL instance **and every
chunk in it**, both Pub/Sub topics and all three subscriptions, all five secrets,
both service accounts, every IAM binding above, and the Artifact Registry
repository including the pushed image.

What it does **not** remove: the enabled APIs (deliberate — `disable_on_destroy =
false` in `apis.tf`, because disabling an API is project-wide and would break
anything else in the project that uses Pub/Sub or Cloud SQL), and anything not in
this state file.

Losing the database is fine and intended: the corpus is committed under
`data/corpus/` and `python -m ingestion ingest` rebuilds it in about a minute.
That is why `backup_configuration` is disabled — paying to protect a derived
artifact a script regenerates is theatre.

---

## Cloud Trace, honestly

Cloud Trace is enabled (`cloudtrace.googleapis.com` in `apis.tf`) and the runtime
service account holds `roles/cloudtrace.agent`, so **Cloud Run's own request
spans reach Cloud Trace** with no further configuration.

The app's per-stage spans — `plan → embed → cache.lookup → retrieve → fuse →
rerank → generate`, the ones with token counts and USD cost on them — do **not**,
and `otel_exporter_otlp_endpoint` defaults to empty for that reason.
`app/observability/tracing.py` exports over OTLP/gRPC, and that exporter has no
way to attach Google credentials, so it cannot talk to Google's telemetry
endpoint on its own. Getting those spans into Cloud Trace needs a collector that
can authenticate — an `otel/opentelemetry-collector-contrib` sidecar with the
`googlecloud` exporter, or Cloud Run's built-in OTLP endpoint (in preview at time
of writing; I have not verified its availability in me-central1). Neither is
wired here. Point `otel_exporter_otlp_endpoint` at whichever you set up.

Saying "Cloud Trace enabled" and leaving it there would be the easy version of
this paragraph, and it would be wrong in the exact way that gets found during a
demo.

---

## Validation

`terraform` is not installed on the machine this was written on. **OpenTofu
v1.12.2 is**, and OpenTofu is a fork of Terraform 1.x that consumes the same HCL
and the same `hashicorp/google` provider, so the checks below are real schema
validation against the real provider — but they were run with `tofu`, not
`terraform`. Substitute the binary name; the commands and the configuration are
otherwise unchanged.

```console
$ tofu fmt -check
$ echo $?
0

$ tofu init -backend=false
Initializing the backend...
Initializing provider plugins...
- Finding hashicorp/random versions matching "~> 3.6"...
- Finding hashicorp/google versions matching "~> 7.41"...
- Installing hashicorp/google v7.41.0...
- Installing hashicorp/random v3.9.0...
- Installed hashicorp/random v3.9.0 (signed, key ID 0C0AF313E5FD9F80)
- Installed hashicorp/google v7.41.0 (signed, key ID 0C0AF313E5FD9F80)

OpenTofu has been successfully initialized!

$ tofu validate
Success! The configuration is valid.
```

One real error was caught and fixed on the way: `authorized_networks` is a
repeatable block in this provider, not a list argument, so `authorized_networks =
[]` failed validation. It is now written by saying nothing at all — which is also
the only spelling of "none" that a future edit cannot weaken by appending one
more CIDR to an existing list.

There is no `.terraform.lock.hcl` committed. `tofu init` writes one pinned to
`registry.opentofu.org`, which a `terraform init` would not accept; shipping a
lock file for the wrong registry is worse than shipping none. Run `terraform init`
once and commit the lock it produces.

**What validation does not tell you:** whether the values are right. `validate`
checks structure and types against the provider schema. It does not check that
`db-f1-micro` is still accepted for `POSTGRES_17` in the `ENTERPRISE` edition,
that Cloud Run in me-central1 will accept a 4 vCPU / 8 GiB revision, that IAM
propagates in time for the first push delivery, or that any of the pricing above
is current. Only an apply does that, and there has not been one.

---

## What I would expect to break first

Ranked by how confident I am it will bite, on a real first apply:

1. **Cold-start timeouts.** First request on a cold instance downloads ~4.4 GB of
   weights, then runs a cross-encoder on CPU. The measured 914 ms p95 rerank in
   `docs/latency-budget.md` was Apple Silicon MPS; Cloud Run has no MPS and no
   GPU, and nobody in this repo has measured the CPU number. The 300 s request
   timeout is slack for that, not a target — and `POST /ask` may still exceed
   the 3.5 s p95 budget that `benchmark/replay.py` enforces locally.
2. **IAM propagation on the first push.** Pub/Sub's service agent is granted
   token-creator on the invoker SA in the same apply that creates the
   subscription; IAM is eventually consistent, so the first few pushes may 403
   and retry. The retry policy absorbs it. It will look alarming in the logs.
3. **`db-f1-micro` under an ingest.** 0.6 GB of RAM is ample for a 233-chunk
   HNSW index and thin for a Postgres 17 that is also being written to. If ingest
   is slow or the instance thrashes, `db_tier = "db-g1-small"` is the one-line
   answer.
4. **The Pub/Sub service agent may not exist yet** in a brand-new project. The
   two IAM bindings that reference `service-<number>@gcp-sa-pubsub...` assume it
   has been created, which happens the first time the Pub/Sub API is used.
   Enabling the API in the same apply usually creates it; if the binding fails,
   re-running `terraform apply` is the fix.

None of these are reasons not to ship the configuration. They are the list I
would work through with a project in front of me, written down now so that the
first person to apply it does not have to rediscover them.

---

## CI deploys, keyless

`github_oidc.tf` creates a Workload Identity pool, a provider locked to one
GitHub repository, and a `arabic-rag-deployer` service account that
`.github/workflows/deploy.yml` impersonates. **No service-account key exists** —
not in the repo, not in GitHub secrets. A leaked key lasts until someone notices;
a leaked OIDC token is worthless outside the run that minted it.

The split is deliberate:

| | Where it runs | Why |
|---|---|---|
| Build image, push, roll a Cloud Run revision | GitHub Actions | Repeatable, needs no local state |
| `terraform apply` / `destroy` | A laptop | State is local (`versions.tf` has no backend). A CI apply would fight that state or need a bucket this stack does not create. |

What the deployer can do: write to the one Artifact Registry repository,
`roles/run.developer` on the one service, and `actAs` the runtime service
account. Not `run.admin` — a role that can set IAM policy on a service is a role
that can make it public. It cannot create infrastructure, read secret values, or
reach the database.

Two guards on who may deploy, because either alone failing open is a hole:
`attribute_condition` on the provider (`assertion.repository == "<repo>"`) and the
`principalSet` on the impersonation binding. The workflow is
`workflow_dispatch`-only: a live demo that redeploys itself mid-presentation is a
way to break it in front of someone.

After the first apply, wire the GitHub environment once:

```bash
gh api -X PUT "repos/$REPO/environments/production"
gh variable set GCP_PROJECT_ID --env production --body "$(terraform output -raw project_id 2>/dev/null || echo "$PROJECT_ID")"
gh variable set REGION         --env production --body "$REGION"
gh variable set SERVICE_NAME   --env production --body arabic-rag
gh variable set WORKLOAD_IDENTITY_PROVIDER --env production --body "$(terraform output -raw workload_identity_provider)"
gh variable set DEPLOYER_SERVICE_ACCOUNT  --env production --body "$(terraform output -raw deployer_service_account)"
gh workflow run deploy.yml -f environment=production
```

Add required reviewers to the `production` environment in GitHub's settings if
the URL will be up long enough for that to matter.
