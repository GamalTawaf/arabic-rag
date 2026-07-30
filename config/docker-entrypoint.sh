#!/bin/sh
set -e

## Run migrations on container start, then serve.
##
## ---------------------------------------------------------------------------
## THIS IS NOT HOW MIGRATIONS SHOULD BE RUN. It is here because the database has
## no public address (terraform/network.tf), so a laptop cannot reach it, and a
## demo stack should not also carry a job resource to solve a one-command
## problem. Everything below is the cost of that shortcut, written down so the
## next person does not have to discover it:
##
##   - Every instance runs this. With max_instances = 2 and scale-from-zero, two
##     cold starts can run `alembic upgrade head` concurrently. Postgres DDL is
##     transactional and Alembic takes no advisory lock, so a race ends in a
##     failed revision or a deadlock, not corruption — but it ends in a container
##     that exited, and Cloud Run will report a failed deploy.
##   - `set -e` means a failed migration takes the service down instead of
##     serving stale-but-working code. That is the right call for a schema the app
##     assumes exists, and the wrong call if you ever need to roll back the image
##     without rolling back the schema.
##   - There is no place to review the SQL before it runs, no dry run, and no
##     separation between "deploy code" and "change the database".
##
## What to do instead, in order of preference: a Cloud Run job (or Kubernetes
## Job) executed as an explicit deploy step; a migration stage in CI that runs
## against the instance before the new revision goes live; or a maintenance
## container run by hand. All three keep the schema change visible and one
## instance at a time.
## ---------------------------------------------------------------------------
alembic -c config/alembic.ini upgrade head

## Then run whatever was asked for — the Dockerfile's CMD by default, or an
## override. `exec "$@"` rather than a hardcoded uvicorn line: with ENTRYPOINT and
## no CMD, `docker run <image> /bin/sh` and a Cloud Run job created with --args
## were appended as $1 and silently ignored, so a one-shot command served HTTP
## until its task timeout instead.
##
## exec, so the command becomes PID 1 and receives Cloud Run's SIGTERM directly.
## Without it the shell holds PID 1, swallows the signal, and every scale-down
## waits out the 10 s grace period before being killed.
exec "$@"
