"""Standalone entrypoints meant to be invoked on a schedule (system cron, k8s
CronJob, etc.) — not by the FastAPI app. Run as `python -m app.crons.<name>`.
"""
