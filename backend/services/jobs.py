import uuid
from typing import Dict

JOBS: Dict[str, dict] = {}

def create_job():
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "pending"}
    return job_id

def set_status(job_id, status, result=None, error=None):
    if job_id not in JOBS:
        JOBS[job_id] = {}
    JOBS[job_id]["status"] = status
    if result is not None:
        JOBS[job_id]["result"] = result
    if error is not None:
        JOBS[job_id]["error"] = error

def get_job(job_id):
    return JOBS.get(job_id, {"status": "unknown"})
