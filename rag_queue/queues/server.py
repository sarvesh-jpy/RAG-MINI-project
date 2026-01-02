from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from redis import Redis
from rq import Queue
from rq.job import Job
from dotenv import load_dotenv
import os

# Import the function we want to run from our worker file
# Note: Ensure worker.py is in the same folder and has a function named 'process_query'
from worker import process_query

# 1. Load Environment & App
load_dotenv()
app = FastAPI()

# 2. Setup Redis Queue
# This connects to the Redis Docker container (default port 6379)
try:
    redis_conn = Redis(host='localhost', port=6379)
    queue = Queue(connection=redis_conn)
    print("✅ Connected to Redis Queue!")
except Exception as e:
    print(f"❌ Failed to connect to Redis: {e}")

# 3. Define the Data Model
class QueryRequest(BaseModel):
    question: str

# 4. The Endpoints

@app.get("/")
def root():
    return {"status": "DocuMind Server is Running!"}

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """
    Takes a question, sends it to the background worker, 
    and returns a Job ID immediately.
    """
    try:
        # Enqueue the job. This is instant.
        # We pass the function 'process_query' and the argument 'request.question'
        job = queue.enqueue(process_query, request.question)
        
        return {
            "status": "queued",
            "job_id": job.get_id(),
            "message": "Use /job_result?job_id=... to check the answer."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job_result")
def get_result(job_id: str = Query(..., description="The ID of the job to check")):
    """
    Checks if the background worker has finished the job.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    if job.is_finished:
        # The worker is done, return the answer
        return {
            "status": "finished", 
            "result": job.result
        }
    elif job.is_failed:
        return {"status": "failed", "error": "Something went wrong in the worker."}
    else:
        # The worker is still thinking
        return {"status": "processing", "message": "Still thinking... try again in 1s."}