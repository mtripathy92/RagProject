from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from app import TelecomRAGService

rag_service = TelecomRAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag_service.initialize()
    except Exception as e:
        print(f"Error setting up indexing or LLM chain: {e}")
    yield


app = FastAPI(title="Telecom Hybrid RAG API", lifespan=lifespan)


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: QuestionRequest):
    if not rag_service.is_ready:
        raise HTTPException(status_code=500, detail="RAG system not initialized properly.")
    try:
        response = rag_service.ask(payload.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Runs local FastAPI instance on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)