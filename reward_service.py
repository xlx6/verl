# reward_server.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from transformers import pipeline
import torch

MODEL_PATH = "/home/xinglixian/models/distilbert-imdb"
sentiment_pipe = None


class ScoreRequest(BaseModel):
    solution_str: List[str]

class ScoreResponse(BaseModel):
    score: List[float]

app = FastAPI()

@app.on_event("startup")
def load_model():
    global sentiment_pipe
    try:
        device = 0 if torch.cuda.is_available() else -1
        print(f"Server Startup: Loading model '{MODEL_PATH}' onto device {device}...")
        
        sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=device,
            top_k=None,
            truncation=True,
            max_length=512
        )
        # resp_text = sentiment_pipe(['I like this', 'I hate this.'])
        # print(f'ReWard Model Test: {resp_text}')
        # print("Server Startup: Model loaded successfully.")
    except Exception as e:
        print(f"Server Startup: Failed to load model! {e}")
        sentiment_pipe = None



@app.post("/score", response_model=ScoreResponse)
async def get_score(request: ScoreRequest):
    if sentiment_pipe is None:
        raise HTTPException(status_code=500, detail="Reward Model is not loaded")

    try:
        texts_to_score = request.solution_str

        if not texts_to_score:
            return ScoreResponse(score=[])

        results_list = sentiment_pipe(texts_to_score)
        
        scores_list = []
        for results in results_list:
            reward_score = 0.0
            for item in results:
                if item['label'] == 'NEGATIVE':
                    reward_score = item['score']
                    break
            scores_list.append(float(reward_score))
        print(f'return scores len: {len(scores_list)}')
        return ScoreResponse(
            score=scores_list
        )

    except Exception as e:
        print(f"Error during scoring: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)