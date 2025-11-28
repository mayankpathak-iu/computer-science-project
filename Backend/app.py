from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from extract_content import run_pipeline_from_tweet_url

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TweetRequest(BaseModel):
    tweet_url: str

@app.post("/check_tweet")
def check_tweet(req: TweetRequest):
    result = run_pipeline_from_tweet_url(req.tweet_url)
    return result