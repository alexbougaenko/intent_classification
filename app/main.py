from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.intent_classification import IntentClassificator


class Message(BaseModel):
    input: str
    output: str = None

app = FastAPI()
clf = IntentClassificator()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/classify_intent")
async def classify(message: Message):
    message.output = clf.predict(text=message.input)
    return {"response": {"intent": message.output}}
