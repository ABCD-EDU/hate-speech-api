
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from app.schema.schema import ModelResponse, TextInput
from app.predict import predict


app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post('/api/predict/{text}', response_model=ModelResponse)
async def do_predict(text: str):
    return predict(text)
