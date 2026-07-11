import base64
import io
import os

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

import google.generativeai as genai

from fastapi.middleware.cors import CORSMiddleware

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestData(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer(req: RequestData):

    image_bytes = base64.b64decode(req.image_base64)

    image = Image.open(io.BytesIO(image_bytes))

    prompt = f"""
Answer ONLY the requested value.

If numeric, return only the number.

Question:
{req.question}
"""

    response = model.generate_content([prompt, image])

    return {
        "answer": response.text.strip()
    }