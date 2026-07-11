import os
import io
import re
import base64

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai

load_dotenv()

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


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/answer-image")
async def answer(req: RequestData):
    try:
        image_b64 = req.image_base64.strip()

        # Handle data URLs
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        # Fix missing padding
        image_b64 += "=" * (-len(image_b64) % 4)

        image_bytes = base64.b64decode(image_b64)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = f"""
You are an OCR and document understanding assistant.

Answer ONLY the user's question using the information in the image.

Rules:
1. Return ONLY the final answer.
2. Do NOT explain.
3. Do NOT include labels.
4. Do NOT include markdown.
5. If the answer is numeric, return only the number.
6. Do NOT include currency symbols.
7. Do NOT include units.
8. Preserve decimal places exactly if present.

Question:
{req.question}
"""

        response = model.generate_content([prompt, image])

        answer = response.text.strip()

        # Remove markdown formatting
        answer = answer.replace("**", "")
        answer = answer.replace("`", "")
        answer = answer.replace("\n", " ")
        answer = answer.strip()

        # Collapse multiple spaces
        answer = re.sub(r"\s+", " ", answer)

        return {"answer": answer}

    except Exception as e:
        return {"answer": str(e)}