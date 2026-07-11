import base64
import io
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

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


@app.post("/answer-image")
async def answer(req: RequestData):
    try:
        image_b64 = req.image_base64.strip()

        # Support data URLs like:
        # data:image/png;base64,...
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        # Fix missing padding if necessary
        image_b64 += "=" * (-len(image_b64) % 4)

        image_bytes = base64.b64decode(image_b64)

        image = Image.open(io.BytesIO(image_bytes))

        prompt = f"""
Answer ONLY the answer.

If the answer is numeric,
return ONLY the number as a string.

Do not include units,
currency symbols,
or explanations.

Question:
{req.question}
"""

        response = model.generate_content([prompt, image])

        return {"answer": response.text.strip()}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}