from typing import Literal
from openai import OpenAI
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # lädt keys.env

# API Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

ModelName = Literal["gpt4", "gemini"]

def gpt4_answer(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def gemini_answer(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_answer(model_name: ModelName, prompt: str) -> str:
    if model_name == "gpt4":
        return gpt4_answer(prompt)
    elif model_name == "gemini":
        return gemini_answer(prompt)
    else:
        raise ValueError(f"Unknown model: {model_name}")
