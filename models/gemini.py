import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

def gemini_chat(user_input, history):
    response = model.generate_content(user_input)
    reply = response.text.strip()
    history.append((user_input, reply))
    return reply, "Normal"