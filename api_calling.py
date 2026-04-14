from google import genai
from dotenv import load_dotenv
import os
from gtts import gTTS
import streamlit as st
import io
from google.genai import types
from PIL import Image

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def pil_to_part(img: Image.Image) -> types.Part:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return types.Part.from_bytes(
        data=buffer.getvalue(),
        mime_type="image/png"
    )


def note_generator(images: list) -> str:
    prompt = """
        Analyze the provided images and generate a well-structured, formal, and professional note.
        Language Requirement:
        - The final output MUST be written entirely in Bangla (বাংলা)
        - Do NOT use any English words in the final answer unless absolutely necessary (e.g., technical terms)
        Writing Style:
        - No emojis
        - Formal, academic, and professional tone
        - Easy to read, clear, and suitable for students
        - Use simple and understandable language
        - Avoid unnecessary complexity
        Structure (Strictly follow this format):
        1. শিরোনাম (Topic Title)
        2. সংজ্ঞা (Definition)
        3. বিস্তারিত ব্যাখ্যা (Detailed Explanation)
        4. উদাহরণ (Example)
        5. গুরুত্বপূর্ণ পয়েন্ট (Key Points / Summary)
        Additional Instructions:
        - Organize the content in bullet points where necessary
        - Highlight important terms or keywords
        - If multiple concepts are present, separate them clearly
        - Maintain logical flow and clarity
        - Make the note suitable for exam preparation and quick revision
    """

    parts = [pil_to_part(img) for img in images] 
    parts.append(prompt)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=parts
    )
    return response.text


def audio_transcription(text: str) -> io.BytesIO:
    speech = gTTS(text=text, lang='bn', slow=False)
    audio_buffer = io.BytesIO()
    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0) 
    return audio_buffer


def quiz_generator(images: list, difficulty: str) -> str:
    prompt = f"""
        Analyze the provided images and generate a quiz based on the content.
        Language Requirement:
        - The final output MUST be written entirely in Bangla (বাংলা)
        - Do NOT use English in the final answer unless absolutely necessary
        Quiz Style:
        - Formal and educational tone
        - Suitable for students and exam preparation
        - Clear and simple language
        - No emojis
        - Difficulty: {difficulty}
        Quiz Structure:
        - Create multiple-choice questions (MCQ)
        - Each question must have 4 options (A, B, C, D), each option on a new line
        - Clearly indicate the correct answer after each question
        Additional Instructions:
        - Cover all important concepts from the images
        - Do not repeat questions
        - Provide answer explanation in 1-2 lines if needed
    """

    parts = [pil_to_part(img) for img in images]  
    parts.append(prompt)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=parts  
    )
    return response.text
