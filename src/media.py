"""Voice notes, screenshots and files attached in the chat.

Speech-to-text (Whisper) and image reading (a vision model) run on Groq, so they need GROQ_API_KEY.
Text files and PDFs are read locally.
"""

import base64
import io
import os
import re

from src import config

IMAGE_TYPES = ("png", "jpg", "jpeg", "webp", "gif")
TEXT_TYPES = ("txt", "log", "md", "csv", "json", "xml", "ini", "cfg", "yaml", "yml")
DOC_TYPES = ("pdf",)
FILE_TYPES = IMAGE_TYPES + TEXT_TYPES + DOC_TYPES

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

VISION_PROMPT = (
    "You are helping an IT helpdesk. Describe this screenshot for a support engineer in at most 5 short lines: "
    "the app or device shown, the exact text of any error message or code, and what the user seems to be doing. "
    "Do not suggest fixes."
)


def groq_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _client():
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")


def kind_of(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in IMAGE_TYPES:
        return "image"
    if ext in DOC_TYPES:
        return "pdf"
    return "text"


def transcribe(audio: bytes, filename: str = "voice.wav") -> str:
    """Speech to text with Whisper on Groq."""
    result = _client().audio.transcriptions.create(
        model=config.STT_MODEL, file=(filename, audio), language=config.STT_LANGUAGE or None
    )
    return result.text.strip()


def describe_image(image: bytes, mime: str) -> str:
    """Read the error text and context out of a screenshot with a vision model."""
    data_url = f"data:{mime or 'image/png'};base64,{base64.b64encode(image).decode()}"
    result = _client().chat.completions.create(
        model=config.VISION_MODEL,
        temperature=0,
        max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
    )
    return _THINK_RE.sub("", result.choices[0].message.content or "").strip()


def extract_text(name: str, data: bytes, limit: int = config.MAX_ATTACHMENT_CHARS) -> str:
    """Plain text from a text file or PDF, truncated to keep the prompt small."""
    if kind_of(name) == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages[:20])
    else:
        text = data.decode("utf-8", errors="replace")
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "\n…(truncated)"


def thumbnail_data_url(image: bytes, size: int = 360) -> str:
    """Small JPEG preview so the chat history can show the screenshot without storing the full file."""
    from PIL import Image

    img = Image.open(io.BytesIO(image))
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_prompt(question: str, attachments: list[dict]) -> str:
    """The question plus what the attachments contain, as sent to the answering LLM."""
    parts = [question]
    for a in attachments:
        if a["kind"] == "image":
            parts.append(f"Attached screenshot ({a['name']}) shows:\n{a['summary']}")
        else:
            parts.append(f"Attached file {a['name']}:\n```\n{a['summary']}\n```")
    return "\n\n".join(parts)


def search_text(question: str, attachments: list[dict]) -> str:
    """Retrieval query: the question plus the first part of each attachment (error text matters most)."""
    extra = " ".join(a["summary"][:300] for a in attachments)
    return f"{question} {extra}".strip()
