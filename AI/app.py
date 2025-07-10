from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

app = FastAPI()

# ─── Code Generation Pipeline ────────────────────────────────────────────────
# Replace with your favorite code‐generation model
code_pipe = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

@app.post("/generate-code")
async def generate_code(req: Request):
    """
    POST JSON:
      {
        "prompt": "Write a Python function to reverse a list",
        "url": "https://example.com/specs"   # optional
      }
    Returns:
      { "code": "...generated snippet..." }
    """
    data = await req.json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(400, "Missing `prompt` in request body")

    # 1) If user provided a URL, fetch up to 5 paragraphs for context
    url = data.get("url", "").strip()
    context_text = ""
    if url:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            paras = [p.get_text().strip() for p in soup.find_all("p")][:5]
            if paras:
                context_text = "## Context from URL ##\n" + "\n\n".join(paras) + "\n\n"
        except Exception as e:
            raise HTTPException(400, f"Failed to crawl URL: {e}")

    # 2) Build the final prompt
    full_prompt = context_text + prompt

    # 3) Generate code (adjust max_length as needed)
    generated = code_pipe(full_prompt, max_length=200, do_sample=True)[0]["generated_text"]

    return {"code": generated}


# ─── Web Crawler Endpoint ────────────────────────────────────────────────────
@app.post("/crawl")
async def crawl_page(req: Request):
    """
    POST JSON:
      { "url": "https://example.com" }
    Returns:
      {
        "paragraphs": [...up to 5 strings...],
        "images": [...up to 5 URLs...]
      }
    """
    data = await req.json()
    url = data.get("url", "").strip()
    if not url:
        raise HTTPException(400, "Missing `url` in request body")

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")

    paragraphs = [p.get_text().strip() for p in soup.find_all("p")][:5]
    images = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        images.append(urljoin(url, src))
        if len(images) >= 5:
            break

    return {"paragraphs": paragraphs, "images": images}