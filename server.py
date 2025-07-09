# AI Platform Backend (server.py)
# All-in-one backend: text gen, spell check, image, audio, video, smart routing, logging, file output

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import language_tool_python
import torch, json, os, uuid
from datetime import datetime
from io import BytesIO
import base64
from PIL import Image
import subprocess

app = FastAPI()

# === Load Models ===
print("Loading GPT model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
if torch.cuda.is_available():
    model.to("cuda")

print("Loading Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

print("Loading LanguageTool...")
tool = language_tool_python.LanguageTool('en-US')

# === Data Models ===
class AIRequest(BaseModel):
    mode: str  # text | image | correct | audio | video
    prompt: str
    max_length: int = 100

# === Helper Functions ===
def correct_spelling(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

def log_json(data):
    os.makedirs("logs", exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"logs/{now}.json", "w") as f:
        json.dump(data, f, indent=2)

def save_audio(text, filename="output.mp3"):
    # Requires TTS engine like Coqui TTS installed via CLI (or replace with pyttsx3 fallback)
    path = f"media/audio/{filename}"
    os.makedirs("media/audio", exist_ok=True)
    subprocess.run(["tts", "--text", text, "--out_path", path])
    return path

def save_video(text, filename="output.mp4"):
    # Generate a static image and convert it into a simple video
    os.makedirs("media/video", exist_ok=True)
    image = pipe(text).images[0]
    image_path = f"media/video/{filename.replace('.mp4', '.png')}"
    video_path = f"media/video/{filename}"
    image.save(image_path)
    subprocess.run(["ffmpeg", "-loop", "1", "-i", image_path, "-c:v", "libx264", "-t", "4", "-pix_fmt", "yuv420p", "-vf", "scale=640:360", video_path, "-y"])
    return video_path

# === Main Endpoint ===
@app.post("/generate")
def generate(req: AIRequest):
    corrected_prompt = correct_spelling(req.prompt)
    result = {"original": req.prompt, "corrected": corrected_prompt, "mode": req.mode}

    try:
        if req.mode == "text":
            inputs = tokenizer.encode(corrected_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            outputs = model.generate(inputs, max_length=req.max_length, do_sample=True, temperature=0.7)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result["response"] = text

        elif req.mode == "image":
            image = pipe(corrected_prompt).images[0]
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            result["response"] = img_str

        elif req.mode == "correct":
            result["response"] = corrected_prompt

        elif req.mode == "audio":
            audio_filename = f"{uuid.uuid4()}.mp3"
            path = save_audio(corrected_prompt, audio_filename)
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            result["audio_base64"] = data
            result["file"] = path

        elif req.mode == "video":
            video_filename = f"{uuid.uuid4()}.mp4"
            path = save_video(corrected_prompt, video_filename)
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            result["video_base64"] = data
            result["file"] = path

        else:
            raise HTTPException(status_code=400, detail="Unknown mode: must be text, image, correct, audio, or video")

        log_json(result)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))