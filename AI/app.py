import os
import json
import uuid
import sqlite3
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# Optional import for TTS
try:
    from tortoise_tts import TTS
except ImportError:
    TTS = None

# ─── Configuration ────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is required")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# ─── Filesystem & DB Setup ─────────────────────────────────────────────────────
for d in ("saved/images", "saved/audio", "saved/video", "saved/text"):
    os.makedirs(d, exist_ok=True)

if not os.path.exists("memory.json"):
    with open("memory.json", "w") as f:
        json.dump({"memory": []}, f)

def init_db():
    conn = sqlite3.connect("users.db")
    conn.execute("""
      CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        hashed_password TEXT,
        tier TEXT
      )
    """)
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# ─── Auth Utilities ───────────────────────────────────────────────────────────
def verify_password(plain, hashed):
    return pwd_ctx.verify(plain, hashed)

def get_password_hash(password):
    return pwd_ctx.hash(password)

def get_user(username: str):
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE username=?", (username,)
    ).fetchone()
    db.close()
    return user

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    creds_exc = HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Could not validate credentials",
      headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise creds_exc
    except JWTError:
        raise creds_exc
    user = get_user(username)
    if not user:
        raise creds_exc
    return user

def tier_required(user, required: str):
    if user["tier"] != required:
        raise HTTPException(403, f"{required} tier required")

# ─── Memory Helpers ────────────────────────────────────────────────────────────
def load_memory():
    with open("memory.json") as f:
        return json.load(f)

def save_memory(mem):
    with open("memory.json", "w") as f:
        json.dump(mem, f, indent=2)

def add_memory(entry: str):
    mem = load_memory()
    mem["memory"].append({"entry": entry, "ts": datetime.utcnow().isoformat()})
    save_memory(mem)

# ─── Prompt Enhancer ──────────────────────────────────────────────────────────
def enhance_prompt(prompt: str, style: str = None) -> str:
    presets = {
      "cinematic": "ultra-detailed, cinematic lighting, 4K",
      "anime": "anime style, vibrant colors, clean lines",
      "noir": "black & white, moody lighting"
    }
    suffix = presets.get(style, "")
    return f"{prompt}, {suffix}".strip(", ")

# ─── File Saver ───────────────────────────────────────────────────────────────
def save_file(kind: str, username: str, content, ext: str) -> str:
    uid = uuid.uuid4()
    path = f"saved/{kind}/{username}_{uid}{ext}"
    if kind == "text":
        with open(path, "w") as f:
            f.write(content)
    elif kind == "images":
        content.save(path)
    else:
        with open(path, "wb") as f:
            f.write(content)
    return path

# ─── Model Generators ─────────────────────────────────────────────────────────
TEXT_PIPES, IMG_PIPE, TTS_ENG = {}, None, None

def load_text_model(model="gpt2"):
    if model not in TEXT_PIPES:
        TEXT_PIPES[model] = pipeline("text-generation", model=model)
    return TEXT_PIPES[model]

def generate_text(prompt: str, model="gpt2") -> str:
    pipe = load_text_model(model)
    return pipe(prompt, max_length=200)[0]["generated_text"]

def generate_image(prompt: str) -> Image.Image:
    global IMG_PIPE
    if IMG_PIPE is None:
        IMG_PIPE = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    return IMG_PIPE(prompt).images[0]

def generate_audio(text: str) -> bytes:
    global TTS_ENG
    if TTS_ENG is None:
        if TTS is None:
            raise HTTPException(500, "TTS engine unavailable")
        TTS_ENG = TTS()
    return TTS_ENG.tts(text)

def generate_video(script: str) -> str:
    vid_path = f"saved/video/{uuid.uuid4()}.mp4"
    open(vid_path, "wb").close()
    return vid_path

# ─── Plugin System ────────────────────────────────────────────────────────────
registry = {}
def plugin(name):
    def deco(fn):
        registry[name] = fn
        return fn
    return deco

@plugin("summarizer")
async def summarizer(user, text: str):
    return text[:200] + "…"

@plugin("translator")
async def translator(user, text: str, target: str = "en"):
    return text[::-1]

@plugin("math_solver")
async def math_solver(user, expr: str):
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

# ─── Auth Endpoints ───────────────────────────────────────────────────────────
@app.post("/register")
async def register(form: OAuth2PasswordRequestForm = Depends()):
    db = get_db()
    hashed = get_password_hash(form.password)
    try:
        db.execute(
            "INSERT INTO users(username,hashed_password,tier) VALUES(?,?,?)",
            (form.username, hashed, "free")
        )
        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(400, "User already exists")
    finally:
        db.close()
    return {"status": "registered"}

@app.post("/token")
async def login_token(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ─── AI Endpoints ─────────────────────────────────────────────────────────────
@app.post("/generate-text")
async def text_ep(req: Request, current=Depends(get_current_user)):
    d = await req.json()
    text = generate_text(
        enhance_prompt(d.get("prompt", ""), d.get("style")),
        d.get("model", "gpt2")
    )
    path = save_file("text", current["username"], text, ".txt")
    add_memory(f"text → {path}")
    return {"text": text, "path": path}

@app.post("/generate-image")
async def img_ep(req: Request, current=Depends(get_current_user)):
    d = await req.json()
    img = generate_image(enhance_prompt(d.get("prompt", ""), d.get("style")))
    path = save_file("images", current["username"], img, ".png")
    add_memory(f"image → {path}")
    return FileResponse(path, media_type="image/png")

@app.post("/generate-audio")
async def audio_ep(req: Request, current=Depends(get_current_user)):
    tier_required(current, "plus")
    d = await req.json()
    audio = generate_audio(d.get("text", ""))
    path = save_file("audio", current["username"], audio, ".mp3")
    add_memory(f"audio → {path}")
    return FileResponse(path, media_type="audio/mpeg")

@app.post("/generate-video")
async def video_ep(req: Request, current=Depends(get_current_user)):
    tier_required(current, "plus")
    d = await req.json()
    vid = generate_video(d.get("script", ""))
    add_memory(f"video → {vid}")
    return FileResponse(vid, media_type="video/mp4")

@app.get("/memory")
async def get_mem(current=Depends(get_current_user)):
    return JSONResponse(load_memory())

@app.post("/memory")
async def post_mem(req: Request, current=Depends(get_current_user)):
    entry = (await req.json()).get("entry", "")
    add_memory(entry)
    return {"status": "ok"}

@app.post("/plugins/run")
async def run_plugin(req: Request, current=Depends(get_current_user)):
    d = await req.json()
    name = d.get("name")
    args = d.get("args", {})
    if name not in registry:
        raise HTTPException(404, "Plugin not found")
    result = await registry[name](current["username"], **args)
    add_memory(f"plugin {name} → {result}")
    return {"result": result}

@app.post("/set-model")
async def set_model(req: Request, current=Depends(get_current_user)):
    model = (await req.json()).get("model", "gpt2")
    load_text_model(model)
    return {"status": "model set"}

# ─── Serve Frontend ───────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
def home():
    return FileResponse("index.html", media_type="text/html")