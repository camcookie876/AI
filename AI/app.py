import os
import io
import json
import uuid
import sqlite3
import requests
from datetime import datetime, timedelta
from urllib.parse import urljoin

from fastapi import (FastAPI, Request, Depends, HTTPException,
                     UploadFile, File, status)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import (OAuth2PasswordBearer,
                              OAuth2PasswordRequestForm)
from fastapi.security.utils import get_authorization_scheme_param
from jose import jwt, JWTError
from passlib.context import CryptContext
from bs4 import BeautifulSoup
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# ─── Config ───────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY env var required")

ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
FREE_STORAGE_LIMIT         = 50 * 1024 * 1024  # 50 MB
FREE_RESET_DAYS            = 30

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2   = OAuth2PasswordBearer(tokenUrl="token")
app      = FastAPI()

# ─── Filesystem & Memory ──────────────────────────────────────────────────────
for d in ("saved/images","saved/audio","saved/video","saved/text"):
    os.makedirs(d, exist_ok=True)
if not os.path.exists("memory.json"):
    with open("memory.json","w") as f:
        json.dump({"memory":[]}, f)

def load_memory():
    return json.load(open("memory.json"))

def save_memory(m):
    json.dump(m, open("memory.json","w"), indent=2)

def add_memory(entry: str):
    m = load_memory()
    m["memory"].append({"entry":entry,"ts":datetime.utcnow().isoformat()})
    save_memory(m)

# ─── DB Setup ────────────────────────────────────────────────────────────────
def init_db():
    c = sqlite3.connect("users.db")
    c.execute("""CREATE TABLE IF NOT EXISTS users(
                   id INTEGER PRIMARY KEY,
                   username TEXT UNIQUE,
                   hashed_password TEXT,
                   tier TEXT
                 )""")
    c.execute("""CREATE TABLE IF NOT EXISTS usage(
                   username TEXT PRIMARY KEY,
                   used_bytes INTEGER,
                   first_upload TEXT
                 )""")
    c.execute("""CREATE TABLE IF NOT EXISTS files(
                   id TEXT PRIMARY KEY,
                   username TEXT,
                   path TEXT,
                   kind TEXT,
                   created TEXT,
                   visibility TEXT
                 )""")
    c.execute("""CREATE TABLE IF NOT EXISTS shared_chats(
                   id TEXT PRIMARY KEY,
                   owner TEXT,
                   created TEXT
                 )""")
    c.execute("""CREATE TABLE IF NOT EXISTS chat_permissions(
                   share_id TEXT,
                   username TEXT,
                   PRIMARY KEY(share_id,username)
                 )""")
    c.commit(); c.close()
init_db()

def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# ─── Auth Helpers ────────────────────────────────────────────────────────────
def verify_password(p, h): return pwd_ctx.verify(p, h)
def get_password_hash(p):   return pwd_ctx.hash(p)

def get_user(u):
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE username=?", (u,)).fetchone()
    db.close()
    return row

def authenticate_user(u, p):
    user = get_user(u)
    if not user or not verify_password(p, user["hashed_password"]):
        return None
    return user

def create_token(data, exp=None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (exp or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_token_opt(req: Request):
    auth = req.headers.get("Authorization")
    if not auth:
        return None
    scheme, tok = get_authorization_scheme_param(auth)
    return tok if scheme.lower() == "bearer" else None

async def current_user_opt(token: str = Depends(get_token_opt)):
    if not token:
        return {"username":"guest","tier":"free"}
    exc = HTTPException(401, "Invalid credentials",
                        headers={"WWW-Authenticate":"Bearer"})
    try:
        p = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        u = p.get("sub")
        if not u: raise exc
    except JWTError:
        raise exc
    user = get_user(u)
    if not user:
        raise exc
    return user

def tier_required(user, req_tier):
    order = ["free","plus","pro"]
    if order.index(user["tier"]) < order.index(req_tier):
        raise HTTPException(403, f"{req_tier} tier required")

# ─── Usage Tracking ────────────────────────────────────────────────────────────
def check_usage(user, filepath):
    size = os.path.getsize(filepath)
    db = get_db()
    row = db.execute(
        "SELECT used_bytes,first_upload FROM usage WHERE username=?", (user,)
    ).fetchone()
    now = datetime.utcnow()
    if not row:
        db.execute(
            "INSERT INTO usage(username,used_bytes,first_upload) VALUES(?,?,?)",
            (user, size, now.isoformat())
        )
        db.commit(); db.close()
        return
    used, fu = row["used_bytes"], datetime.fromisoformat(row["first_upload"])
    if now - fu >= timedelta(days=FREE_RESET_DAYS):
        used, fu = 0, now
    if used + size > FREE_STORAGE_LIMIT:
        reset_date = (fu + timedelta(days=FREE_RESET_DAYS)).date()
        db.close()
        raise HTTPException(403,
            f"Free cap reached. Retry after {reset_date}")
    db.execute(
        "UPDATE usage SET used_bytes=?,first_upload=? WHERE username=?",
        (used+size, fu.isoformat(), user)
    )
    db.commit(); db.close()

# ─── File Recorder ────────────────────────────────────────────────────────────
def record_file(user, path, kind, vis):
    fid     = os.path.basename(path)
    created = datetime.utcnow().isoformat()
    db = get_db()
    db.execute("""
      INSERT INTO files(id,username,path,kind,created,visibility)
      VALUES(?,?,?,?,?,?)
    """, (fid, user, path, kind, created, vis))
    db.commit(); db.close()
    add_memory(f"[{user}] {kind} saved → {path}")

# ─── AI Models ────────────────────────────────────────────────────────────────
TEXT_P, IMG_P, TTS_E = {}, None, None

def load_text(m="gpt2"):
    if m not in TEXT_P:
        TEXT_P[m] = pipeline("text-generation", model=m)
    return TEXT_P[m]

def gen_text(prompt, m="gpt2"):
    return load_text(m)(prompt, max_length=200)[0]["generated_text"]

def gen_img(prompt):
    global IMG_P
    if not IMG_P:
        IMG_P = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5")
    return IMG_P(prompt).images[0]

def gen_audio(txt):
    global TTS_E
    if not TTS_E:
        from tortoise_tts import TTS
        TTS_E = TTS()
    return TTS_E.tts(txt)

def gen_video(script):
    vid = f"saved/video/{uuid.uuid4()}.mp4"
    open(vid,"wb").close()
    return vid

# ─── Crawling Endpoint ────────────────────────────────────────────────────────
@app.post("/crawl-page")
async def crawl_page(req: Request):
    d   = await req.json()
    url = d.get("url")
    if not url:
        raise HTTPException(400, "Missing URL")
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        paras = [p.get_text().strip() for p in soup.find_all("p")][:5]
        imgs  = [
          urljoin(url, img["src"])
          for img in soup.find_all("img")
          if img.get("src")
        ][:5]
        return {"paragraphs": paras, "images": imgs}
    except Exception as e:
        raise HTTPException(400, f"Crawl failed: {e}")

# ─── Shared Chats (Plus-Only) ────────────────────────────────────────────────
@app.post("/chats/share")
async def create_share(current=Depends(current_user_opt)):
    tier_required(current, "plus")
    share_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    db = get_db()
    db.execute(
      "INSERT INTO shared_chats(id,owner,created) VALUES(?,?,?)",
      (share_id, current["username"], now)
    )
    db.execute(
      "INSERT INTO chat_permissions(share_id,username) VALUES(?,?)",
      (share_id, current["username"])
    )
    db.commit(); db.close()
    return {"share_id": share_id}

@app.get("/chats/share")
async def list_shares(current=Depends(current_user_opt)):
    tier_required(current, "plus")
    db = get_db()
    rows = db.execute(
      "SELECT id,created FROM shared_chats WHERE owner=?",
      (current["username"],)
    ).fetchall()
    db.close()
    return {"shares": [{"id":r["id"], "created":r["created"]} for r in rows]}

@app.get("/chats/share/{sid}")
async def get_shared_chat(sid: str, current=Depends(current_user_opt)):
    db = get_db()
    ok = db.execute(
      "SELECT 1 FROM chat_permissions WHERE share_id=? AND username=?",
      (sid, current["username"])
    ).fetchone()
    db.close()
    if not ok:
        raise HTTPException(403, "No permission for this share")
    return JSONResponse(load_memory())

@app.post("/chats/share/{sid}/permissions")
async def edit_share_perms(sid: str, req: Request,
                          current=Depends(current_user_opt)):
    tier_required(current, "plus")
    d      = await req.json()
    action = d.get("action")
    user   = d.get("username")
    db     = get_db()
    owner  = db.execute(
      "SELECT owner FROM shared_chats WHERE id=?", (sid,)
    ).fetchone()
    if not owner or owner["owner"] != current["username"]:
        raise HTTPException(403, "Not share owner")
    if action=="add":
        db.execute(
          "INSERT OR IGNORE INTO chat_permissions(share_id,username) VALUES(?,?)",
          (sid, user)
        )
    elif action=="remove":
        db.execute(
          "DELETE FROM chat_permissions WHERE share_id=? AND username=?",
          (sid, user)
        )
    else:
        raise HTTPException(400, "Invalid action")
    db.commit(); db.close()
    return {"status":"ok"}

# ─── New Fetchers ─────────────────────────────────────────────────────────────
@app.post("/fetch/wiki")
async def fetch_wiki(req: Request):
    d     = await req.json()
    title = d.get("title")
    if not title:
        raise HTTPException(400, "Missing title")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    resp = requests.get(url, timeout=5)
    if resp.status_code!=200:
        raise HTTPException(404, "Page not found")
    data = resp.json()
    return {"extract": data.get("extract")}

@app.post("/fetch/crypto")
async def fetch_crypto(req: Request):
    d   = await req.json()
    ids = d.get("ids")
    if not ids:
        raise HTTPException(400, "Missing ids")
    url = ("https://api.coingecko.com/api/v3/simple/price"
           f"?ids={ids}&vs_currencies=usd")
    resp = requests.get(url, timeout=5).json()
    return {"price": resp.get(ids, {}).get("usd")}

@app.post("/fetch/time")
async def fetch_time(req: Request):
    d = await req.json()
    tz = d.get("timezone", "Etc/UTC")
    url = f"http://worldtimeapi.org/api/timezone/{tz}"
    resp = requests.get(url, timeout=5)
    if resp.status_code!=200:
        raise HTTPException(400, "Invalid timezone")
    return resp.json()

@app.post("/fetch/weather")
async def fetch_weather(req: Request, current=Depends(current_user_opt)):
    tier_required(current, "plus")
    d   = await req.json()
    lat = d.get("latitude"); lon = d.get("longitude")
    if lat is None or lon is None:
        raise HTTPException(400, "Missing coords")
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&current_weather=true")
    resp = requests.get(url, timeout=5).json()
    return {"weather": resp.get("current_weather")}

# ─── Register / Token ────────────────────────────────────────────────────────
@app.post("/register")
async def register(form: OAuth2PasswordRequestForm=Depends()):
    db     = get_db()
    hashed = get_password_hash(form.password)
    try:
        db.execute(
          "INSERT INTO users(username,hashed_password,tier) VALUES(?,?,?)",
          (form.username, hashed, "free")
        )
        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(400, "User exists")
    finally:
        db.close()
    return {"status":"registered"}

@app.post("/token")
async def token(form: OAuth2PasswordRequestForm=Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(401, "Bad credentials")
    tok = create_token(
      {"sub":user["username"]},
      exp=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": tok, "token_type":"bearer"}

# ─── AI Endpoints ─────────────────────────────────────────────────────────────
@app.post("/generate-text")
async def generate_text_ep(req: Request, curr=Depends(current_user_opt)):
    d    = await req.json()
    out  = gen_text(d.get("prompt",""), d.get("model","gpt2"))
    user = curr["username"]
    # save file if not guest
    if user!="guest":
        path=f"saved/text/{user}_{uuid.uuid4()}.txt"
        open(path,"w").write(out)
        if curr["tier"]=="free":
            check_usage(user,path)
            record_file(user,path,"text","public")
        else:
            record_file(user,path,"text",d.get("visibility","private"))
    add_memory(f"[{user}] AI → {out}")
    return {"text":out}

@app.post("/generate-image")
async def generate_image_ep(req: Request, curr=Depends(current_user_opt)):
    d    = await req.json()
    img  = gen_img(d.get("prompt",""))
    user = curr["username"]
    if user!="guest":
        path = f"saved/images/{user}_{uuid.uuid4()}.png"
        img.save(path)
        if curr["tier"]=="free":
            check_usage(user,path)
            record_file(user,path,"image","public")
        else:
            record_file(user,path,"image",d.get("visibility","private"))
        return FileResponse(path, media_type="image/png")
    # guest: stream bytes
    buf = io.BytesIO(); img.save(buf,"PNG"); buf.seek(0)
    return FileResponse(buf, media_type="image/png")

@app.post("/generate-audio")
async def generate_audio_ep(req: Request, curr=Depends(current_user_opt)):
    tier_required(curr,"plus")
    d    = await req.json()
    data = gen_audio(d.get("text",""))
    path = f"saved/audio/{curr['username']}_{uuid.uuid4()}.mp3"
    open(path,"wb").write(data)
    record_file(curr["username"],path,"audio",d.get("visibility","private"))
    return FileResponse(path, media_type="audio/mpeg")

@app.post("/generate-video")
async def generate_video_ep(req: Request, curr=Depends(current_user_opt)):
    tier_required(curr,"plus")
    d   = await req.json()
    vid = gen_video(d.get("script",""))
    record_file(curr["username"],vid,"video",d.get("visibility","private"))
    return FileResponse(vid, media_type="video/mp4")

@app.post("/upload-image")
async def upload_image_ep(file: UploadFile=File(...),
                          curr=Depends(current_user_opt)):
    img  = Image.open(io.BytesIO(await file.read()))
    user = curr["username"]
    if user!="guest":
        path = f"saved/images/{user}_{uuid.uuid4()}.png"
        img.save(path)
        if curr["tier"]=="free":
            check_usage(user,path)
            record_file(user,path,"image","public")
        else:
            record_file(user,path,"image","private")
        return {"filename":os.path.basename(path)}
    return {"filename":None}

@app.get("/memory")
async def get_memory_ep(curr=Depends(current_user_opt)):
    return JSONResponse(load_memory())

@app.get("/", response_class=FileResponse)
def home():
    return FileResponse("index.html", media_type="text/html")