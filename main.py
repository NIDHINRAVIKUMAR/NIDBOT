from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException, Cookie, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from datetime import datetime, timedelta
from transformers import pipeline
import os
import json

# --- Utility Imports ---
from utils.pdf_utils import extract_text_from_pdf
from utils.text_splitter import split_text
from utils.embedding_utils import create_and_save_index
from utils.rag_utils import get_answer_from_query

# --- App setup ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
os.makedirs("vector_store", exist_ok=True)

# --- Security ---
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Load & Save Users ---
def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

# --- Auth Helpers ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    users = load_users()
    user = users.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- Pydantic Model ---
class Question(BaseModel):
    query: str

# --- Load General Model ---
general_chat_model = pipeline("text2text-generation", model="google/flan-t5-base")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def welcome(request: Request, signup_success: bool = False):
    return templates.TemplateResponse("welcome.html", {"request": request, "signup_success": signup_success})

@app.post("/signup")
async def signup(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    users = load_users()

    if username in users:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = pwd_context.hash(password)
    users[username] = {
        "username": username,
        "email": email,
        "hashed_password": hashed_password
    }

    save_users(users)
    return RedirectResponse(url="/?signup_success=true", status_code=302)

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(data={"sub": user["username"]})
    response = RedirectResponse(url="/chat", status_code=302)
    response.set_cookie(key="access_token", value=f"Bearer {token}", httponly=True)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("access_token")
    return response

@app.get("/chat", response_class=HTMLResponse)
async def chatbot_ui(request: Request, access_token: str = Cookie(default=None), mode: str = "pdf"):
    if not access_token:
        return RedirectResponse(url="/", status_code=302)
    try:
        token = access_token.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise JWTError()
    except JWTError:
        return RedirectResponse(url="/", status_code=302)

    return templates.TemplateResponse("index.html", {"request": request, "mode": mode})

@app.post("/upload-pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    try:
        contents = await pdf_file.read()
        file_path = f"temp_{pdf_file.filename}"
        with open(file_path, "wb") as f:
            f.write(contents)

        text = extract_text_from_pdf(file_path)
        chunks = split_text(text)
        create_and_save_index(chunks)
        os.remove(file_path)

        return {"message": "PDF processed and index saved successfully!"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@app.post("/ask/pdf")
async def ask_pdf_question(q: Question, access_token: str = Cookie(default=None)):
    try:
        token = access_token.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        answer = get_answer_from_query(q.query)
        if not answer or answer.strip() == "":
            answer = ("Sorry, I couldn't find the answer in the uploaded PDF. "
                      "Please try another question or switch to General Mode.")
        return {"answer": answer}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

@app.post("/ask/general")
async def ask_general_question(q: Question, access_token: str = Cookie(default=None)):
    try:
        token = access_token.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        answer = general_chat_model(q.query, max_length=200)[0]['generated_text']
        return {"answer": answer}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


