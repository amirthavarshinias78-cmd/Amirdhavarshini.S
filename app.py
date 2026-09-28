"""
PocketSmart AI - Smart Budget & Recommendation Assistant
A single-file, runnable FastAPI backend that consolidates the Home / Party /
Jewelry budget planners, Gemini AI recommendation engine, JWT auth, session
handling and recommendation history described in the project spec.

Run:
    pip install -r requirements.txt
    # create a .env file with:
    #   GOOGLE_API_KEY=your_gemini_api_key
    #   SECRET_KEY=some_random_secret
    python app.py
    # then open http://127.0.0.1:8000
"""

import os
import re
import json
import uuid
import shutil
import asyncio
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import (
    FastAPI, HTTPException, Depends, File, UploadFile, Form,
    Request, status, Cookie
)
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv

# Gemini is optional at import time so the app can still start (and be tested)
# even if the package or API key isn't configured yet.
try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_ENABLED = bool(API_KEY) and genai is not None
if GEMINI_ENABLED:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None  # recommendation functions fall back to a clear error

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="PocketSmart AI: Smart Budget Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


# --------------------------------------------------------------------------- #
# In-memory "database" (swap for a real DB in production)
# --------------------------------------------------------------------------- #
users_db: Dict[str, Dict[str, Any]] = {}
active_sessions: Dict[str, "UserSession"] = {}
blacklisted_tokens: set = set()
user_recommendations: Dict[str, List["RecommendationRecord"]] = {}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class RegisterUser(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str


class UserInDB(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    hashed_password: str
    disabled: bool = False


class Token(BaseModel):
    access_token: str
    token_type: str


class UserSession(BaseModel):
    username: str
    login_time: datetime
    last_activity: datetime
    token: str
    user_data: Dict[str, Any] = {}


class RecommendationRecord(BaseModel):
    id: str
    timestamp: str
    recommendation_type: str
    input_summary: Dict[str, Any]
    result_summary: str
    full_result: Dict[str, Any]


class HomeBudgetInput(BaseModel):
    total_budget: float
    num_lights: int = 0
    num_fans: int = 0
    num_furniture: int = 0
    num_dining_tables: int = 0
    has_living_room: bool = False
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: Optional[str] = None


class PartyBudgetInput(BaseModel):
    total_budget: float
    num_guests: int
    party_type: str
    venue_type: Optional[str] = None
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = True
    additional_requirements: Optional[str] = None


class JewelryBudgetInput(BaseModel):
    total_budget: float
    occasion: str
    preferences: Optional[str] = None


# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def authenticate_user(db: Dict[str, Dict[str, Any]], username: str, password: str):
    user = db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return UserInDB(**user)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_token(request: Request, cookie_token: Optional[str] = Cookie(default=None, alias="access_token")) -> Optional[str]:
    """Read the token from the Authorization header first, then the cookie."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1]
    return cookie_token


async def get_current_user(request: Request, token: Optional[str] = Depends(get_token)) -> Optional[UserInDB]:
    if not token or token in blacklisted_tokens:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    user = users_db.get(username)
    if not user:
        return None
    return UserInDB(**user)


async def get_current_active_user(current_user: Optional[UserInDB] = Depends(get_current_user)) -> UserInDB:
    if current_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# --------------------------------------------------------------------------- #
# Gemini helpers
# --------------------------------------------------------------------------- #
def extract_json_from_response(text: str) -> dict:
    """Gemini sometimes wraps JSON in ```json fences or adds stray text."""
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in AI response")
    return json.loads(match.group(0))


def _require_model():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini API is not configured. Set GOOGLE_API_KEY in your .env file.",
        )


def save_upload_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join("static", "uploads", filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return path


def _shopping_links(search_terms: str, platforms: List[str]) -> Dict[str, str]:
    q = urllib.parse.quote_plus(search_terms)
    catalog = {
        "amazon": f"https://www.amazon.in/s?k={q}",
        "flipkart": f"https://www.flipkart.com/search?q={q}",
        "ikea": f"https://www.ikea.com/in/en/search/?q={q}",
        "myntra": f"https://www.myntra.com/search?q={q}",
        "ajio": f"https://www.ajio.com/search/?text={q}",
        "bigbasket": f"https://www.bigbasket.com/ps/?q={q}",
        "swiggy": f"https://www.swiggy.com/search?query={q}",
        "zomato": f"https://www.zomato.com/search?q={q}",
        "bookmyshow": f"https://in.bookmyshow.com/search?q={q}",
        "google": f"https://www.google.com/search?q={q}",
        "booking": f"https://www.booking.com/search.html?ss={q}",
        "makemytrip": f"https://www.makemytrip.com/hotels/hotel-listing/?searchText={q}",
        "oyorooms": f"https://www.oyorooms.com/search/?location={q}",
        "nobroker": f"https://www.nobroker.in/property/search?searchTerm={q}",
        "bluestone": f"https://www.bluestone.com/search.html?query={q}",
        "tanishq": f"https://www.tanishq.co.in/search?q={q}",
        "caratlane": f"https://www.caratlane.com/search?q={q}",
        "melorra": f"https://www.melorra.com/search?q={q}",
        "meesho": f"https://www.meesho.com/search?q={q}",
    }
    return {p: catalog[p] for p in platforms if p in catalog}


CATEGORY_PLATFORMS = {
    "venue": ["google", "booking", "makemytrip", "oyorooms", "nobroker"],
    "catering": ["swiggy", "zomato"],
    "food": ["swiggy", "zomato", "bigbasket", "amazon", "flipkart"],
    "decoration": ["amazon", "flipkart", "meesho", "myntra"],
    "entertainment": ["bookmyshow", "amazon", "flipkart"],
    "lighting": ["amazon", "flipkart", "ikea", "myntra", "ajio"],
    "ceiling_fans": ["amazon", "flipkart", "ikea", "myntra", "ajio"],
    "furniture": ["amazon", "flipkart", "ikea", "myntra", "ajio"],
    "dining_tables": ["amazon", "flipkart", "ikea", "myntra", "ajio"],
}
DEFAULT_PLATFORMS = ["amazon", "flipkart", "google"]
JEWELRY_PLATFORMS = ["amazon", "flipkart", "bluestone", "tanishq", "caratlane", "melorra", "meesho"]


def _attach_shopping_links(budget_breakdown: List[dict]) -> None:
    for category in budget_breakdown:
        cat_key = str(category.get("category", "")).lower().replace(" ", "_")
        platforms = CATEGORY_PLATFORMS.get(cat_key, DEFAULT_PLATFORMS)
        for item in category.get("items", []):
            terms = item.get("search_terms", "")
            if terms:
                item["shopping_links"] = _shopping_links(terms, platforms)


def get_home_recommendations(budget_input: HomeBudgetInput) -> dict:
    _require_model()
    rooms = ", ".join(
        r for r, flag in [
            ("Living Room", budget_input.has_living_room),
            ("Kitchen", budget_input.has_kitchen),
            ("Bedroom", budget_input.has_bedroom),
        ] if flag
    ) or "Not specified"

    prompt = f"""
I need interior design product recommendations for a home in India with a total
budget of INR {budget_input.total_budget:.2f}.

Requirements:
- {budget_input.num_lights} lights/lighting fixtures
- {budget_input.num_fans} ceiling fans
- {budget_input.num_furniture} furniture pieces
- {budget_input.num_dining_tables} dining tables
Rooms to furnish: {rooms}
Additional requirements: {budget_input.additional_requirements or "None"}

Respond with ONLY valid JSON (no markdown fences) in this exact structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "budget_breakdown": [
    {{
      "category": "lighting",
      "allocation": 0.0,
      "items": [
        {{"name": "", "description": "", "estimated_price": 0.0, "quantity": 0, "search_terms": ""}}
      ]
    }}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": []
}}
Keep the total cost within budget and use Indian pricing (INR).
"""
    response = model.generate_content(prompt)
    result = extract_json_from_response(response.text)
    _attach_shopping_links(result.get("budget_breakdown", []))
    _add_calculation_table(result)
    return result


def get_party_recommendations(budget_input: PartyBudgetInput) -> dict:
    _require_model()
    prompt = f"""
I need party-planning recommendations for India with a total budget of
INR {budget_input.total_budget:.2f}.

Party details:
- Type: {budget_input.party_type}
- Guests: {budget_input.num_guests}
- Venue type: {budget_input.venue_type or "Not specified"}
- Catering needed: {"Yes" if budget_input.needs_catering else "No"}
- Decoration needed: {"Yes" if budget_input.needs_decoration else "No"}
- Entertainment needed: {"Yes" if budget_input.needs_entertainment else "No"}
Additional requirements: {budget_input.additional_requirements or "None"}

Respond with ONLY valid JSON (no markdown fences) in this exact structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "budget_breakdown": [
    {{
      "category": "venue",
      "allocation": 0.0,
      "items": [
        {{"name": "", "description": "", "estimated_price": 0.0, "quantity": 0, "search_terms": ""}}
      ]
    }}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": []
}}
Keep the total cost within budget and use Indian pricing (INR).
"""
    response = model.generate_content(prompt)
    result = extract_json_from_response(response.text)
    _attach_shopping_links(result.get("budget_breakdown", []))
    _add_calculation_table(result)
    return result


def get_jewelry_recommendations(budget_input: JewelryBudgetInput, image_path: Optional[str] = None) -> dict:
    _require_model()
    base_prompt = f"""
I need jewelry recommendations for India with a total budget of
INR {budget_input.total_budget:.2f}.

Occasion: {budget_input.occasion}
Preferences: {budget_input.preferences or "Not specified"}

Respond with ONLY valid JSON (no markdown fences) in this exact structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "jewelry_recommendations": [
    {{"item_type": "", "description": "", "style": "", "estimated_price": 0.0, "search_terms": ""}}
  ],
  "remaining_budget": 0.0,
  "styling_tips": []
}}
Keep prices in INR and within budget.
"""
    if image_path:
        from PIL import Image
        img = Image.open(image_path)
        prompt = base_prompt + """
An outfit image is attached. Also include an "outfit_analysis" object with
"colors" (list), "style" and "formality" fields in the JSON, and choose
jewelry that complements the outfit.
"""
        response = model.generate_content([prompt, img])
    else:
        response = model.generate_content(base_prompt)

    result = extract_json_from_response(response.text)
    for item in result.get("jewelry_recommendations", []):
        terms = item.get("search_terms", "")
        if terms:
            item["shopping_links"] = _shopping_links(terms, JEWELRY_PLATFORMS)
    return result


def _add_calculation_table(result: dict) -> None:
    """Adds a per-category cost/percentage summary to a home or party result."""
    total_budget = result.get("total_budget", 0) or 0
    categories: Dict[str, dict] = {}
    for category in result.get("budget_breakdown", []):
        cat_name = category.get("category", "Misc")
        entry = categories.setdefault(cat_name, {"category": cat_name, "items_count": 0, "total_cost": 0.0})
        for item in category.get("items", []):
            entry["items_count"] += 1
            entry["total_cost"] += item.get("estimated_price", 0) * max(item.get("quantity", 1), 1)
    for entry in categories.values():
        entry["percentage_of_budget"] = round((entry["total_cost"] / total_budget) * 100, 2) if total_budget else 0
    result["calculation_table"] = list(categories.values())


# --------------------------------------------------------------------------- #
# History helper
# --------------------------------------------------------------------------- #
def save_to_history(username: str, recommendation_type: str, input_data: dict, result: dict) -> None:
    record = RecommendationRecord(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        recommendation_type=recommendation_type,
        input_summary=input_data,
        result_summary=f"{recommendation_type.title()} plan - budget INR {input_data.get('total_budget', 0)}",
        full_result=result,
    )
    user_recommendations.setdefault(username, []).append(record)


# --------------------------------------------------------------------------- #
# Auth routes
# --------------------------------------------------------------------------- #
@app.post("/register", status_code=201)
async def register(user: RegisterUser):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": get_password_hash(user.password),
        "disabled": False,
    }
    return {"message": "User registered successfully"}


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)

    if user.username in active_sessions:
        blacklisted_tokens.add(active_sessions[user.username].token)
        existing_user_data = active_sessions[user.username].user_data
    else:
        existing_user_data = {}

    active_sessions[user.username] = UserSession(
        username=user.username,
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        token=access_token,
        user_data=existing_user_data,
    )

    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
    )
    return response


@app.post("/logout")
async def logout(current_user: UserInDB = Depends(get_current_active_user), token: str = Depends(get_token)):
    if token:
        blacklisted_tokens.add(token)
    active_sessions.pop(current_user.username, None)
    response = JSONResponse(content={"message": "Logged out"})
    response.delete_cookie(key="access_token")
    return response


@app.get("/me")
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    return {"username": current_user.username, "email": current_user.email, "full_name": current_user.full_name}


# --------------------------------------------------------------------------- #
# Session routes
# --------------------------------------------------------------------------- #
@app.get("/session-info")
async def get_session_info(current_user: UserInDB = Depends(get_current_active_user)):
    session = active_sessions.get(current_user.username)
    if not session:
        raise HTTPException(status_code=404, detail="No active session found")
    return {
        "username": session.username,
        "login_time": session.login_time,
        "last_activity": session.last_activity,
        "session_duration_minutes": (datetime.utcnow() - session.login_time).total_seconds() // 60,
        "user_data": session.user_data,
    }


@app.post("/session-data")
async def update_session_data(data: Dict[str, Any], current_user: UserInDB = Depends(get_current_active_user)):
    session = active_sessions.get(current_user.username)
    if not session:
        raise HTTPException(status_code=404, detail="No active session found")
    session.user_data.update(data)
    session.last_activity = datetime.utcnow()
    return {"message": "Session data updated", "data": session.user_data}


# --------------------------------------------------------------------------- #
# Planner routes
# --------------------------------------------------------------------------- #
@app.post("/generate-home")
async def generate_home(budget_input: HomeBudgetInput, current_user: UserInDB = Depends(get_current_active_user)):
    result = get_home_recommendations(budget_input)
    save_to_history(current_user.username, "home", budget_input.dict(), result)
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_home_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
        }
    return result


@app.post("/generate-party")
async def generate_party(budget_input: PartyBudgetInput, current_user: UserInDB = Depends(get_current_active_user)):
    result = get_party_recommendations(budget_input)
    save_to_history(current_user.username, "party", budget_input.dict(), result)
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_party_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "party_type": budget_input.party_type,
        }
    return result


@app.post("/generate-jewelry")
async def generate_jewelry(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    current_user: UserInDB = Depends(get_current_active_user),
):
    budget_input = JewelryBudgetInput(total_budget=total_budget, occasion=occasion, preferences=preferences)

    image_path = None
    if image is not None and image.filename:
        image_path = save_upload_file(image)

    result = get_jewelry_recommendations(budget_input, image_path)

    input_summary = budget_input.dict()
    if image is not None and image.filename:
        input_summary["image"] = image.filename
    save_to_history(current_user.username, "jewelry", input_summary, result)

    return result


# --------------------------------------------------------------------------- #
# History routes
# --------------------------------------------------------------------------- #
@app.get("/recommendation-history")
async def get_recommendation_history(current_user: UserInDB = Depends(get_current_active_user)):
    records = user_recommendations.get(current_user.username, [])
    records_sorted = sorted(records, key=lambda r: r.timestamp, reverse=True)
    return {
        "history": [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "type": r.recommendation_type,
                "input": r.input_summary,
                "summary": r.result_summary,
            }
            for r in records_sorted
        ]
    }


@app.get("/recommendation-details/{recommendation_id}")
async def get_recommendation_details(recommendation_id: str, current_user: UserInDB = Depends(get_current_active_user)):
    for r in user_recommendations.get(current_user.username, []):
        if r.id == recommendation_id:
            return {
                "id": r.id,
                "timestamp": r.timestamp,
                "type": r.recommendation_type,
                "input": r.input_summary,
                "full_result": r.full_result,
            }
    raise HTTPException(status_code=404, detail="Recommendation not found")


# --------------------------------------------------------------------------- #
# Simple HTML landing page (self-contained, no template files required)
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
      <head><title>PocketSmart AI</title></head>
      <body style="font-family: sans-serif; max-width: 640px; margin: 60px auto;">
        <h1>PocketSmart AI</h1>
        <p>Backend is running. Key endpoints:</p>
        <ul>
          <li>POST /register</li>
          <li>POST /token (OAuth2 form login → returns JWT)</li>
          <li>POST /generate-home</li>
          <li>POST /generate-party</li>
          <li>POST /generate-jewelry (multipart form, optional image)</li>
          <li>GET /recommendation-history</li>
        </ul>
        <p>See interactive docs at <a href="/docs">/docs</a>.</p>
      </body>
    </html>
    """


# --------------------------------------------------------------------------- #
# Startup: background session cleanup
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def setup_session_cleanup():
    async def cleanup_expired_sessions():
        while True:
            current_time = datetime.utcnow()
            expired = [
                username for username, session in active_sessions.items()
                if (current_time - session.last_activity).total_seconds() > 1800  # 30 min
            ]
            for username in expired:
                active_sessions.pop(username, None)
            await asyncio.sleep(300)

    asyncio.create_task(cleanup_expired_sessions())


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    print("Starting PocketSmart AI Budget Planner...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
