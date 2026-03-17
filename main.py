import os
import io
import json
import pdfplumber
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Resume Sight Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
MODEL_NAME = "llama-3.1-8b-instant" 

def extract_text(pdf_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read PDF file.")

def clean_text(text: str):
    return text.replace("\x00", "").strip()

def safe_json_parse(content: str):
    """Ensures the UI never sees 'undefined' by providing defaults for every key."""
    try:
        data = json.loads(content)
        # Ensure every key exists
        defaults = {
            "total_score": 0,
            "summary": "Analysis complete.",
            "section_scores": {"skills": 0, "projects": 0, "formatting": 0, "ats_keywords": 0},
            "health_check": {"has_email": False, "has_phone": False, "has_linkedin": False, "has_github": False, "metrics_count": 0},
            "action_verbs": [],
            "weak_words": [],
            "strengths": [],
            "weaknesses": [],
            "missing_keywords": [],
            "priority_fixes": [],
            "rewrite_examples": []
        }
        for key, val in defaults.items():
            if key not in data:
                data[key] = val
        return data
    except:
        return defaults

@app.post("/api/analyze")
async def analyze_resume(file: UploadFile = File(...), role: str = Form(...), jd: str = Form(default="")):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF supported.")

    file_bytes = await file.read()
    resume_text = clean_text(extract_text(file_bytes))

    if len(resume_text) < 50:
        raise HTTPException(status_code=400, detail="Empty or unreadable resume.")

    target = f"Job Description:\n{jd}" if jd.strip() else f"General Role: {role}"

    prompt = f"""
    You are a professional ATS and Senior Recruiter. 
    Evaluate this resume against: {target}
    
    Return ONLY a JSON object with this exact structure:
    {{
      "total_score": 85,
      "summary": "Professional 2-sentence summary.",
      "section_scores": {{"skills": 25, "projects": 20, "formatting": 15, "ats_keywords": 25}},
      "health_check": {{
        "has_email": true, "has_phone": true, "has_linkedin": true, "has_github": true, "metrics_count": 5
      }},
      "action_verbs": ["Architected", "Optimized"],
      "weak_words": ["Worked on", "Helped"],
      "strengths": ["Strong tech stack"],
      "weaknesses": ["Lacks metrics"],
      "missing_keywords": ["Docker", "Kubernetes"],
      "priority_fixes": [{{"priority": "High", "task": "Add metrics"}}],
      "rewrite_examples": ["Old bullet → New bullet with metrics"]
    }}

    RESUME TEXT:
    {resume_text[:4000]}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "You are a JSON-only API."}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        parsed = safe_json_parse(response.choices[0].message.content)
        parsed["raw_text"] = resume_text
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)