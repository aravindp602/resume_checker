import os
import io
import json
import pdfplumber
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

app = FastAPI(title="Resume Reality Checker Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq Client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
MODEL_NAME = "llama-3.1-8b-instant" 

# --- HELPER FUNCTIONS ---
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
    try:
        return json.loads(content)
    except:
        return {
            "total_score": 0,
            "summary": "AI parsing failed. The resume format might be too complex or empty.",
            "section_scores": {
                "skills": 0,
                "projects": 0,
                "formatting": 0,
                "ats_keywords": 0
            },
            "strengths": [],
            "weaknesses": ["AI response parsing failed. Try a simpler PDF layout."],
            "missing_keywords": [],
            "ats_issues": [],
            "improvement_suggestions": [],
            "rewrite_examples": []
        }

def normalize_scores(data):
    try:
        total = sum(data.get("section_scores", {}).values())
        data["total_score"] = min(total, 100)
    except:
        data["total_score"] = 0
    return data

# --- API ENDPOINT ---
@app.post("/api/analyze")
async def analyze_resume(file: UploadFile = File(...), role: str = Form(...)):
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    resume_text = extract_text(file_bytes)
    resume_text = clean_text(resume_text)

    if len(resume_text) < 50:
        raise HTTPException(status_code=400, detail="Empty or unreadable resume.")

    prompt = f"""
You are an Applicant Tracking System (ATS) and Senior Technical Recruiter at a top-tier product company.

Your job is to STRICTLY evaluate the resume for the role: {role}.

----------------------------------------
EVALUATION CRITERIA (STRICT SCORING)
----------------------------------------

1. Skills Match (30 points)
- Relevant technologies for {role}
- Depth vs shallow listing
- Industry-standard tools

2. Projects Quality (25 points)
- Real-world complexity
- Use of modern tech stack
- Measurable impact (%, scale, performance)
- Problem-solving depth

3. Formatting & Structure (15 points)
- Clean sections (Education, Skills, Projects, Experience)
- Bullet clarity
- Readability (ATS-friendly, no tables/graphics)

4. ATS Keyword Optimization (30 points)
- Presence of role-specific keywords
- Alignment with job descriptions
- Avoid keyword stuffing

----------------------------------------
STRICT RULES
----------------------------------------

- Be critical and realistic (no inflated scores)
- Penalize generic resumes heavily
- Penalize missing metrics
- Penalize irrelevant skills for {role}
- Do NOT give vague feedback
- All feedback must be actionable

----------------------------------------
ATS BEST PRACTICES CHECK
----------------------------------------

Check if resume follows:
- 1 page (or max 2 pages)
- Uses bullet points
- Uses action verbs (Developed, Built, Optimized)
- Contains measurable results (%, x improvement)
- Avoids paragraphs
- No images, icons, or fancy formatting
- Includes GitHub/portfolio (if technical role)

----------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
----------------------------------------

Return ONLY valid JSON. No markdown. No explanation.

{{
  "total_score": 85,
  "summary": "2-3 line professional evaluation",
  "section_scores": {{
    "skills": 25,
    "projects": 20,
    "formatting": 15,
    "ats_keywords": 25
  }},
  "strengths": [
    "Clear, strong technical strength"
  ],
  "weaknesses": [
    "Specific critical issue"
  ],
  "missing_keywords": [
    "Docker", "Kubernetes", "System Design"
  ],
  "ats_issues": [
    "Uses paragraphs instead of bullet points",
    "Missing measurable achievements"
  ],
  "improvement_suggestions": [
    "Rewrite project bullet: 'Improved API response time by 35% using caching'",
    "Add role-specific tools like AWS, Docker"
  ],
  "rewrite_examples": [
    "Built a web app → Developed a scalable web application handling 10,000+ users"
  ]
}}

----------------------------------------
RESUME
----------------------------------------

{resume_text[:4000]}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict JSON generator. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content.strip()
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()

        parsed = safe_json_parse(raw_content)
        parsed = normalize_scores(parsed)

        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)