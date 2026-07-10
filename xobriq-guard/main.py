import hashlib
import io
import os
import re
import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import pdfplumber
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from newsapi import NewsApiClient
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

from agent import assess
from schema import Case, RiskReport
from sanctions_check import check_name

app = FastAPI(title="Xobriq Guard")

static_dir = Path(__file__).parent / "static"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize NewsAPI
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None


@app.on_event("startup")
async def startup_event() -> None:
    if not os.getenv("FIREWORKS_API_KEY"):
        raise RuntimeError(
            "FIREWORKS_API_KEY is not set. Add it to your environment or .env file before starting the server."
        )


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")


# ---------- Existing endpoints ----------
@app.post("/screen", response_model=RiskReport)
async def screen_case(case: Case) -> RiskReport:
    try:
        return await assess(case)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class SanctionsRequest(BaseModel):
    name: str


@app.post("/sanctions-check")
async def sanctions_check(payload: SanctionsRequest) -> dict:
    try:
        matches = check_name(payload.name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "name": payload.name,
        "flagged": len(matches) > 0,
        "matches": [m.as_dict() for m in matches],
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------- Face detection ----------
@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return {
        "faces_detected": len(faces),
        "confidence": "high" if len(faces) > 0 else "none"
    }


# ---------- MRZ validation ----------
def validate_mrz(text: str) -> dict:
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    mrz_lines = []
    for line in lines:
        if re.match(r'^[A-Z0-9<]{44}$', line) or re.match(r'^[A-Z0-9<]{36}$', line):
            mrz_lines.append(line)
    if len(mrz_lines) < 2:
        return {"valid": False, "reason": "MRZ format not recognized"}
    mrz1, mrz2 = mrz_lines[0], mrz_lines[1]
    if mrz1[0] != 'P':
        return {"valid": False, "reason": "Not a passport MRZ (should start with P)"}
    return {
        "valid": True,
        "type": mrz1[0:2],
        "issuing_country": mrz1[2:5],
        "surname": mrz1[5:].split('<')[0] if '<' in mrz1[5:] else mrz1[5:],
        "given_names": mrz2[0:39].replace('<', ' ').strip(),
        "passport_number": mrz2[0:9],
        "nationality": mrz2[10:13],
        "birth_date": mrz2[13:19],
        "sex": mrz2[20],
        "expiry_date": mrz2[21:27],
    }


# ---------- Document upload with OCR ----------
def extract_text_from_file(content: bytes, filename: str) -> str:
    ext = filename.split('.')[-1].lower()
    text = ""
    if ext == "pdf":
        with io.BytesIO(content) as pdf_stream:
            try:
                with pdfplumber.open(pdf_stream) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
            except Exception:
                pdf_stream.seek(0)
                try:
                    reader = PdfReader(pdf_stream)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                except Exception:
                    pass
    elif ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
        try:
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)
        except Exception:
            text = ""
    elif ext in ["txt", "md"]:
        try:
            text = content.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="ignore")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return text.strip()


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    context: str = Form("")
):
    MAX_SIZE = 10 * 1024 * 1024
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_SIZE:
        raise HTTPException(400, f"File too large: max {MAX_SIZE//1024//1024}MB")

    allowed_extensions = {"pdf", "png", "jpg", "jpeg", "bmp", "tiff", "txt", "md"}
    ext = file.filename.split('.')[-1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"File type .{ext} not supported.")

    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()

    try:
        document_text = extract_text_from_file(content, file.filename)
    except Exception as e:
        raise HTTPException(500, f"Text extraction failed: {str(e)}")

    if not document_text:
        raise HTTPException(400, "No text could be extracted from the document.")

    mrz_info = validate_mrz(document_text)
    enriched_context = context
    if mrz_info.get("valid"):
        enriched_context += f"\nMRZ validation: passport {mrz_info['passport_number']}, expiry {mrz_info['expiry_date']}."
    elif mrz_info.get("reason"):
        enriched_context += f"\nMRZ validation failed: {mrz_info['reason']}"

    case = Case(document=document_text, context=enriched_context)
    report = await assess(case)

    return {
        "report": report.dict(),
        "file_hash": file_hash,
        "filename": file.filename,
        "extracted_text_length": len(document_text),
        "mrz": mrz_info,
    }


# ---------- NEW: Adverse Media Search ----------
@app.get("/adverse-media")
async def get_adverse_media(name: str):
    if not newsapi:
        return {"name": name, "hits": [], "error": "NewsAPI key not configured"}
    try:
        articles = newsapi.get_everything(
            q=name,
            language='en',
            sort_by='relevancy',
            page_size=5
        )
        results = []
        for a in articles.get('articles', []):
            if a.get('title') and a.get('url'):
                results.append({
                    'title': a['title'][:200],
                    'source': a.get('source', {}).get('name', 'Unknown'),
                    'url': a['url'],
                    'published': a.get('publishedAt', '')
                })
        return {"name": name, "hits": results}
    except Exception as e:
        return {"name": name, "hits": [], "error": str(e)}


# ---------- NEW: Batch Screening (CSV) ----------
@app.post("/batch-screen")
async def batch_screen(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Invalid CSV: {str(e)}")

    if 'document' not in df.columns:
        raise HTTPException(400, "CSV must have a 'document' column")

    results = []
    for idx, row in df.iterrows():
        doc = str(row['document'])
        ctx = str(row.get('context', ''))
        try:
            case = Case(document=doc, context=ctx)
            report = await assess(case)
            results.append({
                'row': idx + 1,
                'document_preview': doc[:100] + ('...' if len(doc) > 100 else ''),
                'rating': report.rating,
                'suggestion': report.suggestion,
                'reasons': '; '.join(report.reasons[:3]),
            })
        except Exception as e:
            results.append({
                'row': idx + 1,
                'document_preview': doc[:100] + ('...' if len(doc) > 100 else ''),
                'rating': 'ERROR',
                'suggestion': str(e),
                'reasons': '',
            })

    output = io.StringIO()
    pd.DataFrame(results).to_csv(output, index=False)
    return Response(
        output.getvalue(),
        media_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename=batch_results.csv'}
    )


# ---------- NEW: PDF Report ----------
@app.post("/generate-pdf")
async def generate_pdf(payload: dict):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "OFF Guard – Screening Report")
    y -= 30

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20

    # Rating
    report = payload.get('report', {})
    rating = report.get('rating', 'N/A').upper()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Risk Rating: {rating}")
    y -= 25

    # Suggestion
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Suggestion: {report.get('suggestion', 'N/A')}")
    y -= 20

    # Reasons
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Reasons:")
    y -= 18
    c.setFont("Helvetica", 10)
    for r in report.get('reasons', []):
        c.drawString(60, y, f"• {r}")
        y -= 16
        if y < 50:
            c.showPage()
            y = height - 50

    # File hash
    file_hash = payload.get('file_hash', 'N/A')
    if file_hash and file_hash != 'N/A':
        c.drawString(50, y - 10, f"SHA-256 Hash: {file_hash}")
        y -= 20

    # MRZ info
    mrz = payload.get('mrz', {})
    if mrz.get('valid'):
        c.drawString(50, y - 10, f"MRZ: Passport {mrz.get('passport_number', 'N/A')} (valid)")
        y -= 20

    # Adverse media
    adverse = payload.get('adverse_hits', [])
    if adverse:
        c.drawString(50, y - 10, f"Adverse Media Hits: {len(adverse)}")
        y -= 16
        for a in adverse[:3]:
            c.drawString(60, y - 10, f"• {a.get('title', '')[:80]}")
            y -= 14

    c.save()
    buffer.seek(0)
    return Response(
        buffer.getvalue(),
        media_type='application/pdf',
        headers={'Content-Disposition': 'attachment; filename=off_guard_report.pdf'}
    )
