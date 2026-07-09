import hashlib
import io
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import pdfplumber
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

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


# ---------- NEW: Face detection ----------
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


# ---------- NEW: MRZ validation helper ----------
def validate_mrz(text: str) -> dict:
    """
    Basic MRZ (Machine Readable Zone) validation for passports.
    Returns dict with 'valid' and details if found.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    mrz_lines = []
    for line in lines:
        # Passport MRZ: two lines, each 44 characters (or 36 for some IDs)
        if re.match(r'^[A-Z0-9<]{44}$', line) or re.match(r'^[A-Z0-9<]{36}$', line):
            mrz_lines.append(line)
    if len(mrz_lines) < 2:
        return {"valid": False, "reason": "MRZ format not recognized"}
    mrz1, mrz2 = mrz_lines[0], mrz_lines[1]
    if mrz1[0] != 'P':
        return {"valid": False, "reason": "Not a passport MRZ (should start with P)"}
    # Basic extraction
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


# ---------- NEW: Document upload with OCR and MRZ ----------
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
    # Security: file size
    MAX_SIZE = 10 * 1024 * 1024
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_SIZE:
        raise HTTPException(400, f"File too large: max {MAX_SIZE//1024//1024}MB")

    # Allowed extensions
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

    # MRZ validation
    mrz_info = validate_mrz(document_text)
    enriched_context = context
    if mrz_info.get("valid"):
        enriched_context += f"\nMRZ validation: passport {mrz_info['passport_number']}, expiry {mrz_info['expiry_date']}."
    elif mrz_info.get("reason"):
        enriched_context += f"\nMRZ validation failed: {mrz_info['reason']}"

    # Run the agent
    case = Case(document=document_text, context=enriched_context)
    report = await assess(case)

    return {
        "report": report.dict(),
        "file_hash": file_hash,
        "filename": file.filename,
        "extracted_text_length": len(document_text),
        "mrz": mrz_info,
    }
