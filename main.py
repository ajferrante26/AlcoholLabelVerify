import io
import re
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alcohol Label Verification API", version="1.0.0")

# Allow CORS for local testing (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Image Preprocessing & OCR
# ---------------------------

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to a preprocessed OpenCV image for better OCR.
    Steps: decode, grayscale, contrast enhancement, thresholding.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return gray

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Run Tesseract OCR on the preprocessed image and return raw text.
    """
    try:
        processed = preprocess_image(image_bytes)
        text = pytesseract.image_to_string(processed, config='--psm 6')
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

# ---------------------------
# Field Parsing from OCR Text
# ---------------------------

def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, collapse spaces."""
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_brand_name(text: str, expected_brand: str) -> Optional[str]:
    """
    Try to find the brand name in the OCR text.
    Uses normalized comparison to handle case, punctuation, and spacing differences.
    """
    norm_text = normalize_text(text)
    norm_expected = normalize_text(expected_brand)
    if norm_expected in norm_text:
        # Return the expected brand as a placeholder (or you could extract the actual substring)
        return expected_brand
    return None

def parse_abv(text: str) -> Optional[float]:
    """
    Extract alcohol by volume percentage from text.
    Looks for patterns like "ALC. 13% BY VOL.", "13% ALC./VOL.", "ALCOHOL 13% BY VOLUME".
    """
    # Pattern 1: keyword before number (e.g., "ALC. 13%", "ALCOHOL 13%", "BY VOL. 13%")
    pattern1 = r'(?:ALC\.?|ALCOHOL|BY VOL\.?|VOL\.?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%'
    # Pattern 2: number before keyword (e.g., "13% ALC.", "13% BY VOL.")
    pattern2 = r'(\d+(?:\.\d+)?)\s*%\s*(?:ALC\.?|BY VOL\.?|VOL\.?)'

    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None

def parse_net_contents(text: str) -> Optional[str]:
    """
    Extract net contents (e.g., "750 mL", "1.75 L", "12 fl oz").
    Handles common OCR errors like missing 'L' (e.g., "750M").
    """
    # Pattern: number (with optional decimal) followed by a unit
    # Unit can be: ml, mL, ML, Ml, l, L, oz, fl oz, floz, or M (misread of mL)
    pattern = r'(\d+(?:\.\d+)?)\s*(ml|mL|ML|Ml|l|L|oz|fl\.?\s*oz|floz|M)\b'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    for match in matches:
        value = match.group(1)
        unit = match.group(2).lower().replace('.', '').replace(' ', '')
        # Map to standard units
        if unit in ['ml', 'milliliter', 'm']:
            unit = 'ml'
        elif unit in ['l', 'liter']:
            unit = 'l'
        elif unit in ['oz', 'floz', 'floz']:
            unit = 'oz'
        else:
            continue  # skip if not a recognized volume unit
        return f"{value} {unit}"
    return None

def parse_government_warning(text: str) -> Optional[str]:
    """
    Extract the government warning statement.
    The warning must start with "GOVERNMENT WARNING:" in all caps.
    We capture the text from that point until a blank line or end.
    """
    start = text.find("GOVERNMENT WARNING:")
    if start == -1:
        return None
    end = text.find("\n", start)
    if end == -1:
        end = len(text)
    warning = text[start:end].strip()
    return warning

# ---------------------------
# Comparison Logic
# ---------------------------

def compare_brand(extracted: Optional[str], expected: str) -> Dict[str, str]:
    if not extracted:
        return {"status": "NOT_FOUND", "message": "Brand name not found in OCR text"}
    norm_extracted = normalize_text(extracted)
    norm_expected = normalize_text(expected)
    if norm_extracted == norm_expected:
        return {"status": "MATCH", "message": "Brand name matches"}
    if norm_expected in norm_extracted or norm_extracted in norm_expected:
        return {"status": "LOW_CONFIDENCE", "message": "Brand name partially matches"}
    return {"status": "MISMATCH", "message": f"Brand name mismatch: extracted '{extracted}' vs expected '{expected}'"}

def compare_abv(extracted: Optional[float], expected: float) -> Dict[str, str]:
    if extracted is None:
        return {"status": "NOT_FOUND", "message": "ABV not found in OCR text"}
    if abs(extracted - expected) <= 0.5:
        return {"status": "MATCH", "message": f"ABV matches ({extracted}%)"}
    else:
        return {"status": "MISMATCH", "message": f"ABV mismatch: extracted {extracted}% vs expected {expected}%"}

def compare_net_contents(extracted: Optional[str], expected: str) -> Dict[str, str]:
    if not extracted:
        return {"status": "NOT_FOUND", "message": "Net contents not found in OCR text"}
    norm_extracted = extracted.lower().replace(' ', '')
    norm_expected = expected.lower().replace(' ', '')
    if norm_extracted == norm_expected:
        return {"status": "MATCH", "message": "Net contents match"}
    else:
        return {"status": "MISMATCH", "message": f"Net contents mismatch: extracted '{extracted}' vs expected '{expected}'"}

def compare_warning(extracted: Optional[str], expected: str) -> Dict[str, str]:
    if not extracted:
        return {"status": "NOT_FOUND", "message": "Government warning not found in OCR text"}
    if not extracted.startswith("GOVERNMENT WARNING:"):
        return {"status": "LOW_CONFIDENCE", "message": "Warning does not start with 'GOVERNMENT WARNING:' in all caps"}
    extracted_body = extracted[len("GOVERNMENT WARNING:"):].strip()
    expected_body = expected[len("GOVERNMENT WARNING:"):].strip() if expected.startswith("GOVERNMENT WARNING:") else expected
    extracted_body = re.sub(r'\s+', ' ', extracted_body).strip()
    expected_body = re.sub(r'\s+', ' ', expected_body).strip()
    if extracted_body == expected_body:
        return {"status": "MATCH", "message": "Government warning matches"}
    elif extracted_body.lower() == expected_body.lower():
        return {"status": "LOW_CONFIDENCE", "message": "Government warning matches but case differs"}
    else:
        return {"status": "MISMATCH", "message": "Government warning text differs"}

# ---------------------------
# Main Verification Function
# ---------------------------

def verify_label(image_bytes: bytes, expected_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single label image and compare against expected data.
    Returns a report with per-field results.
    """
    text = extract_text_from_image(image_bytes)
    logger.info(f"OCR text: {text[:200]}...")
    logger.info(f"Expected brand: '{expected_data.get('brand_name', '')}'")

    brand_extracted = parse_brand_name(text, expected_data.get("brand_name", ""))
    abv_extracted = parse_abv(text)
    net_extracted = parse_net_contents(text)
    warning_extracted = parse_government_warning(text)

    results = {
        "brand_name": compare_brand(brand_extracted, expected_data.get("brand_name", "")),
        "abv": compare_abv(abv_extracted, expected_data.get("abv", 0.0)),
        "net_contents": compare_net_contents(net_extracted, expected_data.get("net_contents", "")),
        "government_warning": compare_warning(warning_extracted, expected_data.get("government_warning", "")),
    }

    statuses = [r["status"] for r in results.values()]
    if "MISMATCH" in statuses:
        overall = "FAIL"
    elif "NOT_FOUND" in statuses or "LOW_CONFIDENCE" in statuses:
        overall = "REVIEW"
    else:
        overall = "PASS"

    return {
        "overall_status": overall,
        "fields": results,
        "raw_ocr_text": text,
    }

# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML file."""
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """Test endpoint: extract raw OCR text from an image."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        text = extract_text_from_image(image_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify")
async def verify_single(
    file: UploadFile = File(...),
    expected_data: str = Form(...)
):
    """
    Verify a single label image against expected data.
    - file: image file (jpg, png, etc.)
    - expected_data: JSON string with fields: brand_name, abv, net_contents, government_warning
    """
    try:
        expected = json.loads(expected_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in expected_data")

    required = ["brand_name", "abv", "net_contents", "government_warning"]
    for field in required:
        if field not in expected:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = verify_label(image_bytes, expected)
        return result
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-batch")
async def verify_batch(
    files: List[UploadFile] = File(...),
    expected_data_file: UploadFile = File(...)
):
    """
    Verify multiple label images against a CSV file.
    - files: list of image files (names must match the 'image_filename' column in CSV)
    - expected_data_file: CSV file with columns: image_filename, brand_name, abv, net_contents, government_warning
    """
    try:
        df = pd.read_csv(expected_data_file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    # Build a mapping from filename to expected data
    expected_map = {}
    for _, row in df.iterrows():
        filename = row.get("image_filename")
        if not filename:
            continue
        expected_map[filename] = {
            "brand_name": row.get("brand_name", ""),
            "abv": float(row.get("abv", 0.0)),
            "net_contents": row.get("net_contents", ""),
            "government_warning": row.get("government_warning", ""),
        }

    # Process each uploaded file
    results = []
    for file in files:
        filename = file.filename
        if filename not in expected_map:
            results.append({
                "filename": filename,
                "error": "No expected data found for this file in CSV"
            })
            continue

        image_bytes = await file.read()
        try:
            result = verify_label(image_bytes, expected_map[filename])
            result["filename"] = filename
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results.append({
                "filename": filename,
                "error": str(e)
            })

    return {"results": results}

# ---------------------------
# main
# ---------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3456)
