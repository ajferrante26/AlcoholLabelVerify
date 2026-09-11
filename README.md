# AlcoholLabelVerify - An AI-Powered Alcohol Label Verification App

This application is a prototype for **an interview requirement** to automate label compliance checking. It uses OCR (optical character recognition) to extract text from alcohol beverage label images and compares the extracted fields against expected data provided by the user. The tool is designed to be fast (under 5 seconds per label), simple to use, and capable of handling batch uploads.

## Features

- **Single Label Verification** – Upload a label image and expected data (JSON) to get a detailed pass/fail/review report.
- **Batch Verification** – Upload multiple label images and a CSV file with expected data for each image to process them in one request.
- **OCR Text Extraction** – Provides an endpoint to get raw OCR text from an image for debugging or troubleshooting.
- **Web Interface** – A clean, minimal frontend served directly by the backend, making it easy to test from any browser.
- **Fast & Local** – Uses Tesseract OCR locally, no cloud APIs required (works even in restricted networks).
- **Containerized** – Docker image available for easy deployment and consistent environment.

## Prerequisites

- **Docker** (if using the container method)

or 

- **Python 3.10+** and **virtualenv** (if running from source)
- **Tesseract OCR** (if running from source)

## Installation & Running

### Option 1: Use Docker (recommended)

A pre-built Docker image is available publicly at `ajferrante26/alcohol-label-verify`. Pull and run it:

```bash
docker pull ajferrante26/alcohol-label-verify:latest
docker run -p 3456:3456 ajferrante26/alcohol-label-verify:latest
```

Open your browser and go to `http://localhost:3456`. The application will be ready to use.

If you prefer to build the image yourself from source, you can clone the repository and build:

```bash
git clone https://github.com/yourusername/alcohol-label-verify.git
cd alcohol-label-verify
docker build -t alcohol-label-verify .
docker run -p 3456:3456 alcohol-label-verify
```

### Option 2: Run from Source

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/alcohol-label-verify.git
cd alcohol-label-verify
```

2. **Create and activate a virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**:

```bash
pip install -r requirements.txt
```

4. **Install Tesseract OCR** (system dependency):

- **Ubuntu/Debian**: `sudo apt-get install -y tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Download and install from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), and add it to your PATH.

5. **Run the application**:

```bash
python main.py
```

The server will start on port `3456`. Open `http://localhost:3456` in your browser.

## Usage

### Web Interface

Once the app is running, you will see two sections:

- **Single Label Verification** – Upload a label image and paste the expected data as JSON. Click "Verify Single Label" to see results.
- **Batch Verification** – Upload multiple label images and a CSV file containing expected data for each. Click "Verify Batch".

The results will show an overall status (`PASS`, `REVIEW`, `FAIL`) and per-field statuses for brand name, ABV, net contents, and government warning.

### API Endpoints

The backend exposes the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend HTML. |
| `POST` | `/extract-text` | Accepts an image file and returns raw OCR text. |
| `POST` | `/verify` | Accepts an image file and expected data (JSON string) and returns verification report. |
| `POST` | `/verify-batch` | Accepts multiple image files and a CSV file, returns results for each. |

### Example CSV for Batch Verification

The CSV file must have a column named `image_filename` that matches the uploaded image filenames (including extension). Other required columns are `brand_name`, `abv`, `net_contents`, and `government_warning`.

```csv
image_filename,brand_name,abv,net_contents,government_warning
label1.jpg,ABC WINERY,13.0,750 mL,"GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS."
label2.jpg,OLD TOM DISTILLERY,45.0,750 mL,"GOVERNMENT WARNING: ..."
```

## Configuration

- **Port**: The server runs on port `3456` by default. To change it, modify the `uvicorn.run` line in `main.py` or set the `--port` flag when running with `uvicorn` directly.
- **Tesseract Path**: If Tesseract is not in your PATH (source installation), you can set its location in `main.py`:

  ```python
  pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust path as needed
  ```

## How It Works

1. **Image Preprocessing** – OpenCV is used to convert the image to grayscale, apply CLAHE contrast enhancement, and adaptive thresholding to improve OCR accuracy.
2. **OCR** – Tesseract (with the LSTM engine) extracts raw text from the preprocessed image.
3. **Field Parsing** – Regular expressions are used to locate brand name, alcohol by volume (ABV), net contents, and the government warning statement.
4. **Comparison** – Extracted values are compared against the expected data using normalization (case-insensitive, punctuation removal) and fuzzy tolerance (e.g., ±0.5% for ABV). Each field is assigned a status: `MATCH`, `MISMATCH`, `NOT_FOUND`, or `LOW_CONFIDENCE`.
5. **Report** – The overall status is `PASS` if all fields match, `REVIEW` if any field is missing or low-confidence, and `FAIL` if any field clearly mismatches.

## Limitations & Assumptions

- The government warning must start with `"GOVERNMENT WARNING:"` in all caps (as per TTB regulations). Case differences are flagged as `LOW_CONFIDENCE`.
- Brand names are matched after normalizing case and removing punctuation, which handles common variations like `STONE'S THROW` vs. `Stone's Throw`.
- Net contents are parsed with tolerance for OCR errors (e.g., `750M` is interpreted as `750 mL`).
- The system is optimized for typical label layouts; extremely stylized fonts or poor image quality may reduce accuracy. Consider image preprocessing adjustments for production use.

## Future Enhancements

- Integration with the TTB COLA system.
- Enhanced image preprocessing for skewed angles, glare, and low-light (already partially handled).
- Confidence scoring for each field to prioritize human review.
- Ability to accept additional label types (beer, wine, spirits) with specific parsing rules.

## License

This project is provided for evaluation purposes as part of a take-home assignment. All third-party libraries are used under their respective licenses.

---

*Built with FastAPI, OpenCV, Tesseract, and Docker on an Nvidia Jetson Orin Nano*
