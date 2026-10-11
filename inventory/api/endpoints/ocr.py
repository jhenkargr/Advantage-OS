"""Receipt OCR and Retail Ingestion API Endpoints."""

from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from inventory.config import WORKSPACE_ROOT, INVENTORY_DIR
from inventory.models.schemas import (
    ScannedReceiptResponse,
    ApplyReceiptRequest,
    ApplyReceiptResponse,
)
from inventory.services.ocr_service import OCRService

router = APIRouter(prefix="/ocr", tags=["Retail Operations & OCR Ingestion"])


@router.post("/scan-receipt", response_model=ScannedReceiptResponse, summary="Upload and scan receipt image via OCR")
async def scan_receipt(file: UploadFile = File(...)):
    """
    Upload a receipt image (JPG, PNG, WebP) to:
    - Run Tesseract OCR to extract raw text
    - Parse store name, date, invoice number, and line items
    - Fuzzy match item names to catalog product IDs with confidence scores
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files (JPEG, PNG, etc.) are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return OCRService.process_receipt_image(content, filename=file.filename or "uploaded_receipt.jpg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")


@router.post("/scan-sample", response_model=ScannedReceiptResponse, summary="Scan pre-bundled sample receipt")
def scan_sample_receipt():
    """Run OCR on the project's included sample receipt (ocr_samp1.jpeg) for instant verification."""
    sample_path = WORKSPACE_ROOT / "ocr_samp1.jpeg"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample receipt image not found on server.")

    with open(sample_path, "rb") as f:
        content = f.read()

    try:
        return OCRService.process_receipt_image(content, filename="ocr_samp1.jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")


@router.post("/apply-receipt", response_model=ApplyReceiptResponse, summary="Apply parsed receipt to update inventory")
def apply_receipt(request: ApplyReceiptRequest):
    """
    Confirm and apply scanned line items to update current stock levels in the product catalog.
    Supports both STOCK_IN (replenishment receipt) and STOCK_OUT (sales slip).
    """
    try:
        return OCRService.apply_receipt_items(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply receipt: {e}")
