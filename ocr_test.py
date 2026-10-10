import sys
from PIL import Image
import pytesseract

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Open your image
img = Image.open(r"c:\major project\ocr_samp1.jpeg")

# Run OCR
text = pytesseract.image_to_string(img)

# Show result in terminal
print(text)

# Save result to file
with open(r"c:\major project\output.txt", "w", encoding="utf-8") as f:
    f.write(text)
