import pytesseract
from PIL import Image

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
