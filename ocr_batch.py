import sys
from pathlib import Path
from PIL import Image
import pytesseract

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Input and output folders
input_folder = Path(r"c:\major project\ocr_images")
output_folder = Path(r"c:\major project\ocr_results")
output_folder.mkdir(parents=True, exist_ok=True)

if not input_folder.is_dir():
    raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
image_paths = sorted(
    path
    for path in input_folder.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_paths:
    raise FileNotFoundError(f"No supported images found in: {input_folder}")

# Combined results file
combined_file = output_folder / "all_results.txt"
errors = []

with combined_file.open("w", encoding="utf-8") as combined:
    for number, img_path in enumerate(image_paths, start=1):
        try:
            with Image.open(img_path) as img:
                text = pytesseract.image_to_string(img)

            # Prefix the number to keep duplicate filenames from overwriting results.
            result_file = output_folder / f"{number:04d}_{img_path.stem}.txt"
            result_file.write_text(text, encoding="utf-8")

            combined.write(f"=== {img_path.relative_to(input_folder)} ===\n")
            combined.write(text + "\n\n")

            print(f"[{number}/{len(image_paths)}] Processed: {img_path.name}")
        except Exception as error:
            errors.append(f"{img_path.relative_to(input_folder)}: {error}")
            print(f"[{number}/{len(image_paths)}] Error with {img_path.name}: {error}")

if errors:
    (output_folder / "errors.txt").write_text("\n".join(errors), encoding="utf-8")

print(f"Completed: {len(image_paths) - len(errors)}/{len(image_paths)} images")
print(f"Combined output: {combined_file}")
