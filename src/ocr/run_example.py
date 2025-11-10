from src.ocr.service import OCRService
from src.ocr.tesseract_ocr import TesseractOCREngine

def main():
    engine = TesseractOCREngine(lang="eng+kaz+rus")
    service = OCRService(engine)

    with open("/Users/abdiakhmet/Documents/FORTE TZ/ocr-preprocess-tuner/src/ocr/imgs/img.png", "rb") as f:
        img_bytes = f.read()

    text = service.recognize(img_bytes)
    print(text)

if __name__ == "__main__":
    main()