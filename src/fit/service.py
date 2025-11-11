from typing import Tuple, List

from src.ocr.service import OCRService
from src.preprocess.service import PreprocessService
from src.parser.service import ParserService

class FitImageUseCase:

    def __init__(self, ocr_service: OCRService, preprocessor: PreprocessService,):
        self._ocr_service = ocr_service
        self._preprocessor = preprocessor


    def fit(self, dataset_path: str, params: dict) -> List[str, List[Tuple[str, float]]]:
        pass

