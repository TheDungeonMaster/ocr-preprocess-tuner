from typing import Tuple, List

from src.ocr.service import OCRService
from src.ocr.tesseract_ocr import TesseractOCREngine

from src.preprocess.service import PreprocessService
from src.parser.service import ParserService

from src.preprocess.pillow_preprocesser import PillowImagePreprocessor
from src.parser.service import ParserService
from numpy.typing import NDArray
from itertools import product
import numpy as np
from src.evaluate import DocumentPair





class FitImageUseCase:
    def __init__(self):
        pass

    def pipeline(self, ground_truth_path: str, image_path: str, params: list) -> Tuple[str, str]:
        """
        Generates and returns differend docs depending on params values
        """
        with open(ground_truth_path, 'r', encoding='utf-8') as file:
                ground_truth_text= file.read().strip()

        pillow_impl = PillowImagePreprocessor(
            contrast=params[0],
            denoise_strength = int(params[1])
        )

        preprocessor_service = PreprocessService(pillow_impl)

        with open(image_path, "rb") as f:
                raw_bytes = f.read()

                processed_img_bytes = preprocessor_service.run(raw_bytes)

        engine = TesseractOCREngine(lang="eng+kaz+rus")
        service = OCRService(engine)

        predicted_text = service.recognize(processed_img_bytes)

        return ground_truth_text, predicted_text


    def generate_grid(self, params) -> NDArray[np.float64]:
        """
        Generates all the possible combinations between different parameters
        """
        parameters = []
        for key, value in params.items():
            mn, mx, step = value
            n_steps = int(round((mx - mn) / step)) + 1
            arr = np.linspace(mn, mx, n_steps)
            parameters.append(arr)
        permutations = np.array(list(product(*parameters)))

        return permutations
    
    def evaluate_pair(self, permutations: NDArray[np.float64], document_pair: list) -> Tuple[list, list]:
        """
        Calculates WER and CER values between docs and
        Returns list of WER and CER for all the different parameter combinations
        """
        wers = []
        cers = []
        for i in range(len(permutations)):

            ground_truth_text, predicted_text = self.pipeline(document_pair[0], document_pair[1], permutations[i])


            doc = DocumentPair(ground_truth_text, predicted_text)
            wer = doc.compute_wer()
            cer = doc.compute_cer()
            wers.append(round(wer, 2))
            cers.append(round(cer, 2))
        # logging.info(f"WERs: {wers}, CERs: {cers}")
        
        return wers, cers


    def fit(self, dataset_path: str, params: dict) -> List[Tuple[str, List[float]]]:
        parser = ParserService()
        document_pairs = parser.parse(dataset_path)

        permutations = self.generate_grid(params)

        image_and_best_parameters = []
        i = 0

        for document_pair in document_pairs:
             if i == 1:
                  break
             wers, cers = self.evaluate_pair(permutations, document_pair)
             
             best_idxs = np.lexsort((wers, cers))

             image_and_best_parameters.append((document_pair[1], permutations[best_idxs[0]]))
             i += 1

        return image_and_best_parameters
        


params = {'contrast': (1.2, 1.3, 0.1),
          'denoise_strength': (5, 7, 2)
          }


trying = FitImageUseCase()

dataset_path = 'ocr_train_dataset'

a = trying.fit(dataset_path, params)

print(a)

