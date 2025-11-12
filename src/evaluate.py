import jiwer
from itertools import product
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import logging
from pathlib import Path
from glob import glob
import re
from src.ocr.service import OCRService
from src.ocr.tesseract_ocr import TesseractOCREngine
from src.parser.service import ParserService

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("evaluation.log"),
#         logging.StreamHandler()
#     ]
# )

transform_word = jiwer.Compose([
    jiwer.ToLowerCase(),               # Convert to lowercase
    jiwer.RemoveMultipleSpaces(),      # Remove extra spaces
    jiwer.Strip(),                     # Remove leading/trailing whitespace
    jiwer.ReduceToListOfListOfWords()  # Prepare for WER calculation
])

transform_character = jiwer.Compose([
    jiwer.ToLowerCase(),               # Convert to lowercase
    jiwer.RemoveMultipleSpaces(),      # Remove extra spaces
    jiwer.Strip(),                     # Remove leading/trailing whitespace
    jiwer.RemoveWhiteSpace(replace_by_space=False),
    jiwer.ReduceToListOfListOfChars() # Prepare for WER calculation
])



class DocumentPair:
    def __init__(self, ground_truth: str, prediction: str):
        self.ground_truth = ground_truth.strip()
        self.prediction = prediction.strip()

    def compute_wer(self) -> float:
        """
        Calculates WER between docs
        """
        return jiwer.wer(self.ground_truth, self.prediction, reference_transform=transform_word, hypothesis_transform=transform_word)

    def compute_cer(self) -> float:
        """
        Calculates CER between doc
        """
        return jiwer.cer(self.ground_truth, self.prediction, reference_transform=transform_character, hypothesis_transform=transform_character)


class DocumentPairsEvaluator:
    def __init__(self, params: list, document_pair: tuple):
        self.params = params
        self.engine = TesseractOCREngine(lang="eng+kaz+rus")
        self.service = OCRService(self.engine)
        self.document_pair = document_pair


    def pipeline(self, ground_truth_path: str, image_path: str, params: list) -> Tuple[str, str]:
        """
        Generates and returns differend docs depending on params values
        """
        with open(ground_truth_path, 'r', encoding='utf-8') as file:
                ground_truth_text= file.read().strip()

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        img_bytes_new = preprocessor(img, params)
        
        predicted_text = self.service.recognize(img_bytes, params)

        return ground_truth_text, predicted_text


    def generate_grid(self) -> NDArray[np.float64]:
        """
        Generates all the possible combinations between different parameters
        """
        parameters = []
        for key, value in self.params.items():
            mn, mx, step = value
            n_steps = int(round((mx - mn) / step)) + 1
            arr = np.linspace(mn, mx, n_steps)
            parameters.append(arr)
        permutations = np.array(list(product(*parameters)))

        return permutations


    def evaluate_pair(self, permutations: NDArray[np.float64]) -> Tuple[list, list]:
        """
        Calculates WER and CER values between docs and
        Returns list of WER and CER for all the different parameter combinations
        """
        wers = []
        cers = []
        for i in range(len(permutations)):
            ground_truth_text, predicted_text = self.pipeline(self.document_pair[0], self.document_pair[1], permutations[i])
            doc = DocumentPair(ground_truth_text, predicted_text)
            wer = doc.compute_wer()
            cer = doc.compute_cer()
            wers.append(round(wer, 2))
            cers.append(round(cer, 2))
        # logging.info(f"WERs: {wers}, CERs: {cers}")
        
        return wers, cers


    def find_best_parameters(self) -> list:
        """
        Return the combination of paramters with the lowest CER and in the case of a tie
        With the lowest WER
        """
        permutations = self.generate_grid()

        wers, cers = self.evaluate_pair(permutations)
        
        best_idxs = np.lexsort((wers, cers))

        # print(permutations)

        return permutations[best_idxs[0]]



# parser = ParserService()
# dataset_path = 'ocr_train_dataset'
# document_pairs = parser.parse(dataset_path)



# params = {'parameter_1': (0, 3, 1),
#           'parameter_2': (0, 13, 1),
#           }

# best_parameters = []
# i = 0
# for document_pair in document_pairs:
#     evaluator = DocumentPairsEvaluator(params, document_pair)

#     best_parameters.append(evaluator.find_best_parameters())
#     if i == 5:
#         break
#     i += 1
# print(best_parameters)



    