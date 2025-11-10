import jiwer
from itertools import product
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

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


ground_truth_path_1 = "docs/doc1/Contract_AEOMPI25.txt"
predicted_path_1 = "docs/doc1/AEOMPI25_signed_by_2_parts[1]_p001.txt"

ground_truth_path_2 = "docs/doc2/Contract_AEOMPI25_part2.txt"
predicted_path_2 = "docs/doc2/AEOMPI25_signed_by_2_parts[1]_p002.txt"


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


class DocumentEvaluator:
    def __init__(self, params):
        self.params = params


    def read_text_file(self, file_path: str) -> str:
        """
        Returns contents of files
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
        except Exception as e:
            logging.error(f"Could not open the file with path: {file_path} with error {e}", exc_info=True)
            return ''
        return text


    def pipeline(self, ground_truth_path: str, predicted_path: str, params: dict = None) -> Tuple[str, str]:
        """
        Generates and returns differend docs depending on params values
        """
        ground_truth_text = self.read_text_file(ground_truth_path)
        predicted_text = self.read_text_file(predicted_path)

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
            ground_truth_text, predicted_text = self.pipeline(ground_truth_path_1, predicted_path_1, permutations[i])
            doc = DocumentPair(ground_truth_text, predicted_text)
            wer = doc.compute_wer()
            cer = doc.compute_cer()
            wers.append(round(wer, 2))
            cers.append(round(cer, 2))
        logging.info(f"WERs: {wers}, CERs: {cers}")
        
        return wers, cers


    def find_best_parameters(self, wers: list, cers: list, permutations: list) -> list:
        """
        Return the combination of paramters with the lowest CER and in the case of a tie
        With the lowest WER
        """
        best_idxs = np.lexsort((wers, cers))

        return permutations[best_idxs[0]]


params = {'parameter_1': (0, 1, 0.2),
          'parameter_2': (2, 3, 0.2),
          }

evaluator = DocumentEvaluator(params)

permutations = evaluator.generate_grid()

wers, cers = evaluator.evaluate_pair(permutations)

best_parameters = evaluator.find_best_parameters(wers, cers, permutations)

print(best_parameters)

