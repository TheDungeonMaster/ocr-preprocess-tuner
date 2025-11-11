from typing import Tuple, List
from pathlib import Path
from glob import glob


class ParserService:
    def __init__(self):
        pass

    def parse(self, dataset_path:str) -> List[Tuple[str,str]]:
        document_pairs = []
        folder = Path(dataset_path)
        document_pairs = []

        for dir in folder.iterdir():
            texts = glob(f'{dir}/gt/main*[0-9][0-9][0-9].png.gpt.txt')
            texts.sort(key= lambda x: int(re.findall(r'\d+', x)[-1]))

            images = glob(f'{dir}/pngs/main*[0-9][0-9][0-9].png')
            images.sort(key= lambda x: int(re.findall(r'\d+', x)[-1]))

            document_pairs.extend(list(zip(texts, images)))
        return document_pairs


