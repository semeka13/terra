from transformers import BertTokenizerFast, TFBertModel
from ml.utils import cos_similatiry
import numpy as np
from os import listdir, mkdir

"""
Как это использовать: 
    1) Нужно вызвать функцию DOWNLOAD_BERT она делает папки и скачивает модели тк запушить это на гит ненвозможно
    2) Класс TextModel делает 2 вещи: преобразование строки ввектор и определение сходства строк
"""


def DOWNLOAD_BERT():
    files = listdir()
    if "tokenizer" not in files:
        mkdir("tokenizer")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        tokenizer.save_pretrained("tokenizer")

    if "vectorizer" not in files:
        mkdir("vectorizer")
        vectorizer = TFBertModel.from_pretrained("bert-base-multilingual-cased")
        vectorizer.save_pretrained("vectorizer")


class TextModel():
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("tokenizer")
        self.vectorizer = TFBertModel.from_pretrained("vectorizer")

    def similarity(self, vector: np.ndarray, line: str) -> float:
        """
        :param vector: a 1d numpy array i.e. word vector
        :param line: string
        :return: cosine similarity i.e. angle between two word vectors
        """
        vectorized = self.to_vector(line)
        return cos_similatiry(vector, vectorized)

    def to_vector(self, line: str) -> np.ndarray:
        """
        :param line: string
        :return: a 1d numpy array i.e. word vector
        """
        line = [line]
        token = self.tokenizer(line)["input_ids"]
        vector = self.vectorizer(np.array(token)).pooler_output[0]
        return vector.numpy()
