from ml.text_model import DOWNLOAD_BERT, TextModel

DOWNLOAD_BERT()
tm = TextModel()

s1 = 'У меня есть машина'
s2 = "I have a car"

print(tm.similarity(tm.to_vector(s1), s2))