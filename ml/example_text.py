from ml.text_model import DOWNLOAD_BERT, TextModel

DOWNLOAD_BERT()
tm = TextModel()

s1 = 'Сноуборд Burton'
s2 = "Сноуборд F2"

print(tm.similarity(tm.to_vector(s1), s2))