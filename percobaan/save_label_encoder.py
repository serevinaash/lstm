import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

# Membaca file JSON
with open('C:\\lstm\\all_ayat.json', 'r', encoding='utf-8') as f:
    ayat_json = json.load(f)

# Ekstrak label (teks ayat) dari struktur JSON
labels = [ayat['text'] for ayat in ayat_json['tafsir'].values()]

# Encode labels as integers
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

# One-hot encode the integer labels
onehot_encoded = to_categorical(integer_encoded)

# Simpan LabelEncoder untuk digunakan nanti
with open('C:\\lstm\\label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

print("Label encoding selesai dan disimpan.")
