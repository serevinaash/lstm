import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Path ke file transkrip Anda
transcripts_path = 'C:\\lstm\\filter_transcripts.tsv'

# Membaca data transkrip
transcripts = pd.read_csv(transcripts_path, sep='\t')

# Membuat LabelEncoder dan fit dengan data transkrip
label_encoder = LabelEncoder()
labels = transcripts['TRANSCRIPT']
label_encoder.fit(labels)

# Menyimpan LabelEncoder ke dalam file
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

print("Label encoder berhasil dibuat dan disimpan.")
