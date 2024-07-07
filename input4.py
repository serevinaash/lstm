import os
import numpy as np
import librosa
import tensorflow as tf
import pickle

# Tentukan panjang urutan maksimum berdasarkan data pelatihan
max_seq_length = 500  # Ubah nilai ini sesuai dengan panjang maksimum urutan yang digunakan selama pelatihan

# Fungsi untuk memproses input audio
def preprocess_audio_input(file_path, max_seq_length):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    if mfccs.shape[1] < max_seq_length:
        padded_mfccs = np.pad(mfccs.T, ((0, max_seq_length - mfccs.shape[1]), (0, 0)), mode='constant')
    else:
        padded_mfccs = mfccs.T[:max_seq_length]
    return padded_mfccs

# Path ke file audio input Anda
audio_input_path = r'C:\lstm\audio\001003.wav'

# Muat model yang telah dilatih
best_model = tf.keras.models.load_model('best_model.keras')

# Muat label encoder
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Proses input audio
processed_audio_input = preprocess_audio_input(audio_input_path, max_seq_length)
processed_audio_input = np.expand_dims(processed_audio_input, axis=0)  # Tambahkan dimensi batch

# Lakukan prediksi
predictions = best_model.predict(processed_audio_input)
predicted_label_index = np.argmax(predictions[0])  # Dapatkan indeks dengan probabilitas tertinggi
predicted_transcript = label_encoder.inverse_transform([predicted_label_index])[0]  # Ubah kembali ke teks

print("Prediksi teks:", predicted_transcript)
