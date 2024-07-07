import os
import numpy as np
import tensorflow as tf
import sounddevice as sd
import queue
import wave
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# Create a queue to store audio data
q = queue.Queue()

# Audio callback to store audio data in the queue
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

# Function to record audio
def record_audio(duration=5, samplerate=16000):
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
        sd.sleep(int(duration * 1000))
    audio_data = []
    while not q.empty():
        audio_data.extend(q.get())
    return np.array(audio_data)

# Function to save recorded audio to a WAV file
def save_wav(file_path, audio_data, samplerate=16000):
    with wave.open(file_path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        f.writeframes(audio_data.astype(np.int16).tobytes())

# Feature extraction function with augmentation
def extract_features(file_path, augment=False):
    audio, sr = librosa.load(file_path, sr=None)
    if augment:
        if np.random.rand() < 0.5:
            audio = audio + 0.005 * np.random.randn(len(audio))  # Add white noise
        if np.random.rand() < 0.5:
            audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))  # Time stretching
        if np.random.rand() < 0.5:
            audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=np.random.randint(-3, 3))  # Pitch shifting
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs.T

def process_audio(model, audio_data_path, label_encoder, max_seq_length):
    # Extract MFCC features from the audio data
    mfccs = extract_features(audio_data_path)
    mfccs = pad_sequences([mfccs], maxlen=max_seq_length, padding='post', dtype='float32')
    mfccs = (mfccs - np.mean(mfccs, axis=0)) \\ np.std(mfccs, axis=0)

    # Predict the text
    prediction = model.predict(mfccs)
    predicted_label = np.argmax(prediction, axis=1)
    predicted_text = label_encoder.inverse_transform(predicted_label)

    return predicted_text[0]

def main():
    # Load the trained model
    best_model = tf.keras.models.load_model('best_model.keras')

    # Load label encoder
    with open('label_encoder.pkl', 'rb') as le_file:
        label_encoder_classes = pickle.load(le_file)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_encoder_classes

    # Maximum sequence length
    max_seq_length = 500  # Sesuaikan dengan dataset Anda

    # Record audio from the microphone
    print("Recording...")
    audio_data = record_audio(duration=5)
    print("Recording finished.")

    # Simpan audio yang direkam ke file untuk diproses
    audio_file_path = 'recorded_audio.wav'
    save_wav(audio_file_path, audio_data)

    # Proses audio dan dapatkan prediksi teks
    predicted_text = process_audio(best_model, audio_file_path, label_encoder, max_seq_length)
    print(f'Recognized Text: {predicted_text}')

if __name__ == "__main__":
    main()
