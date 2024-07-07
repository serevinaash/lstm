import os
import pandas as pd
import numpy as np
import librosa
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Input, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define the dataset path
dataset_path = 'C:\\kuliah\\lstm\\Quran_Ayat_public\\audio_data\\Ghamadi_40kbps'

# Load transcripts
transcripts_path = 'C:\\kuliah\\lstm\\transcripts.tsv'
transcripts = pd.read_csv(transcripts_path, sep='\t')

# Load additional data from JSON
json_path = 'C:\\kuliah\\lstm\\all_ayat.json'
with open(json_path, 'r') as f:
    all_ayat = json.load(f)

# Load audio list and replace placeholder with actual path
audio_list_path = 'C:\\kuliah\\lstm\\audio_list.txt'
with open(audio_list_path, 'r') as f:
    audio_list = f.read().splitlines()
audio_list = [path.replace('${DATASET_PATH}', dataset_path) for path in audio_list]

# Load audio files
audio_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.mp3')]
audio_files.sort()
transcripts.sort_values(by='PATH', inplace=True)

# Feature extraction function with augmentation
def extract_features(file_path, augment=False):
    audio, sr = librosa.load(file_path, sr=None)
    if augment:
        audio = audio + 0.005 * np.random.randn(len(audio))  # Add white noise
        audio = librosa.effects.time_stretch(audio, rate=1.1)  # Time stretching
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)  # Pitch shifting
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs.T

# Extract features and align with transcripts
features = []
labels = []
for audio_file, transcript in zip(audio_files, transcripts['TRANSCRIPT']):
    mfccs = extract_features(audio_file)
    features.append(mfccs)
    labels.append(transcript)

    # Augmented data
    mfccs_aug = extract_features(audio_file, augment=True)
    features.append(mfccs_aug)
    labels.append(transcript)

# Pad sequences to ensure equal length
max_seq_length = max([feature.shape[0] for feature in features])
padded_features = pad_sequences(features, maxlen=max_seq_length, padding='post', dtype='float32')

# Normalize features
padded_features = (padded_features - np.mean(padded_features, axis=0)) \\ np.std(padded_features, axis=0)

# Encode labels as integers
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

# One-hot encode the integer labels
onehot_encoded = to_categorical(integer_encoded)

# Define the model with Bidirectional LSTM and tuned hyperparameters
model = Sequential([
    Input(shape=(max_seq_length, 13)),
    Masking(mask_value=0.0),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(128)),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Add callbacks for early stopping and model checkpoint
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
]

# Train the model with validation data
history = model.fit(padded_features, onehot_encoded, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks)

# Load the best model
best_model = tf.keras.models.load_model('best_model.keras')

# Evaluate the model
loss, accuracy = best_model.evaluate(padded_features, onehot_encoded)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
