import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Input, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define dataset paths and transcripts path
dataset_paths = [
    'C:\\kuliah\\lstm\\Quran_Ayat_public\\audio_data\\Ghamadi_40kbps'
]
transcripts_path = 'C:\\kuliah\\lstm\\Quran_Ayat_public\\transcripts.tsv'

# Load transcripts and process
try:
    transcripts = pd.read_csv(transcripts_path, sep='\t')
except FileNotFoundError:
    print(f"File not found: {transcripts_path}")
    exit(1)

# Replace placeholder with actual dataset paths in transcripts
for dataset_path in dataset_paths:
    transcripts['PATH'] = transcripts['PATH'].str.replace('${DATASET_PATH}', dataset_path)

# Filter transcripts to match available audio files
audio_files = []
for dataset_path in dataset_paths:
    audio_files.extend([os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.mp3')])
audio_files = sorted(audio_files)

transcripts = transcripts[transcripts['PATH'].isin(audio_files)]

# Check if number of audio files and transcripts match
if len(audio_files) != len(transcripts):
    raise ValueError(f"Number of audio files ({len(audio_files)}) does not match number of transcripts ({len(transcripts)})")

# Function to extract MFCC features from audio files
def extract_features(file_path, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if augment:
        # Apply augmentation techniques
        if np.random.rand() < 0.5:
            audio = audio + 0.005 * np.random.randn(len(audio))  # Add white noise
        if np.random.rand() < 0.5:
            audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))  # Time stretching
        if np.random.rand() < 0.5:
            audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=np.random.randint(-3, 3))  # Pitch shifting

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    combined = np.concatenate((mfccs, delta_mfccs), axis=0)
    return combined.T

# Extract features and align with transcripts
features = []
labels = []
for audio_file, transcript in zip(transcripts['PATH'], transcripts['TRANSCRIPT']):
    mfccs = extract_features(audio_file)
    if mfccs is not None:
        features.append(mfccs)
        labels.append(transcript)

        # Augment data (optional)
        mfccs_aug = extract_features(audio_file, augment=True)
        if mfccs_aug is not None:
            features.append(mfccs_aug)
            labels.append(transcript)

# Pad sequences to ensure equal length
max_seq_length = max([feature.shape[0] for feature in features])
padded_features = pad_sequences(features, maxlen=max_seq_length, padding='post', dtype='float32')

# Normalize features
padded_features = (padded_features - np.mean(padded_features, axis=0)) / np.std(padded_features, axis=0)

# Encode labels as integers
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

# One-hot encode the integer labels
onehot_encoded = to_categorical(integer_encoded)

# Define LSTM model
model = Sequential([
    Input(shape=(max_seq_length, 26)),  # Input shape for MFCC + delta MFCC
    Masking(mask_value=0.0),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(256)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Define callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
]

# Train the model with validation data
history = model.fit(padded_features, onehot_encoded, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks)

# Load the best model
best_model = tf.keras.models.load_model('best_model.keras')

# Evaluate the model
loss, accuracy = best_model.evaluate(padded_features, onehot_encoded)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
