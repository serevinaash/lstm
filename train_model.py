import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Input, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define the dataset paths
dataset_paths = [
    'C:\\lstm\\audio_data'
]

# Load transcripts
transcripts_path = 'C:\\lstm\\filter_transcripts.tsv'
transcripts = pd.read_csv(transcripts_path, sep='\t')

# Replace placeholder with actual path in transcripts
for dataset_path in dataset_paths:
    transcripts['PATH'] = transcripts['PATH'].str.replace('${DATASET_PATH}', dataset_path)

# Load audio files
audio_files = []
for dataset_path in dataset_paths:
    audio_files.extend([os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.mp3')])
audio_files = sorted(audio_files)

# Filter transcripts to match audio files
transcripts = transcripts[transcripts['PATH'].isin(audio_files)]

# Feature extraction function with augmentation and delta MFCC
def extract_features(file_path, augment=False):
    print(f"Processing file: {file_path}")
    try:
        audio, sr = librosa.load(file_path, sr=None)
        if audio is None:
            raise ValueError(f"Empty audio file: {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    try:
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
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Extract features and align with transcripts
features = []
labels = []
for audio_file, transcript in zip(transcripts['PATH'], transcripts['TRANSCRIPT']):
    mfccs = extract_features(audio_file)
    if mfccs is not None and mfccs.shape[0] > 0:  # Check if mfccs is not empty
        features.append(mfccs)
        labels.append(transcript)

        # Augmented data
        mfccs_aug = extract_features(audio_file, augment=True)
        if mfccs_aug is not None and mfccs_aug.shape[0] > 0:  # Check if mfccs_aug is not empty
            features.append(mfccs_aug)
            labels.append(transcript)

# Handle case where no features are extracted
if not features:
    raise ValueError("No features extracted. Check your audio files and extraction process.")

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

# Define the model with increased complexity
model = Sequential([
    Input(shape=(max_seq_length, 26)),  # Update input shape for MFCC + delta MFCC
    Masking(mask_value=0.0),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(256)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Add callbacks for early stopping and model checkpoint
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
]

# Train the model with validation data
history = model.fit(padded_features, onehot_encoded, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks)

# Load the best model
best_model = tf.keras.models.load_model('best_model.keras')

# Save the best model for deployment
best_model.save('final_model.keras')
print("Best model saved as 'final_model.keras'.")
