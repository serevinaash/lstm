import os
import pandas as pd
import numpy as np
import librosa
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import custom_object_scope

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

# Feature extraction function
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs.T

# Extract features and align with transcripts
features = []
labels = []
for audio_file, transcript in zip(audio_files, transcripts['TRANSCRIPT']):
    mfccs = extract_features(audio_file)
    features.append(mfccs)
    labels.append(transcript)

# Pad sequences to ensure equal length
max_seq_length = max([feature.shape[0] for feature in features])
padded_features = pad_sequences(features, maxlen=max_seq_length, padding='post', dtype='float32')

# Encode labels as integers
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

# One-hot encode the integer labels
onehot_encoded = to_categorical(integer_encoded)

# Define the model
model = Sequential([
    Input(shape=(max_seq_length, 13)),
    Masking(mask_value=0.0),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(128),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with validation data
history = model.fit(padded_features, onehot_encoded, epochs=50, batch_size=32, validation_split=0.2)

# Save the model in Keras native format
model.save('my_model.keras')
print("Model saved as my_model.keras")

# Load model from Keras native format with custom object registration
class NotEqualLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(NotEqualLayer, self).__init__()

    def call(self, inputs):
        return tf.not_equal(inputs[0], inputs[1])

# Register the custom layer
with custom_object_scope({'NotEqualLayer': NotEqualLayer}):
    loaded_model = tf.keras.models.load_model('my_model.keras')

print("Model loaded from my_model.keras")

# Evaluate the model
loss, accuracy = loaded_model.evaluate(padded_features, onehot_encoded)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
