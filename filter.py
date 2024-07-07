import pandas as pd

# Path ke transcript
transcripts_path = 'C:\\kuliah\\lstm\\transcripts.tsv'
dataset_path = 'C:\\kuliah\\lstm\\ghamadi'  # Sesuaikan dengan path dataset Anda

# Load transcripts
try:
    transcripts = pd.read_csv(transcripts_path, sep='\t')
except FileNotFoundError:
    print(f"File not found: {transcripts_path}")
    exit(1)

# Replace placeholder with actual dataset paths in transcripts
transcripts['PATH'] = transcripts['PATH'].str.replace('${DATASET_PATH}', dataset_path)

# Simpan kembali file transcript yang telah diubah
transcripts.to_csv(transcripts_path, sep='\t', index=False)

print("Transcripts berhasil diupdate dengan paths yang benar.")
