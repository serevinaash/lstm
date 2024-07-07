import pandas as pd

# Ganti path ini dengan path file yang benar
file_path = 'C:/lstm/transcripts.tsv'
output_path = 'C:/lstm/filter_transcripts.tsv'

# Membaca file TSV ke dalam DataFrame
df = pd.read_csv(file_path, sep='\t')

# Menampilkan beberapa baris pertama untuk memastikan data telah terbaca dengan benar
print(df.head())

# Memfilter transkrip yang path-nya dimulai dengan path yang diberikan
dataset_path = 'C:\\lstm\\audio_data'
filtered_df = df[df['PATH'].str.startswith(dataset_path)]

# Menyimpan hasil filter ke file baru
filtered_df.to_csv(output_path, sep='\t', index=False)

print(f'Hasil filter disimpan ke {output_path}')
