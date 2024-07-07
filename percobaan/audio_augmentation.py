from pydub import AudioSegment
import os

def mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Contoh penggunaan
mp3_file = "C:\\kuliah\\lstm\\Quran_Ayat_public\\audio_data\\Ghamadi_40kbps"
wav_file = "C:\\kuliah\\lstm\\audio_wav"

mp3_to_wav(mp3_file, wav_file)
