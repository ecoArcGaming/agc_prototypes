# Test Transcriber (Ukranian Voice to English Text)
# run via python test_model/test_audio.py or update the path
from src.audio.transcriber import WhisperTranscriber
import os
from pathlib import Path

# Test all audio files in test_data/audio
transcriber = WhisperTranscriber()

audio_path = Path("tests/test_data/audio")
for file in audio_path.iterdir():
    print(f"\nTesting {file}...")
    translated_voice = transcriber.transcribe(file)
    print(f"Audio Translated to: {translated_voice}")

