# piper_tts.py

import subprocess
import numpy as np
import soundfile as sf


class PiperTTS:
    def __init__(
        self,
        model_path: str = "/app/models/piper/en_US-lessac-medium.onnx",
        output_path: str = "/tmp/response.wav",
    ):
        self.model_path = model_path
        self.output_path = output_path

    def speak(self, text: str) -> tuple[np.ndarray, int]:
        """
        Convert text to audio using Piper TTS.

        Args:
            text: Text to synthesize.

        Returns:
            Tuple of (audio_array int16, sample_rate).
        """
        subprocess.run(
            ["piper", "--model", self.model_path, "--output_file", self.output_path],
            input=text.encode(),
            check=True,
        )

        audio_data, sample_rate = sf.read(self.output_path, dtype="int16")
        return audio_data, sample_rate