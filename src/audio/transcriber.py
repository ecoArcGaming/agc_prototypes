# whisper_transcriber.py

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class WhisperTranscriber:
    def __init__(self, model_name="openai/whisper-tiny", language="ukrainian", task="translate"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )

def transcribe(self, audio: np.ndarray, fs: int = 16000) -> str:
    # Convert int16 → float32 (librosa/whisper expect float32 in range -1.0 to 1.0)
    audio_float = audio.flatten().astype(np.float32) / 32768.0

    input_features = self.processor(
        audio_float,
        sampling_rate=fs,
        return_tensors="pt"
    ).input_features

    predicted_ids = self.model.generate(
        input_features,
        forced_decoder_ids=self.forced_decoder_ids
    )

    return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]