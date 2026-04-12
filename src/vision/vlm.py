# qwen_vl.py

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVL:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        device: str = "cuda",
        min_pixels: int = None,
        max_pixels: int = None,
        use_flash_attention: bool = False,
    ):
        self.device = device

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )

        processor_kwargs = {}
        if min_pixels:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels:
            processor_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    def _build_messages(self, prompt: str, image=None) -> list:
        """
        Build the message payload.
        `image` can be:
          - None (text only)
          - A local file path string  e.g. "/path/to/image.jpg"
          - A URL string              e.g. "http://..."
          - A base64 string           e.g. "data:image;base64,..."
        """
        content = []

        if image is not None:
            content.append({"type": "image", "image": image})

        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def generate(self, prompt: str, image=None, max_new_tokens: int = 512) -> str:
        messages = self._build_messages(prompt, image)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Trim input tokens from output
        trimmed_ids = [
            out[len(in_ids):]
            for in_ids, out in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]