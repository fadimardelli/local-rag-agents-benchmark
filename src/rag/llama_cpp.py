from pathlib import Path
from typing import Optional

from llama_cpp import Llama


class LlamaCppModel:
    def __init__(
        self,
        model_path: Path,
        n_ctx: int,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        output = self.model.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return output["choices"][0]["message"]["content"].strip()
