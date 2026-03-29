from google import genai
from google.genai import types

from research_agent.config import AppSettings


class GeminiTextService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client: genai.Client | None = None

    @property
    def available(self) -> bool:
        return bool(self._settings.gemini_api_key)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> str:
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini generation.")

        response = self._client_or_create().models.generate_content(
            model=self._settings.gemini_generation_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

        text = getattr(response, "text", "")
        if isinstance(text, str) and text.strip():
            return text.strip()

        raise RuntimeError("Gemini response did not include output text.")

    def _client_or_create(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client
