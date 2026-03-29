from research_agent.config import AppSettings
from research_agent.services.gemini_text import GeminiTextService
from research_agent.services.groq_text import GroqTextService


class TextGenerationService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._groq = GroqTextService(settings)
        self._gemini = GeminiTextService(settings)

    @property
    def available(self) -> bool:
        provider = self._provider()
        if provider == "groq":
            return self._groq.available
        if provider == "gemini":
            return self._gemini.available
        return self._groq.available or self._gemini.available

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> str:
        provider = self._provider()
        if provider == "groq":
            return self._groq.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        if provider == "gemini":
            return self._gemini.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        return self._generate_with_auto_fallback(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def _generate_with_auto_fallback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        attempts: list[str] = []
        providers = [
            ("groq", self._groq),
            ("gemini", self._gemini),
        ]
        for name, service in providers:
            if not service.available:
                continue
            try:
                return service.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as error:  # pragma: no cover - provider-specific network errors
                attempts.append(f"{name}: {str(error)[:180]}")
                continue

        if attempts:
            joined = " | ".join(attempts)
            raise RuntimeError(f"All generation providers failed. {joined}")

        raise RuntimeError(
            "No generation provider configured. Set GROQ_API_KEY or GEMINI_API_KEY."
        )

    def _provider(self) -> str:
        provider = (self._settings.generation_provider or "auto").strip().lower()
        if provider in {"auto", "groq", "gemini"}:
            return provider
        return "auto"
