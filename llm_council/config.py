from typing import Annotated, List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class CouncilConfig(BaseSettings):
    # 🔐 OpenRouter API Key (required)
    openrouter_api_key: str = Field(validation_alias="OPENROUTER_API_KEY")

    # 🤖 Models list
    # NoDecode: read the env value as plain text (comma-separated), not JSON
    council_models: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: [
            "openai/gpt-4.1-mini",
            "qwen/qwen3.5-plus-02-15",
            "z-ai/glm-5",
        ],
        validation_alias="COUNCIL_MODELS",
    )

    # 🧠 Strategy
    council_strategy: Literal["synthesis", "voting"] = Field(
        default="synthesis", validation_alias="COUNCIL_STRATEGY"
    )
    # Model that merges the answers in "synthesis"; empty = first council model
    judge_model: str = Field(default="", validation_alias="COUNCIL_JUDGE_MODEL")

    max_retries: int = Field(default=3, validation_alias="COUNCIL_MAX_RETRIES")
    timeout: float = Field(default=30.0, validation_alias="COUNCIL_TIMEOUT")

    # 🌐 OpenRouter base URL
    api_base: str = "https://openrouter.ai/api/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("council_models", mode="before")
    @classmethod
    def parse_council_models(cls, v):
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v


# Singleton config loader
_config_cache: CouncilConfig | None = None


def get_config() -> CouncilConfig:
    global _config_cache
    if _config_cache is None:
        _config_cache = CouncilConfig()

        if not _config_cache.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    return _config_cache
