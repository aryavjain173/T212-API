"""Config loading. Credentials come from the environment, never the repo."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_secret: str | None
    environment: str
    allow_live: bool

    @classmethod
    def load(cls) -> "Settings":
        key = os.getenv("T212_API_KEY")
        if not key:
            raise RuntimeError(
                "T212_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        env = os.getenv("T212_ENVIRONMENT", "demo").strip().lower()
        allow_live = os.getenv("T212_ALLOW_LIVE", "false").strip().lower() == "true"
        return cls(
            api_key=key,
            api_secret=os.getenv("T212_API_SECRET") or None,
            environment=env,
            allow_live=allow_live,
        )


def build_client():
    from .client import T212Client

    s = Settings.load()
    return T212Client(
        api_key=s.api_key,
        api_secret=s.api_secret,
        environment=s.environment,
        allow_live=s.allow_live,
    )
