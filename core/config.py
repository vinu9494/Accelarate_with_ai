import os
import sys
import builtins
from pathlib import Path
from dotenv import load_dotenv

# Windows consoles and Streamlit's stdout wrapper can fail on non-ASCII characters.
# sys.stdout.reconfigure() only works on raw TextIOWrapper streams — Streamlit
# replaces stdout with a custom writer that doesn't support it, so the reconfigure
# silently no-ops and any print() containing user-entered text (business intent, goal
# strings, etc.) raises OSError: [Errno 22] Invalid argument.
#
# Patch builtins.print globally so every module that imports config gets safe printing.
# This is the single place to fix rather than every print() call across all agents.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_original_print = builtins.print

def _safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except (UnicodeEncodeError, OSError):
        safe_args = tuple(
            a.encode("ascii", errors="replace").decode("ascii") if isinstance(a, str) else a
            for a in args
        )
        try:
            _original_print(*safe_args, **kwargs)
        except Exception:
            pass

builtins.print = _safe_print

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
PROFILES_DIR = DATA_DIR / "profiles"
STTM_DIR = DATA_DIR / "sttm"
BRONZE_DIR = DATA_DIR / "bronze_layer"
SILVER_DIR = DATA_DIR / "silver_layer"
GOLD_DIR = DATA_DIR / "gold_layer"
REPORTS_DIR = BASE_DIR / "reports"
AUDIT_DIR = BASE_DIR / "audit_logs"
CHROMA_DIR = BASE_DIR / ".chroma"


def _clean_env_value(value: str | None) -> str:
    return (value or "").strip()


PLACEHOLDER_API_KEYS = {
    "your_api_key_here",
    "your-anthropic-api-key-here",
    "replace_me",
    "changeme",
    "placeholder",
    "demo",
}

# Anthropic Configuration
ANTHROPIC_API_KEY = _clean_env_value(os.getenv("ANTHROPIC_API_KEY"))
ANTHROPIC_MODEL = _clean_env_value(os.getenv("ANTHROPIC_MODEL")) or "claude-3-5-sonnet-20241022"
LLM_MAX_TOKENS = max(1024, int(os.getenv("LLM_MAX_TOKENS", "4096")))
LLM_PROVIDER = "anthropic"


def validate_llm_configuration() -> None:
    """Fail fast with a clear message when the app is configured with a placeholder key."""
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.lower() in PLACEHOLDER_API_KEYS:
        raise ValueError(
            "Anthropic is not configured. Update the .env file with a real ANTHROPIC_API_KEY "
            "before running the pipeline."
        )

    if not ANTHROPIC_MODEL:
        raise ValueError("ANTHROPIC_MODEL is missing. Set it in the .env file.")


# Ensure directories exist
for d in [LANDING_DIR, PROFILES_DIR, STTM_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, REPORTS_DIR, AUDIT_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
