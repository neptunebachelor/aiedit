from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_JOBS_DIR = Path("workspace") / "jobs"
SOURCE_NAME = "gemini_files_api"


def now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def fail(message: str, *, code: int = 1) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_env() -> None:
    load_dotenv()


def resolve_api_key(*, api_key: str | None, api_key_env: str) -> str:
    load_env()
    resolved = (api_key or "").strip()
    if resolved:
        return resolved

    env_name = (api_key_env or DEFAULT_API_KEY_ENV).strip()
    if env_name:
        resolved = os.environ.get(env_name, "").strip()
        if resolved:
            return resolved

    if env_name != "GOOGLE_API_KEY":
        resolved = os.environ.get("GOOGLE_API_KEY", "").strip()
        if resolved:
            return resolved

    raise RuntimeError(f"Missing Gemini API key. Set {env_name or DEFAULT_API_KEY_ENV} or pass --api-key.")


def make_client(*, api_key: str | None, api_key_env: str) -> Any:
    if genai is None:
        raise RuntimeError("Missing dependency google-genai. Run: pip install google-genai")
    return genai.Client(api_key=resolve_api_key(api_key=api_key, api_key_env=api_key_env))


def resolve_existing_file(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")
    return path


def detect_mime_type(path: Path, explicit_mime_type: str | None) -> str:
    mime_type = (explicit_mime_type or "").strip()
    if mime_type:
        return mime_type

    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed

    raise ValueError(f"Could not infer MIME type for {path}. Pass --mime-type explicitly.")


def make_job_id(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        raw = f"{datetime.now().astimezone():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not clean:
        raise ValueError(f"Invalid job id: {value!r}")
    return clean


def get_value(obj: Any, *names: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    data = to_jsonable(obj)
    if isinstance(data, dict):
        for name in names:
            if name in data:
                return data[name]
    return None


def normalize_state(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "name"):
        return str(value.name)
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple | set):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="json", by_alias=False))
    if hasattr(value, "to_json_dict"):
        return to_jsonable(value.to_json_dict())
    if hasattr(value, "__dict__"):
        return {
            key: to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
    return str(value)


def copy_original_file(source: Path, job_dir: Path) -> Path:
    destination = job_dir / "original.bin"
    shutil.copy2(source, destination)
    return destination
