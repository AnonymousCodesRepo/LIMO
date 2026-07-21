"""YAML / JSON config loader + dotted-path CLI overrides."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

from .config import (
    DataConfig,
    EvalConfig,
    ExperiencePoolConfig,
    LLMConfig,
    PretrainConfig,
    ReportConfig,
    RunnerConfig,
    StageSpec,
)


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    """Parse a config file and ``--set a.b.c=value`` overrides.

    ``--output PATH`` is a shortcut for ``--set reporting.output=PATH``.
    """
    p = argparse.ArgumentParser(
        description=(
            "Run the standardized cascade evaluation. Pass a YAML / JSON "
            "config plus optional dotted-path overrides via --set."
        )
    )
    p.add_argument("--config", required=True,
                   help="Path to the YAML or JSON config file.")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="DOTTED.PATH=VALUE",
        help=(
            "Override a config field. May be repeated. Value is parsed as "
            "JSON (so use --set runner.workers=8, --set router.kwargs.threshold=0.6, "
            "--set llms.large_prediction_cache='[\"path/a.csv\"]')."
        ),
    )
    p.add_argument(
        "--output", default=None,
        help="Shortcut for --set reporting.output=...; required if "
             "the config doesn't already set it.",
    )
    args = p.parse_args(argv)

    raw = _load_file(args.config)
    for spec in args.overrides:
        _apply_override(raw, spec)
    if args.output is not None:
        raw.setdefault("reporting", {})["output"] = args.output

    cfg = _to_dataclass(EvalConfig, raw)
    if not cfg.reporting.output:
        raise SystemExit(
            "reporting.output is required (set it in the config or pass --output)."
        )
    return cfg


# ── file loading ─────────────────────────────────────────────────────────


def _load_file(path: str) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    if p.suffix.lower() == ".json":
        return json.loads(text)
    # Auto-detect by trying YAML (which is a JSON superset).
    return yaml.safe_load(text) or {}


# ── overrides ────────────────────────────────────────────────────────────


def _apply_override(raw: dict[str, Any], spec: str) -> None:
    if "=" not in spec:
        raise SystemExit(f"--set spec must be 'a.b.c=value', got: {spec!r}")
    key, _, val_str = spec.partition("=")
    parts = key.split(".")
    try:
        value: Any = json.loads(val_str)
    except json.JSONDecodeError:
        value = val_str
    cur = raw
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


# ── dict → dataclass ─────────────────────────────────────────────────────


def _to_dataclass(cls: type, data: Any) -> Any:
    """Recursively coerce a (possibly nested) dict into a dataclass tree.

    Handles the small surface we use: nested dataclasses, ``StageSpec |
    None``, lists, primitives. Unknown keys raise so a YAML typo can't
    silently select the wrong run.
    """
    if data is None:
        return None
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"expected dict for {cls.__name__}, got {type(data).__name__}")

    field_map = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(field_map)
    if unknown:
        raise SystemExit(
            f"unknown keys in {cls.__name__}: {sorted(unknown)} "
            f"(known: {sorted(field_map)})"
        )

    init_kwargs: dict[str, Any] = {}
    for name, f in field_map.items():
        if name not in data:
            continue
        val = data[name]
        init_kwargs[name] = _coerce(f.type, val)
    return cls(**init_kwargs)


def _coerce(annot: Any, val: Any) -> Any:
    """Coerce ``val`` to the type given by the field annotation.

    Annotation may be a class, a string (PEP 563 forward ref under
    ``from __future__ import annotations``), or a typing generic
    (``StageSpec | None``, ``list[str]``, ``dict[str, Any]``).
    """
    if isinstance(annot, str):
        annot = _resolve_forward_ref(annot)

    origin = get_origin(annot)
    if origin is None:
        if is_dataclass(annot):
            return _to_dataclass(annot, val)
        return val

    # `X | None` / `Optional[X]`
    args = get_args(annot)
    if origin is type(None) or _is_union(origin):
        non_none = [a for a in args if a is not type(None)]
        if val is None:
            return None
        if len(non_none) == 1:
            return _coerce(non_none[0], val)
        return val

    if origin in (list, tuple):
        if not isinstance(val, list):
            return val
        if args:
            return [_coerce(args[0], v) for v in val]
        return list(val)

    if origin is dict:
        return dict(val)

    return val


def _is_union(origin: Any) -> bool:
    import types
    import typing
    return origin in (typing.Union, getattr(types, "UnionType", None))


def _resolve_forward_ref(name: str) -> Any:
    """Map an annotation string back to the class object."""
    table: dict[str, Any] = {
        "DataConfig": DataConfig,
        "ExperiencePoolConfig": ExperiencePoolConfig,
        "StageSpec": StageSpec,
        "StageSpec | None": _opt(StageSpec),
        "PretrainConfig": PretrainConfig,
        "LLMConfig": LLMConfig,
        "RunnerConfig": RunnerConfig,
        "ReportConfig": ReportConfig,
        "list[str]": list,
        "dict[str, Any]": dict,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "str | None": _opt(str),
        "int | None": _opt(int),
        "float | None": _opt(float),
    }
    return table.get(name, object)


def _opt(t: type) -> Any:
    import typing
    return typing.Optional[t]
