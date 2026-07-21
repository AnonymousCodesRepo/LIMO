"""Drive the offline pretrain pipeline when ``pretrain.mode == "fit"``.

Chains the two offline warm-start stages, spawning each as a subprocess
through the repo's ``run.sh`` wrapper:

    Stage 1 — query synthesis → <output_dir>/candidates.jsonl
        (contrastively-mined predicates, each applied to
        ``docs_per_query`` documents so active labeling can select
        document–query pairs per query across rounds — paper §6.1/§6.2)
    Stage 2 — paper_pretrain  → <output_dir>/router_lgbm.pkl
                                <output_dir>/care_snapshot.json
                                <output_dir>/experiences.jsonl
                                <output_dir>/rollouts.jsonl

Stage 2 is the paper-faithful JOINT warm-start (LIMO Section 6): the
cost-aware active-labeling loop trains the router AND the experience
utility estimator together, bootstrapping the experience pool E_0 from
labeling disagreements, and writes the final router checkpoint. Both
learned modules are then loaded by the eval: ``router_checkpoint`` and
``retriever_snapshot`` are pointed at the produced artifacts and the run
proceeds as if ``pretrain.mode`` were ``"load"``.

Each stage is resumable: if its output artifact already exists, the stage
is skipped, so a half-finished fit resumes by re-invoking the same command.

The YAML schema is nested per stage:

    pretrain:
      mode: fit
      output_dir: <dir>
      fit:
        synth:
          n_predicates: 40
          docs_per_query: 50
          n_clusters: 20
          rng_seed: 0
        label:
          rounds: 4
          budget_per_round: [100, 100, 100, 100]
          k_experiences: 8
          seed: 0

Each subdict's ``key: value`` becomes ``--key-with-dashes value`` on the
corresponding stage's command line; list values render as multiple args
after the flag (e.g. ``--budget-per-round``).
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .config import PretrainConfig


REPO_ROOT = Path(__file__).resolve().parent.parent


def fit_warm_start(cfg: PretrainConfig, *, dataset: str) -> tuple[str, str]:
    """Run synth → joint pretrain. Returns ``(checkpoint, snapshot)`` paths.

    Raises ``SystemExit`` if ``output_dir`` is missing or any stage's
    subprocess returns a non-zero exit code.
    """
    if not cfg.output_dir:
        raise SystemExit(
            "pretrain.output_dir is required when pretrain.mode='fit'."
        )

    out_dir = _abs_path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = out_dir / "candidates.jsonl"
    checkpoint = out_dir / "router_lgbm.pkl"
    snapshot = out_dir / "care_snapshot.json"

    fit_kwargs = cfg.fit or {}
    synth_kwargs = dict(fit_kwargs.get("synth", {}))
    label_kwargs = dict(fit_kwargs.get("label", {}))

    _run_stage(
        artifact=candidates,
        label="synth",
        script="scripts/pretrain_offline/synth_queries_multidoc.py",
        fixed={"output": candidates, "dataset": dataset},
        kwargs=synth_kwargs,
    )
    _run_stage(
        artifact=checkpoint,
        label="joint-pretrain",
        script="scripts/pretrain_offline/paper_pretrain.py",
        fixed={"candidates": candidates, "dataset": dataset,
               "out_dir": out_dir},
        kwargs=label_kwargs,
    )
    if not snapshot.exists():
        raise SystemExit(
            f"[pretrain.fit] joint-pretrain produced no retriever snapshot "
            f"at {snapshot}."
        )
    print(
        f"[pretrain.fit] done. checkpoint: {checkpoint}  "
        f"snapshot: {snapshot}",
        flush=True,
    )
    return str(checkpoint), str(snapshot)


# ── internals ────────────────────────────────────────────────────────────


def _run_stage(
    *,
    artifact: Path,
    label: str,
    script: str,
    fixed: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> None:
    """Spawn one stage script via ``run.sh``.

    ``fixed`` are flag → value pairs that the eval orchestrator owns
    (input / output paths). ``kwargs`` are user-supplied hyperparameter
    overrides from the YAML's ``pretrain.fit.<stage>`` block. ``fixed``
    is applied first, then ``kwargs``; on key collision the YAML wins
    (so the user can override e.g. an output path if they really want).
    """
    if artifact.exists():
        print(
            f"[pretrain.fit] skip {label} (artifact exists): {artifact}",
            flush=True,
        )
        return

    overlap = set(fixed) & set(kwargs)
    if overlap:
        print(
            f"[pretrain.fit] warning: pretrain.fit.{label} overrides "
            f"orchestrator-owned flags: {sorted(overlap)}",
            flush=True,
        )

    merged: dict[str, Any] = {**fixed, **kwargs}
    cmd: list[str] = ["bash", str(REPO_ROOT / "run.sh"), script]
    for key, value in merged.items():
        cmd.extend(_render_flag(key, value))

    pretty = " ".join(shlex.quote(part) for part in cmd)
    print(f"[pretrain.fit] STAGE {label}: {pretty}", flush=True)

    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise SystemExit(
            f"[pretrain.fit] stage {label!r} failed (return code {rc}). "
            f"See stdout above. Re-running with the same output_dir will "
            f"resume from this stage."
        )
    if not artifact.exists():
        raise SystemExit(
            f"[pretrain.fit] stage {label!r} returned 0 but did not "
            f"produce {artifact}."
        )


def _render_flag(key: str, value: Any) -> list[str]:
    """Render one ``key: value`` pair as command-line tokens.

    * ``bool`` true / false → ``--key-with-dashes`` is added / omitted
      (matching argparse ``action="store_true"`` flags).
    * ``list`` / ``tuple`` → ``--flag v1 v2 v3`` (matches ``nargs='+'``).
    * Anything else → ``--flag str(value)``.
    """
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        return [flag] if value else []
    if isinstance(value, (list, tuple)):
        return [flag, *(str(item) for item in value)]
    return [flag, str(value)]


def _abs_path(path: str) -> Path:
    pp = Path(path)
    return pp if pp.is_absolute() else REPO_ROOT / pp
