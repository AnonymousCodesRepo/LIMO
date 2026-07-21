"""Standardized end-to-end evaluation for the cascade pipeline.

Configs are YAML / JSON files (loaded into the ``EvalConfig`` dataclass) so
each stage — data loader, experience pool, experience retriever, online
generator, fewshot retriever, router, pretrain (load or fit), LLM endpoints,
runner, reporting — is an independent, swappable module.

Entry point::

    python -m eval --config configs/examples/zs_small.yaml \\
        [--set router.kwargs.threshold=0.6 ...]
"""

from __future__ import annotations
