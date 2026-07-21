"""Offline trainers for router and experience-retriever heads.

Trainers consume `Rollout` records from `pipeline.label_synth` and emit
checkpoints in formats the corresponding online stages can warm-start
from. Each sub-module pairs with one online stage:

* ``trainer.router.offline_lgbm``  ↔  ``router.lightgbm_router``
* ``trainer.retriever.offline_care`` ↔ ``experience_retriever.care``
"""

__all__: list[str] = []
