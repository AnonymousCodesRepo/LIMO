# ScaleDoc upstream patches (LIMO baseline port)

We drive the official ScaleDoc (https://github.com/Seurgul/ScaleDoc) with
mpnet + cached 70B predictions. We made 3 **non-algorithmic** changes to its
`src/`, solely so it runs in our environment / on our data without crashing.
The algorithmic logic (contrastive encoder, dual-threshold cascade) is
unchanged.

## 1. `src/utils/embed.py` — lazy llm2vec import

Our Python environment has no `llm2vec` (we use mpnet and never instantiate
`L2V_Encoder`), but the top-level `from llm2vec import LLM2Vec` makes
`import utils.train` fail. Change:
- delete the top-level `from llm2vec import LLM2Vec`
- do `from llm2vec import LLM2Vec` inside `L2V_Encoder.encode()` (lazy)

## 2. `src/cascade.py` — threshold-search bounds guard

The while loops in `select_sim_filter_AT` and `select_sim_filterB` can push
`r_idx`/`l_idx` past the end of the `steps` array (`IndexError: index N out
of bounds for axis 0 with size N`), which reliably happens on our
reconstructed distributions. Add at the top of both while-loop bodies:

```python
        if r_idx >= len(steps) - 1 or l_idx + 1 >= len(steps) - 1:
            break
```

(Only guards the bounds and returns the best bounds found so far; the search
intent is unchanged.)

## 3. Selector: use `select_sim_filterB` (controlled by `run_scaledoc.py`, not a cascade.py edit)

The repo's `cascade.py:cascade()` enables `select_sim_filter_AT` by default,
but on our data it collapses (le≈re, the uncertainty band is always empty,
accuracy worse than random). `run_scaledoc.py` switches to
`select_sim_filterB` (the F1 version of the paper's Algorithm 2) via
`SCALEDOC_SELECTOR=B` (the default).

## Redeploying

Clone the official repo anywhere, apply the 3 changes above, and place
`run_scaledoc.py` (from this directory) into its `src/`. Outputs are written
under `runs/`.
