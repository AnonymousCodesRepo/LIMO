"""Compute all-mpnet-base-v2 embeddings for the ScaleDoc baseline (GPU server).

Runs on GPU server where the shared embedding server (:8200, the SAME
all-mpnet-base-v2 our other methods use) is reachable. For each dataset it reads
``<data>/doc/{ds}.json`` and ``<data>/query.json`` and writes, under
``<data>/embeds``:
  {ds}_embeds_mpnet.pt : torch.float32 [N_docs, 768], row = document_id
  {ds}_qemb_mpnet.pt   : torch.float32 [N_queries, 768], row = q_id

mpnet outputs are already L2-normalized by the server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import torch


def embed(texts, url, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        r = requests.post(url, json={"texts": texts[i:i + bs]}, timeout=120)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    return torch.tensor(out, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="scaledoc_data dir")
    ap.add_argument("--url", default="http://localhost:8200/embed")
    ap.add_argument("--datasets", default="cuad,contract,opp115,hoc,sembench_medical")
    args = ap.parse_args()
    data = Path(args.data)
    qj = json.loads((data / "query.json").read_text())

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        docs = json.loads((data / "doc" / f"{ds}.json").read_text())
        texts = [d["content"] for d in sorted(docs, key=lambda x: x["id"])]
        demb = embed(texts, args.url)
        torch.save(demb, data / "embeds" / f"{ds}_embeds_mpnet.pt")
        qtexts = [q["query"] for q in sorted(qj[ds], key=lambda x: x["q_id"])]
        qemb = embed(qtexts, args.url)
        torch.save(qemb, data / "embeds" / f"{ds}_qemb_mpnet.pt")
        print(f"{ds:16s} docs={demb.shape} queries={qemb.shape}")


if __name__ == "__main__":
    main()
