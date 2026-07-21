"""Zero-shot prediction for ALL queries in LegalBench document_query_matrix.

For each (document, query) pair, ask the model whether the document satisfies
the query condition. Results are saved per-model as CSV files.

Models: Llama-3.1-70B-Instruct (port 8102), Llama-3.1-8B-Instruct (port 8104)

Usage:
    python examples/eval_zeroshot_all_queries.py --model 70b
    python examples/eval_zeroshot_all_queries.py --model 8b
    python examples/eval_zeroshot_all_queries.py --model both

    # Resume from checkpoint
    python examples/eval_zeroshot_all_queries.py --model 70b --resume

    # Limit documents for testing
    python examples/eval_zeroshot_all_queries.py --model 8b --max-docs 5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "LegalBench"
OUTPUT_DIR = Path(__file__).resolve().parent / "predictions" / "zeroshot_all_queries"

# ── Model configs ────────────────────────────────────────────────
MODELS = {
    "70b": {
        "name": "Llama-3.1-70B-Instruct",
        "model_path": os.environ.get("MOP_LARGE_MODEL", "Meta-Llama-3.1-70B-Instruct"),
        "base_url": os.environ.get("MOP_LARGE_URL", "http://localhost:8102/v1"),
        "max_workers": 8,
    },
    "8b": {
        "name": "Llama-3.1-8B-Instruct",
        "model_path": "Meta-Llama-3.1-8B-Instruct",
        "base_url": "http://localhost:8104/v1",
        "max_workers": 8,
    },
}

SYSTEM_PROMPT = (
    "You are a legal clause classifier. "
    "Given a legal document and a question about the document, "
    "answer with exactly one word: Yes or No."
)


def make_messages(query_description: str, document_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Question: {query_description}\n\n"
            f"Document:\n{document_text}\n\n"
            f"Answer (Yes or No):"
        )},
    ]


def parse_answer(raw: str) -> str:
    """Parse Yes/No from model response. Returns 'Yes', 'No', or 'PARSE_FAIL'."""
    if not raw:
        return "PARSE_FAIL"
    text = raw.strip()
    # Check last non-empty line
    last_lines = [l.strip().lower() for l in text.split("\n") if l.strip()]
    if last_lines:
        last = last_lines[-1]
        if last in ("yes", "yes.", "no", "no."):
            return "Yes" if last.startswith("yes") else "No"
    low = text.lower()
    if low.startswith("yes"):
        return "Yes"
    if low.startswith("no"):
        return "No"
    m = re.search(r"\b(Yes|No)\b", text)
    if m:
        return m.group(1)
    return "PARSE_FAIL"


def load_data():
    """Load documents, queries, and ground truth matrix."""
    merged_df = pd.read_csv(DATASET_DIR / "merged_df.csv")
    docs = {}
    for _, row in merged_df.iterrows():
        docs[int(row["document_id"])] = {
            "name": row["document_name"],
            "text": str(row["merged_text"]),
        }

    query_map_df = pd.read_csv(DATASET_DIR / "query_name_mapping.csv")
    queries = {}
    for _, row in query_map_df.iterrows():
        qname = str(row["query_name"]).strip()
        if qname:
            queries[qname] = str(row["query_description"]).strip()

    matrix_df = pd.read_csv(DATASET_DIR / "document_query_matrix.csv")
    ground_truth = {}
    for _, row in matrix_df.iterrows():
        doc_id = int(row["document_id"])
        for qname in queries:
            if qname in matrix_df.columns:
                gt = str(row[qname]).strip()
                ground_truth[(doc_id, qname)] = gt

    return docs, queries, ground_truth


def load_checkpoint(output_file: Path) -> set[tuple[int, str]]:
    done = set()
    if output_file.exists():
        with open(output_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((int(row["document_id"]), row["query_name"]))
    return done


def run_predictions(
    model_key: str,
    docs: dict,
    queries: dict,
    ground_truth: dict,
    max_docs: int | None = None,
    resume: bool = False,
):
    cfg = MODELS[model_key]
    model_name = cfg["name"]
    model_path = cfg["model_path"]
    base_url = cfg["base_url"]
    max_workers = cfg["max_workers"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{model_name}_zeroshot.csv"

    client = OpenAI(base_url=base_url, api_key="dummy")
    try:
        client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5, temperature=0.0,
        )
        print(f"[OK] Connected to {model_name} at {base_url}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to {model_name} at {base_url}: {e}")
        return

    doc_ids = sorted(docs.keys())
    if max_docs:
        doc_ids = doc_ids[:max_docs]
    query_names = sorted(queries.keys())

    tasks = [(doc_id, qname) for doc_id in doc_ids for qname in query_names]

    done = set()
    if resume:
        done = load_checkpoint(output_file)
        print(f"Resuming: {len(done)} completed, {len(tasks) - len(done)} remaining")
    tasks = [(d, q) for d, q in tasks if (d, q) not in done]

    if not tasks:
        print("All predictions already completed!")
        return

    total = len(tasks)
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Documents: {len(doc_ids)}, Queries: {len(query_names)}")
    print(f"Total predictions: {total}")
    print(f"Output: {output_file}")
    print(f"Workers: {max_workers}")
    print(f"{'='*70}\n")

    write_header = not output_file.exists() or not resume
    f_out = open(output_file, "a" if resume and output_file.exists() else "w", newline="")
    writer = csv.DictWriter(f_out, fieldnames=[
        "document_id", "document_name", "query_name", "query_description",
        "ground_truth", "prediction", "correct", "raw_response",
    ])
    if write_header:
        writer.writeheader()

    completed = 0
    correct = 0
    parse_fails = 0
    t_start = time.time()

    def predict_one(doc_id: int, qname: str) -> dict:
        messages = make_messages(queries[qname], docs[doc_id]["text"])
        try:
            resp = client.chat.completions.create(
                model=model_path,
                messages=messages,
                temperature=0.0,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            raw = f"ERROR: {e}"

        pred = parse_answer(raw)
        gt = ground_truth.get((doc_id, qname), "N/A")
        is_correct = pred == gt

        return {
            "document_id": doc_id,
            "document_name": docs[doc_id]["name"],
            "query_name": qname,
            "query_description": queries[qname][:100],
            "ground_truth": gt,
            "prediction": pred,
            "correct": is_correct,
            "raw_response": raw.strip()[:200],
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for doc_id, qname in tasks:
            fut = executor.submit(predict_one, doc_id, qname)
            futures[fut] = (doc_id, qname)

        for fut in as_completed(futures):
            result = fut.result()
            writer.writerow(result)
            f_out.flush()

            completed += 1
            if result["correct"]:
                correct += 1
            if result["prediction"] == "PARSE_FAIL":
                parse_fails += 1

            if completed % 200 == 0 or completed == total:
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                acc = correct / completed if completed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(
                    f"  [{completed:>6}/{total}] "
                    f"Acc={acc:.4f} PF={parse_fails} "
                    f"Rate={rate:.1f}/s ETA={eta/60:.1f}min"
                )

    f_out.close()

    elapsed = time.time() - t_start
    acc = correct / completed if completed > 0 else 0
    print(f"\n{'='*70}")
    print(f"DONE: {model_name}")
    print(f"  Total: {completed}, Correct: {correct}, Accuracy: {acc:.4f}")
    print(f"  Parse failures: {parse_fails}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {output_file}")
    print(f"{'='*70}\n")

    # Per-query accuracy summary
    print("Per-query accuracy:")
    results_df = pd.read_csv(output_file)
    query_acc = results_df.groupby("query_name")["correct"].mean().sort_values()
    for qname, acc in query_acc.items():
        print(f"  {qname:<55} {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot LLM prediction on all LegalBench queries")
    parser.add_argument("--model", choices=["70b", "8b", "both"], default="both")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    docs, queries, ground_truth = load_data()
    print(f"Loaded {len(docs)} documents, {len(queries)} queries")
    print(f"Total (doc, query) pairs: {len(docs) * len(queries)}")

    models_to_run = ["70b", "8b"] if args.model == "both" else [args.model]

    for model_key in models_to_run:
        run_predictions(
            model_key=model_key,
            docs=docs,
            queries=queries,
            ground_truth=ground_truth,
            max_docs=args.max_docs,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
