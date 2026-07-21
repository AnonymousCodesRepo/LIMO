"""Collect Qwen3.5-0.8B confidence with 3-way unsure prompt for all document-query pairs.

Uses the 3-choice prompt: {"match": true} / {"match": false} / {"match": "unsure"}
Extracts P(true), P(false), P(unsure) from logprobs.

Output: Qwen3.5-0.8B_confidence_3way.csv

Usage:
    python examples/collect_qwen08b_confidence_3way.py
    python examples/collect_qwen08b_confidence_3way.py --max-docs 5
    python examples/collect_qwen08b_confidence_3way.py --resume
"""

from __future__ import annotations
import os

import argparse
import csv
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "LegalBench"
OUTPUT_DIR = Path(__file__).resolve().parent / "predictions" / "zeroshot_all_queries"

MODEL_PATH = os.environ.get("MOP_SMALL_MODEL", "Qwen3.5-0.8B")
BASE_URL = os.environ.get("MOP_SMALL_URL", "http://localhost:8105/v1")
MAX_WORKERS = 8

SYSTEM_PROMPT = (
    "You are a precise data filtering assistant. "
    "Determine whether the given document satisfies the condition. "
    "If you are uncertain, it is better to say so rather than guessing.\n"
    'Respond with ONLY one of:\n'
    '  {"match": true}\n'
    '  {"match": false}\n'
    '  {"match": "unsure"}'
)

TRUE_TOKENS = {"true", "true}"}
FALSE_TOKENS = {"false", "false}"}
UNSURE_TOKENS = {"unsure", '"unsure"', 'unsure"', '"unsure', "unsure}"}


def make_messages(query_description: str, document_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Condition: {query_description}\n\nDocument:\n{document_text}"
        )},
    ]


def extract_3way_confidence(client: OpenAI, model_path: str,
                            query_desc: str, doc_text: str) -> dict:
    messages = make_messages(query_desc, doc_text)

    try:
        resp = client.chat.completions.create(
            model=model_path,
            messages=messages,
            temperature=0.0,
            max_tokens=20,
            logprobs=True,
            top_logprobs=20,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""

        # Parse prediction
        try:
            result = json.loads(raw.strip())
            match_val = result.get("match", False)
            if isinstance(match_val, str) and match_val.lower() == "unsure":
                prediction = "unsure"
            else:
                prediction = "Yes" if bool(match_val) else "No"
        except (json.JSONDecodeError, AttributeError):
            lower = raw.lower()
            if "unsure" in lower:
                prediction = "unsure"
            elif "true" in lower and "false" not in lower:
                prediction = "Yes"
            elif "false" in lower:
                prediction = "No"
            else:
                prediction = "PARSE_FAIL"

        # Extract logprobs
        raw_lp_true = -100.0
        raw_lp_false = -100.0
        raw_lp_unsure = -100.0

        logprobs_data = resp.choices[0].logprobs
        if logprobs_data and logprobs_data.content:
            for token_info in logprobs_data.content:
                tok_lower = token_info.token.strip().lower()

                is_decision = (
                    tok_lower in TRUE_TOKENS or tok_lower in FALSE_TOKENS or
                    tok_lower in UNSURE_TOKENS or tok_lower == '"'
                )
                if not is_decision:
                    continue

                if tok_lower in TRUE_TOKENS:
                    raw_lp_true = token_info.logprob
                elif tok_lower in FALSE_TOKENS:
                    raw_lp_false = token_info.logprob
                elif tok_lower in UNSURE_TOKENS:
                    raw_lp_unsure = token_info.logprob

                if token_info.top_logprobs:
                    for alt in token_info.top_logprobs:
                        alt_lower = alt.token.strip().lower()
                        if alt_lower in TRUE_TOKENS:
                            raw_lp_true = max(raw_lp_true, alt.logprob)
                        elif alt_lower in FALSE_TOKENS:
                            raw_lp_false = max(raw_lp_false, alt.logprob)
                        elif alt_lower in UNSURE_TOKENS:
                            raw_lp_unsure = max(raw_lp_unsure, alt.logprob)
                break  # only first decision token

        # 3-way softmax
        lps = [raw_lp_true, raw_lp_false, raw_lp_unsure]
        active = [lp for lp in lps if lp > -100.0]
        if len(active) >= 2:
            max_lp = max(lps)
            exps = [math.exp(lp - max_lp) if lp > -100.0 else 0.0 for lp in lps]
            total = sum(exps)
            p_true, p_false, p_unsure = exps[0] / total, exps[1] / total, exps[2] / total
        elif len(active) == 1:
            p_true = 1.0 if raw_lp_true > -100.0 else 0.0
            p_false = 1.0 if raw_lp_false > -100.0 else 0.0
            p_unsure = 1.0 if raw_lp_unsure > -100.0 else 0.0
        else:
            p_true = p_false = 1 / 3
            p_unsure = 1 / 3

        # Confidence of chosen prediction
        if prediction == "Yes":
            confidence = p_true
        elif prediction == "No":
            confidence = p_false
        elif prediction == "unsure":
            confidence = p_unsure
        else:
            confidence = max(p_true, p_false, p_unsure)

        # 3-way entropy
        entropy = 0.0
        for p in [p_true, p_false, p_unsure]:
            if p > 1e-10:
                entropy -= p * math.log2(p)

        return {
            "prediction": prediction,
            "raw_response": raw.strip()[:200],
            "p_true": p_true,
            "p_false": p_false,
            "p_unsure": p_unsure,
            "confidence": confidence,
            "entropy": entropy,
            "logprob_true": raw_lp_true,
            "logprob_false": raw_lp_false,
            "logprob_unsure": raw_lp_unsure,
        }

    except Exception as e:
        return {
            "prediction": "PARSE_FAIL",
            "raw_response": f"ERROR: {e}"[:200],
            "p_true": 1/3, "p_false": 1/3, "p_unsure": 1/3,
            "confidence": 1/3, "entropy": math.log2(3),
            "logprob_true": -100.0, "logprob_false": -100.0, "logprob_unsure": -100.0,
        }


def load_data():
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
                ground_truth[(doc_id, qname)] = str(row[qname]).strip()
    return docs, queries, ground_truth


def load_checkpoint(output_file: Path) -> set[tuple[int, str]]:
    done = set()
    if output_file.exists():
        with open(output_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((int(row["document_id"]), row["query_name"]))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    docs, queries, ground_truth = load_data()
    print(f"Loaded {len(docs)} documents, {len(queries)} queries")

    client = OpenAI(base_url=BASE_URL, api_key="dummy")
    try:
        client.chat.completions.create(
            model=MODEL_PATH,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        print(f"[OK] Connected to Qwen3.5-0.8B at {BASE_URL}")
    except Exception as e:
        print(f"[ERROR] Cannot connect: {e}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "Qwen3.5-0.8B_confidence_3way.csv"

    doc_ids = sorted(docs.keys())
    if args.max_docs:
        doc_ids = doc_ids[:args.max_docs]
    query_names = sorted(queries.keys())

    tasks = [(doc_id, qname) for doc_id in doc_ids for qname in query_names]

    done = set()
    if args.resume:
        done = load_checkpoint(output_file)
        print(f"Resuming: {len(done)} completed, {len(tasks) - len(done)} remaining")
    tasks = [(d, q) for d, q in tasks if (d, q) not in done]

    if not tasks:
        print("All done!")
        return

    total = len(tasks)
    print(f"Total predictions: {total}")
    print(f"Output: {output_file}")

    write_header = not output_file.exists() or not args.resume
    f_out = open(output_file, "a" if args.resume and output_file.exists() else "w", newline="")
    fieldnames = [
        "document_id", "document_name", "query_name",
        "ground_truth", "prediction", "correct",
        "p_true", "p_false", "p_unsure",
        "confidence", "entropy",
        "logprob_true", "logprob_false", "logprob_unsure",
        "raw_response",
    ]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    completed = 0
    correct_count = 0
    unsure_count = 0
    t_start = time.time()

    def predict_one(doc_id: int, qname: str) -> dict:
        result = extract_3way_confidence(client, MODEL_PATH, queries[qname], docs[doc_id]["text"])
        gt = ground_truth.get((doc_id, qname), "N/A")
        pred_mapped = result["prediction"]
        if pred_mapped == "unsure":
            is_correct = False
        elif pred_mapped == "PARSE_FAIL":
            is_correct = False
        else:
            is_correct = pred_mapped == gt

        return {
            "document_id": doc_id,
            "document_name": docs[doc_id]["name"],
            "query_name": qname,
            "ground_truth": gt,
            "prediction": result["prediction"],
            "correct": is_correct,
            "p_true": f"{result['p_true']:.6f}",
            "p_false": f"{result['p_false']:.6f}",
            "p_unsure": f"{result['p_unsure']:.6f}",
            "confidence": f"{result['confidence']:.6f}",
            "entropy": f"{result['entropy']:.6f}",
            "logprob_true": f"{result['logprob_true']:.5f}",
            "logprob_false": f"{result['logprob_false']:.5f}",
            "logprob_unsure": f"{result['logprob_unsure']:.5f}",
            "raw_response": result["raw_response"],
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
                correct_count += 1
            if result["prediction"] == "unsure":
                unsure_count += 1

            if completed % 200 == 0 or completed == total:
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                acc = correct_count / completed
                uns = unsure_count / completed
                eta = (total - completed) / rate if rate > 0 else 0
                print(
                    f"  [{completed:>6}/{total}] "
                    f"Acc={acc:.4f} Unsure={uns:.2%} "
                    f"Rate={rate:.1f}/s ETA={eta/60:.1f}min"
                )

    f_out.close()

    elapsed = time.time() - t_start
    acc = correct_count / completed
    uns = unsure_count / completed
    print(f"\nDONE: {completed} predictions, Accuracy={acc:.4f}, Unsure={uns:.2%}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
