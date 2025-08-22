import os
import pandas as pd
from rouge_score import rouge_scorer
import textstat

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def compute_metrics(original, summary):
    metrics = {}

    # Length-based
    orig_len = len(original.split())
    sum_len = len(summary.split())
    metrics["Original Length"] = orig_len
    metrics["Summary Length"] = sum_len
    metrics["Compression Ratio"] = round(sum_len / orig_len, 3) if orig_len > 0 else 0

    # Readability (summary only)
    metrics["Flesch Reading Ease"] = textstat.flesch_reading_ease(summary)
    metrics["Gunning Fog Index"] = textstat.gunning_fog(summary)

    # ROUGE (using judgment as reference, summary as system output)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(original, summary)
    for k, v in scores.items():
        metrics[f"{k.upper()} Precision"] = round(v.precision, 3)
        metrics[f"{k.upper()} Recall"] = round(v.recall, 3)
        metrics[f"{k.upper()} F1"] = round(v.fmeasure, 3)

    return metrics

def evaluate_dataset(judgement_folder, summary_folder, output_csv="summary_eval.csv"):
    rows = []
    judgement_files = sorted(os.listdir(judgement_folder))
    summary_files = sorted(os.listdir(summary_folder))

    # Match by filename (assuming same base name, different folder)
    for j_file, s_file in zip(judgement_files, summary_files):
        j_path = os.path.join(judgement_folder, j_file)
        s_path = os.path.join(summary_folder, s_file)

        if not os.path.isfile(j_path) or not os.path.isfile(s_path):
            continue

        original = load_text(j_path)
        summary = load_text(s_path)

        metrics = compute_metrics(original, summary)
        metrics["Document"] = os.path.splitext(j_file)[0]
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Done. Results saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    evaluate_dataset("./judgement", "./summary")
