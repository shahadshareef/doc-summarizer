import os
import csv
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_dataset(judgement_folder, summary_folder, output_csv="evaluation_results.csv"):
    # Make sure folders exist
    if not os.path.exists(judgement_folder) or not os.path.exists(summary_folder):
        print("Error: One of the dataset folders does not exist.")
        return
    
    judgement_files = sorted(os.listdir(judgement_folder))
    summary_files = sorted(os.listdir(summary_folder))

    results = []

    for j_file, s_file in zip(judgement_files, summary_files):
        with open(os.path.join(judgement_folder, j_file), "r", encoding="utf-8") as f:
            reference_text = f.read()
        with open(os.path.join(summary_folder, s_file), "r", encoding="utf-8") as f:
            system_summary = f.read()

        # Tokenize
        ref_sentences = sent_tokenize(reference_text)
        sys_sentences = sent_tokenize(system_summary)

        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference_text, system_summary)

        # BLEU
        smoothie = SmoothingFunction().method1
        bleu = sentence_bleu([ref_sentences], sys_sentences, smoothing_function=smoothie)

        results.append({
            "file": j_file,
            "ROUGE-1": rouge_scores['rouge1'].fmeasure,
            "ROUGE-L": rouge_scores['rougeL'].fmeasure,
            "BLEU": bleu
        })

    # Save to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "ROUGE-1", "ROUGE-L", "BLEU"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Evaluation complete. Results saved to {output_csv}")

    # --- Visualization ---
    files = [r["file"] for r in results]
    rouge1 = [r["ROUGE-1"] for r in results]
    rougel = [r["ROUGE-L"] for r in results]
    bleu = [r["BLEU"] for r in results]

    plt.figure(figsize=(12, 6))
    x = range(len(files))

    plt.bar(x, rouge1, width=0.25, label="ROUGE-1", align="center")
    plt.bar([i + 0.25 for i in x], rougel, width=0.25, label="ROUGE-L", align="center")
    plt.bar([i + 0.5 for i in x], bleu, width=0.25, label="BLEU", align="center")

    plt.xticks([i + 0.25 for i in x], files, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics per Document")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_chart.png")

    print("📊 Chart saved as evaluation_chart.png")

if __name__ == "__main__":
    evaluate_dataset("dataset/judgement", "dataset/summary")
