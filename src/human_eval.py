"""
Human Evaluation Agreement Calculator for Vision Transformer SAE features.

Usage:
    python src/human_eval.py <path_to_completed_human_eval_csv>

Instructions:
1. Export the template with `python src/main.py --export_human_eval` (outputs to `src/out/human_eval/`).
2. Two independent human raters inspect the per-feature exemplar grids in the directory
   and manually fill in the 'Rater1 Label' and 'Rater2 Label' columns in the CSV template.
3. Run this script pointing to the completed CSV to compute agreement metrics.
"""
import sys
import csv

def compute_human_eval_agreement(csv_path: str):
    total = 0
    rater_agreement = 0
    rater1_clip_agreement = 0
    rater2_clip_agreement = 0

    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip and lowercase to be case-insensitive and robust
                clip_concept = (row.get("CLIP Concept") or "").strip().lower()
                r1_label = (row.get("Rater1 Label") or "").strip().lower()
                r2_label = (row.get("Rater2 Label") or "").strip().lower()

                if not r1_label or not r2_label:
                    continue  # skip unfilled rows

                total += 1
                if r1_label == r2_label:
                    rater_agreement += 1
                if r1_label == clip_concept:
                    rater1_clip_agreement += 1
                if r2_label == clip_concept:
                    rater2_clip_agreement += 1
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if total == 0:
        print("No completed rater entries found in the CSV. Please ensure both Rater1 and Rater2 columns are filled.")
        return

    print("=" * 60)
    print("                 HUMAN EVALUATION AGREEMENT SUMMARY")
    print("=" * 60)
    print(f"Total completed features: {total}")
    print(f"Rater 1 vs Rater 2 agreement: {rater_agreement / total * 100:.2f}% ({rater_agreement}/{total})")
    print(f"Rater 1 vs CLIP agreement:    {rater1_clip_agreement / total * 100:.2f}% ({rater1_clip_agreement}/{total})")
    print(f"Rater 2 vs CLIP agreement:    {rater2_clip_agreement / total * 100:.2f}% ({rater2_clip_agreement}/{total})")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python human_eval.py <csv_path>")
        sys.exit(1)
    compute_human_eval_agreement(sys.argv[1])
