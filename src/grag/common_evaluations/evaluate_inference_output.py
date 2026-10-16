import json
import logging
import os
from argparse import ArgumentParser
from datetime import datetime

import evaluate
import numpy as np
from rouge_score import rouge_scorer
from torchmetrics.text import CHRFScore
from tqdm import tqdm

from grag.common_evaluations.metrics import calculate_bert_scores
from grag.common_evaluations.metrics import calculate_new_copy_metrics
from grag.common_evaluations.utils import save_individual_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    parser = ArgumentParser()
    parser.add_argument("--inference_output_file", type=str, required=True,
                        help="Path to the inference output JSON file.")
    parser.add_argument("--metrics_output_file", type=str, default=None,
                        help="Path to save individual metrics.")

    args = parser.parse_args()

    if not os.path.exists(args.inference_output_file):
        logger.error(f"Inference output file not found: {args.inference_output_file}")
        return

    with open(args.inference_output_file, 'r') as f:
        inference_data = json.load(f)

    # Initialize metrics
    rouge_metric = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Generate unique experiment ID for sacrebleu to avoid collisions
    experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    bleu_metric = evaluate.load("sacrebleu", experiment_id=experiment_id)
    chrf_metric = CHRFScore()

    r1, r2, rl = 0, 0, 0
    bleu = 0
    chrf1 = 0
    bert_pr, bert_rc, bert_f1 = 0, 0, 0

    copy_rate_generic_pred_total = 0
    copy_rate_ref_pred_total = 0
    ecr_hat_total = 0
    copy_rate_generic_ref_total = 0
    f1_type_a_total = 0
    f1_type_b_total = 0

    pred_len_total = 0
    gold_len_total = 0
    gen_len_total = 0

    copy_rate_generic_ref_list = []
    individual_metrics = []

    logger.info(f"Evaluating {len(inference_data)} samples...")

    for i, item in enumerate(tqdm(inference_data)):
        gold_reply = item["gold_reply"]
        pred_reply = item["pred_reply"]
        generic_response = item["generic_response"]

        # In evaluate_test.py, generic_response might be a list or a string depending on how it was logged
        if isinstance(generic_response, (list, tuple)):
            generic_response = generic_response[0]

        # ROUGE
        r = rouge_metric.score(pred_reply, gold_reply)
        r1 += r['rouge1'].fmeasure
        r2 += r['rouge2'].fmeasure
        rl += r['rougeL'].fmeasure

        # BLEU
        bleu_score = bleu_metric.compute(predictions=[pred_reply], references=[[gold_reply]])['score']
        bleu += bleu_score

        # CharF1
        chrf_val = chrf_metric([pred_reply], [[gold_reply]]).item()
        chrf1 += chrf_val

        # BERT scores
        pr, rc, f = calculate_bert_scores(gold_reply, pred_reply)
        bert_pr += pr
        bert_rc += rc
        bert_f1 += f

        # Copy rate analysis
        copy_metrics = calculate_new_copy_metrics(pred_reply, gold_reply, generic_response, rouge_metric)

        copy_rate_generic_pred_total += copy_metrics['copy_rate_generic_pred']
        copy_rate_ref_pred_total += copy_metrics['copy_rate_ref_pred']
        ecr_hat_total += copy_metrics['ecr_hat']
        copy_rate_generic_ref_total += copy_metrics['copy_rate_generic_ref']
        f1_type_a_total += copy_metrics['f1_type_a']
        f1_type_b_total += copy_metrics['f1_type_b']

        pred_len_total += copy_metrics['pred_len']
        gold_len_total += copy_metrics['gold_len']
        gen_len_total += copy_metrics['gen_len']

        copy_rate_generic_ref_list.append(copy_metrics['copy_rate_generic_ref'])

        # Individual metrics
        individual_metrics.append({
            "index": i,
            "rouge1": r['rouge1'].fmeasure,
            "rouge2": r['rouge2'].fmeasure,
            "rougeL": r['rougeL'].fmeasure,
            "bleu": bleu_score,
            "chrf": chrf_val,
            "bert_pr": pr,
            "bert_rc": rc,
            "bert_f1": f,
            "copy_rate_generic_pred": copy_metrics['copy_rate_generic_pred'],
            "copy_rate_ref_pred": copy_metrics['copy_rate_ref_pred'],
            "ecr_hat": copy_metrics['ecr_hat'],
            "copy_rate_generic_ref": copy_metrics['copy_rate_generic_ref'],
            "f1_type_a": copy_metrics['f1_type_a'],
            "f1_type_b": copy_metrics['f1_type_b'],
            "pred_len": copy_metrics['pred_len'],
            "gold_len": copy_metrics['gold_len'],
            "gen_len": copy_metrics['gen_len']
        })

    # Save individual metrics
    save_individual_metrics(args, individual_metrics)

    # Calculate final metrics
    num_samples = len(inference_data)
    if num_samples > 0:
        rouge1_result = r1 / num_samples
        rouge2_result = r2 / num_samples
        rougel_result = rl / num_samples
        bleu_result = bleu / num_samples
        chrf1_result = chrf1 / num_samples
        bert_pr_result = bert_pr / num_samples
        bert_rc_result = bert_rc / num_samples
        bert_f1_result = bert_f1 / num_samples

        copy_rate_generic_pred_result = copy_rate_generic_pred_total / num_samples
        copy_rate_ref_pred_result = copy_rate_ref_pred_total / num_samples
        ecr_hat_result = ecr_hat_total / num_samples
        copy_rate_generic_ref_result = copy_rate_generic_ref_total / num_samples
        f1_type_a_result = f1_type_a_total / num_samples
        f1_type_b_result = f1_type_b_total / num_samples

        pred_len_result = pred_len_total / num_samples
        gold_len_result = gold_len_total / num_samples
        gen_len_result = gen_len_total / num_samples

        print("\n" + "=" * 30)
        print("AGGREGATED EVALUATION RESULTS")
        print("=" * 30)
        print(f"ROUGE-1: {rouge1_result:.4f}")
        print(f"ROUGE-2: {rouge2_result:.4f}")
        print(f"ROUGE-L: {rougel_result:.4f}")
        print(f"BLEU: {bleu_result:.4f}")
        print(f"CharF1: {chrf1_result:.4f}")
        print(f"BERT Precision: {bert_pr_result:.4f}")
        print(f"BERT Recall: {bert_rc_result:.4f}")
        print(f"BERT F1: {bert_f1_result:.4f}")
        print("-" * 30)
        print("Copy-Rate Analysis (ROUGE-2 Precision based)")
        print(f"Copy Rate Generic Pred (Model vs Generic): {copy_rate_generic_pred_result:.4f}")
        print(f"Copy Rate Ref Pred (Model vs Reference): {copy_rate_ref_pred_result:.4f}")
        print(f"NewECR(hat_R) (P_RG - P_RP): {ecr_hat_result:.4f}")
        print(f"Copy Rate Generic Ref (Reference vs Generic): {copy_rate_generic_ref_result:.4f}")
        if copy_rate_generic_ref_list:
            quartiles = np.percentile(copy_rate_generic_ref_list, [25, 50, 75])
            print(
                f"Copy Rate Generic Ref Quartiles: Q1={quartiles[0]:.4f}, Q2={quartiles[1]:.4f}, Q3={quartiles[2]:.4f}")
        print("-" * 30)
        print("ROUGE-2 F1 Decomposition")
        print(f"Type A F1 (Shared): {f1_type_a_result:.4f}")
        print(f"Type B F1 (Novel): {f1_type_b_result:.4f}")
        print(f"Total ROUGE-2 F1: {rouge2_result:.4f}")
        print("-" * 30)
        print("Mean Token Lengths")
        print(f"Prediction Length: {pred_len_result:.2f}")
        print(f"Reference Length: {gold_len_result:.2f}")
        print(f"Generic Length: {gen_len_result:.2f}")
        print("=" * 30 + "\n")
    else:
        logger.warning("No samples were evaluated.")


if __name__ == "__main__":
    run()
