import math
import re

import torch
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer


def get_bigrams(text):
    """Simple bigram extractor."""
    tokens = re.findall(r'\w+', text.lower())
    if len(tokens) < 2:
        return []
    return [tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1)]


def calculate_new_copy_metrics(pred_reply, gold_reply, generic_response, rouge_scorer):
    """
    Implements the new Copy-Rate Analysis metrics.
    """
    # Use rouge_scorer's tokenizer if available, otherwise fallback to simple regex
    if hasattr(rouge_scorer, '_tokenizer') and rouge_scorer._tokenizer:
        tokenizer = rouge_scorer._tokenizer
    else:
        from rouge_score import tokenizers
        tokenizer = tokenizers.DefaultTokenizer()

    pred_tokens = tokenizer.tokenize(pred_reply)
    gold_tokens = tokenizer.tokenize(gold_reply)
    gen_tokens = tokenizer.tokenize(generic_response)

    pred_len = len(pred_tokens)
    gold_len = len(gold_tokens)
    gen_len = len(gen_tokens)

    # Use internal _create_ngrams from rouge_scorer for consistency
    from rouge_score import rouge_scorer as rs
    pred_bigrams = rs._create_ngrams(pred_tokens, 2)
    gold_bigrams = rs._create_ngrams(gold_tokens, 2)
    gen_bigrams = rs._create_ngrams(gen_tokens, 2)

    # Use rouge_scorer.score for standard ROUGE-2 precision
    copy_rate_generic_pred = rouge_scorer.score(generic_response, pred_reply)['rouge2'].precision
    copy_rate_ref_pred = rouge_scorer.score(gold_reply, pred_reply)['rouge2'].precision
    copy_rate_generic_ref = rouge_scorer.score(generic_response, gold_reply)['rouge2'].precision

    ecr_hat = copy_rate_generic_pred - copy_rate_ref_pred

    type_a_matches = 0
    type_b_matches = 0

    for gram, count in pred_bigrams.items():
        match_in_gold = min(count, gold_bigrams[gram])
        if match_in_gold > 0:
            match_in_gen = min(match_in_gold, gen_bigrams[gram])
            type_a_matches += match_in_gen
            type_b_matches += (match_in_gold - match_in_gen)

    def calculate_f1(matches, hyp_len, ref_len):
        if hyp_len == 0 or ref_len == 0 or matches == 0:
            return 0.0
        p = matches / hyp_len
        r = matches / ref_len
        return (2 * p * r) / (p + r)

    pred_in_gen_count = sum(min(count, gen_bigrams[gram]) for gram, count in pred_bigrams.items())
    pred_not_in_gen_count = sum(pred_bigrams.values()) - pred_in_gen_count

    f1_type_a = calculate_f1(type_a_matches, pred_in_gen_count, sum(gold_bigrams.values()))
    f1_type_b = calculate_f1(type_b_matches, pred_not_in_gen_count, sum(gold_bigrams.values()))

    return {
        'copy_rate_generic_pred': copy_rate_generic_pred,
        'copy_rate_ref_pred': copy_rate_ref_pred,
        'ecr_hat': ecr_hat,
        'copy_rate_generic_ref': copy_rate_generic_ref,
        'f1_type_a': f1_type_a,
        'f1_type_b': f1_type_b,
        'pred_len': pred_len,
        'gold_len': gold_len,
        'gen_len': gen_len
    }


_bert_model = None
_bert_tokenizer = None


def load_bert_model():
    global _bert_model
    global _bert_tokenizer

    bnb_config = None
    model_name = 'google-bert/bert-base-uncased'
    _bert_model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16,
                                            quantization_config=bnb_config)
    _bert_tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_bert_token_embeddings(text):
    global _bert_model
    global _bert_tokenizer

    if _bert_model is None or _bert_tokenizer is None:
        load_bert_model()

    inputs = _bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(_bert_model.device)
    with torch.no_grad():
        outputs = _bert_model(**inputs)

    return outputs.last_hidden_state.squeeze(0), _bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])


def calculate_bert_scores(gold_response, pred_response):
    emb_gold, tokens_gold = get_bert_token_embeddings(gold_response)
    emb_pred, tokens_pred = get_bert_token_embeddings(pred_response)

    emb_gold_norm = F.normalize(emb_gold, p=2, dim=1)
    emb_pred_norm = F.normalize(emb_pred, p=2, dim=1)

    similarity_matrix = torch.mm(emb_gold_norm, emb_pred_norm.T)

    precision = similarity_matrix.max(dim=1).values.mean().item()
    recall = similarity_matrix.max(dim=0).values.mean().item()
    f = (2 * precision * recall) / (precision + recall)

    return precision, recall, f


def calculate_sample_bpc_bpb(logits, labels, tokenizer):
    """
    Calculate BPC and BPB for each sample in a batch.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)

    ignore_index = -100
    valid_mask = shift_labels != ignore_index

    safe_labels = shift_labels.clone()
    safe_labels[~valid_mask] = 0

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_log_probs = token_log_probs * valid_mask

    batch_size = logits.size(0)
    bpc_list = []
    bpb_list = []
    total_tokens = 0
    total_chars = 0
    total_bytes = 0

    for i in range(batch_size):
        sample_log_prob = token_log_probs[i].sum().item()
        sample_tokens = valid_mask[i].sum().item()

        if sample_tokens == 0:
            continue

        valid_token_mask = labels[i] != -100
        if tokenizer.pad_token_id is not None:
            valid_token_mask = torch.logical_and(labels[i] != tokenizer.pad_token_id, valid_token_mask)

        valid_token_ids = labels[i][valid_token_mask]
        text = tokenizer.decode(valid_token_ids, skip_special_tokens=True)

        num_chars = len(text)
        num_bytes = len(text.encode('utf-8'))

        if num_chars == 0:
            continue

        sample_log_prob_base2 = sample_log_prob / math.log(2)
        sample_bpc = -sample_log_prob_base2 / num_chars
        sample_bpb = -sample_log_prob_base2 / num_bytes

        bpc_list.append(sample_bpc)
        bpb_list.append(sample_bpb)

        total_tokens += sample_tokens
        total_chars += num_chars
        total_bytes += num_bytes

    stats = {
        'total_tokens': total_tokens,
        'total_chars': total_chars,
        'total_bytes': total_bytes
    }

    return bpc_list, bpb_list, stats


def print_ppl_evaluation_results(ppl_avg, bpc_list, bpb_list, total_tokens, total_chars, total_bytes):
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Perplexity (PPL): {sum(ppl_avg) / len(ppl_avg):.4f}")

    if bpc_list:
        print(f"Bits per Character (BPC): {sum(bpc_list) / len(bpc_list):.4f}")
    if bpb_list:
        print(f"Bits per Byte (BPB): {sum(bpb_list) / len(bpb_list):.4f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total characters: {total_chars}")
    print(f"Total bytes: {total_bytes}")
    print("=" * 60)
