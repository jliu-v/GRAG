import json
import logging
import os

logger = logging.getLogger(__name__)

_inference_outputs = []
_inference_outputs_written = False

def maybe_write_inference_outputs(args):
    global _inference_outputs
    global _inference_outputs_written

    output_file = getattr(args, 'inference_output_file', getattr(args, 'output_file', None))

    if not _inference_outputs_written and output_file is not None:
        inference_output_size = args.inference_output_size
        msg = f"Writing inference output to file {output_file}.\n"
        if inference_output_size is not None and len(_inference_outputs) < inference_output_size:
            msg += f"Inference output length is {len(_inference_outputs)} less than requested output size {inference_output_size}.\n"

        logger.info(msg)
        with open(output_file, 'w') as fp:
            json.dump(_inference_outputs, fp, indent=2)
        _inference_outputs_written = True

def log_inference_outputs(gold_reply, pred_reply, persona, history, generic_response, args):
    global _inference_outputs
    inference_output_size = args.inference_output_size

    if inference_output_size is None or len(_inference_outputs) < inference_output_size:
        outputdict = {
            "persona": persona,
            "history": history,
            "gold_reply": gold_reply,
            "pred_reply": pred_reply,
            "generic_response": generic_response
        }
        _inference_outputs.append(outputdict)

    if inference_output_size is not None and len(_inference_outputs) >= inference_output_size:
        maybe_write_inference_outputs(args)

def save_individual_metrics(args, individual_metrics):
    if args.metrics_output_file is not None:
        existing_metrics = []
        if os.path.exists(args.metrics_output_file):
            try:
                with open(args.metrics_output_file, 'r') as f:
                    existing_metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing metrics file {args.metrics_output_file}: {e}")

        if existing_metrics:
            # Create a map for quick lookup by index
            metrics_map = {m['index']: m for m in existing_metrics}
            for m in individual_metrics:
                idx = m['index']
                if idx in metrics_map:
                    metrics_map[idx].update(m)
                else:
                    metrics_map[idx] = m
            # Sort back by index and update individual_metrics
            individual_metrics = [metrics_map[idx] for idx in sorted(metrics_map.keys())]

        logger.info(f"Writing individual metrics to {args.metrics_output_file}")
        with open(args.metrics_output_file, 'w') as f:
            json.dump(individual_metrics, f, indent=2)
