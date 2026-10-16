import logging
import math
import os
import time
from argparse import ArgumentParser
from pprint import pformat

import torch
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import PiecewiseLinear
from ignite.handlers import ProgressBar
from ignite.handlers.tensorboard_logger import OptimizerParamsHandler
from ignite.handlers.tensorboard_logger import OutputHandler
from ignite.handlers.tensorboard_logger import TensorboardLogger
from ignite.metrics import Loss
from ignite.metrics import MetricsLambda
from ignite.metrics import RunningAverage
from torch.nn import Softmax
from torch.optim import AdamW
# from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import WEIGHTS_NAME

from data_utils import BART_MODEL_INPUTS
from data_utils import DatasetCreatorConvAI2
from data_utils import DatasetCreatorEmpathetic
from data_utils import DatasetCreatorPFG
from data_utils import DatasetCreatorSPC
from data_utils import GPT2_MODEL_INPUTS
from data_utils import add_special_tokens_
from grag.prefusion.dialogue_modules.model_utils import GPUUtilizationTracker
from grag.prefusion.dialogue_modules.model_utils import convert_to_lora_model
from grag.prefusion.dialogue_modules.model_utils import get_model_type
from grag.prefusion.dialogue_modules.model_utils import load_model_class
from grag.prefusion.dialogue_modules.model_utils import save_checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__file__)


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--task", type=str, required=True, choices=["pfg", "empathetic", "spc", "convai2"],
                        help="{pfg, empathetic, spc, convai2}")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="{gpt2, bart, llama3}")
    parser.add_argument("--llama3_model_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="pre-trained model path for llama3 models")
    parser.add_argument("--gpt2_model_path", type=str, default="gpt2",
                        help="pre-trained model path for decoder only models")  # gpt2-medium
    parser.add_argument("--bart_model_path", type=str, default="facebook/bart-base",
                        help="pre-trained model path for encoder-decoder models")  # facebook/bart-large
    parser.add_argument("--t5_model_path", type=str, default="google-t5/t5-base",
                        help="pre-trained model path for T5 models")  # google-t5/t5-base
    parser.add_argument("--qwen3_model_path", type=str, default="Qwen/Qwen3-0.6B",
                        help="pre-trained model path for Qwen3 models")  # Qwen/Qwen3-4B
    parser.add_argument("--train_dataset_path", type=str, default="./data/persuasionforgood/sample_train.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--train_dataset_cache", type=str, default='./cache/persuasionforgood_sample_train.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--valid_dataset_path", type=str, default="./data/persuasionforgood/sample_valid.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--valid_dataset_cache", type=str, default='./cache/persuasionforgood_sample_valid.tar.gz',
                        help="Path or url of the dataset.")
    parser.add_argument("--method", type=str, default="GRAG",
                        help="Method of model architecture and implementation. Options: ['GRAG', 'Hist', 'TST']")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--flag", type=str, default="", help="Assign the name of the folder")
    parser.add_argument("--model_dir", type=str, default="models", help="Parent folder for storing the models.")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--use_quantization", action='store_true',
                        help="Load model is 4-bit quantization to save GPU memory.")
    parser.add_argument("--jepa_weight", type=float, default=0.0,
                        help="Weight for JEPA loss. Only available for version 2.")
    parser.add_argument("--use_lora", action='store_true', help="Use LoRA training.")

    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    # Load dataset_creator
    if args.task == "pfg":
        dataset_creator_cls = DatasetCreatorPFG
    elif args.task == "empathetic":
        dataset_creator_cls = DatasetCreatorEmpathetic
    elif args.task == "spc":
        dataset_creator_cls = DatasetCreatorSPC
    elif args.task == "convai2":
        dataset_creator_cls = DatasetCreatorConvAI2
    else:
        raise ValueError(f"Task not recognized: {args.task}")

    model_name = args.model_name.lower()

    model_class = load_model_class(args)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=getattr(model_class, "_custom_modules", []),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if args.use_quantization and "cuda" in args.device else None

    extra_model_kwargs = {}
    model_path_maps = {
        "gpt2": args.gpt2_model_path,
        "bart": args.bart_model_path,
        "llama3": args.llama3_model_path,
        "t5": args.t5_model_path,
        "qwen3": args.qwen3_model_path
    }
    try:
        model_path = model_path_maps[model_name]
    except KeyError:
        raise NotImplementedError(f"model_name '{model_name}' not implemented.")
    model = model_class.from_pretrained(model_path, device_map="auto", quantization_config=bnb_config,
                                        torch_dtype=torch.bfloat16, **extra_model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    add_special_tokens_(model, tokenizer)

    if args.use_lora:
        model = convert_to_lora_model(model)

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Size: {total_params / 1e6:.2f}M parameters")
    logger.info(f"Trainable Parameters: {trainable_params_count / 1e6:.2f}M parameters")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)

    logger.info("Prepare datasets")

    dataset_creator = dataset_creator_cls(tokenizer=tokenizer,
                                          train_dataset_path=args.train_dataset_path,
                                          valid_dataset_path=args.valid_dataset_path,
                                          train_dataset_cache=args.train_dataset_cache,
                                          valid_dataset_cache=args.valid_dataset_cache)
    # dataset = dataset_creator.create_dataset()
    dataset_creator.create_data_loader(args, generation_mode=False, selected_partitions=["train", "valid"])
    dataloaders = dataset_creator.dataset_loader
    # train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    model_type = get_model_type(model_name)
    if model_type == "decoder-only":
        model_input_names = GPT2_MODEL_INPUTS
    elif model_type == "encoder-decoder":
        model_input_names = BART_MODEL_INPUTS
    else:
        raise ValueError(f"Model model_type is not supported: {model_type}.")

    train_loader, val_loader = dataloaders["train"], dataloaders["valid"]

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        model_inputs = dict(zip(model_input_names, batch))
        if args.method == "2":
            model_inputs.update({
                "jepa_weight": args.jepa_weight
            })

        try:
            output = model(training=True,
                           padding=tokenizer.pad_token_id,
                           tokenizer=tokenizer,
                           **model_inputs)
            lm_loss = output["lm_loss"]

            # 1. Catch NaN/Inf in the loss BEFORE backpropagation
            if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                print(f"WARNING: NaN/Inf Loss detected at iteration {engine.state.iteration}. Skipping batch.")
                # If we are mid-accumulation window, we must clear gradients to avoid corruption
                optimizer.zero_grad(set_to_none=True)
                return (0.0,)

            loss = lm_loss / args.gradient_accumulation_steps
            loss.backward()

            # 2. Check and apply optimizer step only at the accumulation boundary
            if engine.state.iteration % args.gradient_accumulation_steps == 0:
                # clip_grad_norm_ calculates the global norm across all parameters
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

                # 3. Catch NaN/Inf inside the fully accumulated gradient vector
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"WARNING: NaN/Inf Gradient Norm ({total_norm}) detected. Skipping optimizer step.")
                    optimizer.zero_grad(set_to_none=True)  # Purge the bad gradients completely [1]
                    return (0.0,)

                # Gradients are pristine; safe to update model weights [1]
                optimizer.step()
                optimizer.zero_grad()

            return (lm_loss.item(),)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: OOM at iteration {engine.state.iteration}. Resetting accumulation window.")
                optimizer.zero_grad(set_to_none=True)

                # Force Ignite to skip to the start of the next clean accumulation window
                steps_over = engine.state.iteration % args.gradient_accumulation_steps
                if steps_over != 0:
                    engine.state.iteration += (args.gradient_accumulation_steps - steps_over)

                # Force memory flush [1]
                if 'output' in locals(): del output
                if 'lm_loss' in locals(): del lm_loss
                if 'loss' in locals(): del loss
                torch.cuda.empty_cache()  # release memory [1]

                return (0.0,)
            else:
                raise e

    trainer = Engine(update)

    # Efficiency tracking
    epoch_times = []
    batch_times = []
    vram_samples = []

    @trainer.on(Events.EPOCH_STARTED)
    def start_epoch_timer(engine):
        engine.state.epoch_start_time = time.time()
        if gpu_tracker:
            gpu_tracker.resume()

    @trainer.on(Events.EPOCH_COMPLETED)
    def stop_epoch_timer(engine):
        if gpu_tracker:
            gpu_tracker.pause()
        # This will be called before the evaluation handler if attached first
        epoch_times.append(time.time() - engine.state.epoch_start_time)

    @trainer.on(Events.ITERATION_STARTED)
    def start_batch_timer(engine):
        engine.state.batch_start_time = time.time()

    @trainer.on(Events.ITERATION_COMPLETED)
    def stop_batch_timer(engine):
        batch_times.append(time.time() - engine.state.batch_start_time)
        if torch.cuda.is_available():
            vram_samples.append(torch.cuda.memory_reserved() / 1024 ** 2)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            model_inputs = dict(zip(model_input_names, batch))

            output = model(training=False,
                           padding=tokenizer.pad_token_id,
                           tokenizer=tokenizer,
                           **model_inputs)
            lm_labels, lm_logits = output["lm_labels"], output["lm_logits"]

            lm_logits_flat_shifted = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[:, 1:].contiguous().view(-1)

            softmax = Softmax(dim=-1)
            lm_pred = softmax(lm_logits_flat_shifted)
            lm_val, lm_idx = torch.topk(lm_pred, k=1, dim=-1)
            lm_idx = lm_idx.squeeze(-1)

            mask = (lm_labels_flat_shifted != -100)
            lm_labels_only = [lm_labels_flat_shifted[mask].tolist()]
            lm_idx_only = lm_idx[mask].tolist()

            return (lm_logits_flat_shifted, lm_idx_only), (lm_labels_flat_shifted, lm_labels_only)

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    # Note: These are attached AFTER stop_epoch_timer, so stop_epoch_timer runs first.
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Linearly decrease the learning rate from lr to half of lr
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.01 * args.lr)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # attach metrics
    metrics = create_metrics(trainer, evaluator)

    # setup progress bar
    set_up_progress_bar(trainer, evaluator)

    # setup tensorboard
    model_dir = args.model_dir
    model_dir_identifier = args.flag
    args.log_dir = os.path.join(model_dir, model_dir_identifier)
    tb_logger = set_up_tensorboard_logger(trainer, evaluator, optimizer, metrics, args)

    # create/save model check point
    checkpoint_handler = save_checkpoint(trainer, model, tokenizer, args)

    # Run the training
    gpu_tracker = None
    if "cuda" in args.device:
        gpu_idx = torch.cuda.current_device()
        gpu_tracker = GPUUtilizationTracker(gpu_idx)
        gpu_tracker.start()

    start = time.time()
    try:
        trainer.run(train_loader, max_epochs=args.n_epochs)
    finally:
        if gpu_tracker:
            gpu_tracker.stop()
            gpu_tracker.join()
    end = time.time()

    print("Total time needed:", end - start)
    if epoch_times:
        print(f"Average time per epoch: {sum(epoch_times) / len(epoch_times):.4f} s")
    if batch_times:
        print(f"Average time per batch: {(sum(batch_times) / len(batch_times)) * 1000:.4f} ms")
    if torch.cuda.is_available():
        print(f"Peak VRAM Usage: {torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB")
        if vram_samples:
            print(f"Average VRAM Usage: {sum(vram_samples) / len(vram_samples):.2f} MB")
    if gpu_tracker:
        print(f"Average GPU Utilization: {gpu_tracker.get_avg_utilization():.2f} %")
    print(f"Model Size: {total_params} parameters")
    print(f"Trainable Parameters: {trainable_params_count} parameters")

    # close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.n_epochs > 0:
        os.rename(os.path.join(args.log_dir, checkpoint_handler._saved[-1][1]),
                  os.path.join(args.log_dir, WEIGHTS_NAME))
        tb_logger.close()


def create_metrics(trainer, evaluator):
    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "lm_loss")

    metric_loss = Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0]))
    metric_avg_ppl = MetricsLambda(math.exp, metric_loss)
    metrics = {
        "lm_loss": metric_loss,
        "average_ppl": metric_avg_ppl
    }
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return metrics


def set_up_progress_bar(trainer, evaluator):
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, event_name=Events.ITERATION_COMPLETED(every=500), metric_names=["lm_loss"])
    evaluator.add_event_handler(Events.COMPLETED,
                                lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))


def set_up_tensorboard_logger(trainer, evaluator, optimizer, metrics, args):
    log_dir = args.log_dir
    tb_logger = TensorboardLogger(log_dir)

    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training",
                                                        metric_names=["lm_loss"]),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
    tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                     event_name=Events.ITERATION_COMPLETED(every=5000))

    return tb_logger


# def save_checkpoint(trainer, model, tokenizer, args):
#     log_dir = args.log_dir
#     checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', n_saved=3, require_empty=False)
#     trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
#         'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation
#
#     torch.save(args, log_dir + '/model_training_args.bin')
#     getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
#     tokenizer.save_pretrained(log_dir)
#     return checkpoint_handler


if __name__ == "__main__":
    main()
