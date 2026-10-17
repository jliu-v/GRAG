# GRAG: Generic Response-Augmented Generation Framework

GRAG is a framework for Personalized Conversational Systems, focusing on augmenting generation with generic responses using prefusion and postfusion techniques.

## Project Structure

```text
.
├── .run/                   # IDE run configurations (XML)
├── src/
│   └── grag/
│       ├── common_evaluations/  # Metrics and evaluation scripts for inference output
│       ├── few-shot/            # Few-shot learning and in-context inference
│       ├── generic-response-generation/ # Scripts for generating generic convo partitions
│       ├── postfusion/          # Postfusion model training and inference
│       └── prefusion/           # Prefusion model training and inference
├── templates/              # Prompt templates for fewshot and generic responses
├── pyproject.toml          # Poetry dependencies and project metadata
└── README.md               # Project documentation
```

## Requirements

- **Python**: >= 3.10, < 3.13
- **Package Manager**: [Poetry](https://python-poetry.org/)
- **Version Management**: [pyenv](https://github.com/pyenv/pyenv) (recommended)

## Environment Setup

This project uses `pyenv` to manage Python versions and `poetry` for dependency management.

1. **Install Python version**:
   ```bash
   pyenv install 3.10.12  # Or any supported version
   pyenv local 3.10.12
   ```

2. **Install Poetry**:
   Follow the instructions at [Poetry Installation](https://python-poetry.org/docs/#installation).

3. **Install Dependencies**:
   ```bash
   poetry install
   ```

4. **Activate Virtual Environment**:
   ```bash
   poetry shell
   ```

## Environment Variables

Commonly used environment variables:
- `CUDA_VISIBLE_DEVICES`: Specify GPU devices (e.g., `0` or `0,1`).
- `TOKENIZERS_PARALLELISM`: Set to `false` to avoid warnings from the Hugging Face Tokenizers library.
- `PYTHONUNBUFFERED`: Set to `1` for real-time logging.

## Running the Code
 
*Note: Dataset cache files (e.g., `*.tar.gz` files in `./dataset_cache/`) are automatically generated when you first run postfusion or prefusion models. These caches can be reused across all subsequent postfusion and prefusion model runs to speed up data loading.*
 
### 0. Generate Generic Responses
 
Before running downstream tasks, you must generate generic responses for the conversational datasets. This step amends the original datasets by adding generic agent responses, saving the results to `amended_datasets`.
 
#### Original Dataset Sources
 
Users are responsible for providing the original datasets in the `input_datasets` directory. Download the original datasets from the following sources:
- **ConvAI2** [1]: Available via the [ParlAI website](https://parl.ai/projects/convai2/) or Hugging Face ([`convai-challenge/conv_ai_2`](https://huggingface.co/datasets/convai-challenge/conv_ai_2)).
- **EmpatheticDialogues** [2]: Available on [GitHub](https://github.com/facebookresearch/EmpatheticDialogues) or Hugging Face ([`empathetic_dialogues`](https://huggingface.co/datasets/empathetic_dialogues)).
- **PersuasionForGood** [3]: Available on [GitHub](https://github.com/ohyj1002/persuasionforgood).
 
*Note: The **SPC** (Synthetic Persona Chat) [4] dataset is automatically retrieved via Hugging Face dataset cards ([`google/Synthetic-Persona-Chat`](https://huggingface.co/datasets/google/Synthetic-Persona-Chat)).*
 
#### Dataset Structure
 
```text
.
├── input_datasets/
│   ├── convai2/                # Original ConvAI2 files
│   ├── empatheticdialogues/    # Original EmpatheticDialogues CSVs
│   └── persuasionforgood-master-data/ # Original PFG data
└── amended_datasets/           # Generated/Amended datasets (used by downstream steps)
    ├── convai2/
    ├── empathetic/
    ├── pfg/
    └── spc/
```
 
#### Example Command
 
Run the generation script for a specific task:
 
```bash
# Generate generic responses for SPC
python src/grag/generic-response-generation/generate_generic_convo.py \
    --task spc \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir ./amended_datasets/spc \
    --seed 42
```
 
*Note: You can use the `JOB_SPLIT` environment variable (set to `0` or `1`) to run the generation in parallel across different partitions.*
 
#### Merging Partitions (Optional)
 
If you ran the generation in partitions, merge them using:
 
```bash
python src/grag/generic-response-generation/merge_output_data_parititions.py \
    --partition-files ./amended_datasets/spc/train_0.json ./amended_datasets/spc/train_1.json \
    --output-file ./amended_datasets/spc/train.json \
    --sample-output-file ./amended_datasets/spc/sample_train.json
```
 
### 1. Fewshot Learning
 
Use the in-context inference script for few-shot learning experiments.
 
#### Inference
```bash
python src/grag/few-shot/incontext_inference.py \
    --task spc \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --seed 0 \
    --flag test \
    --n_shot 0 \
    --train_dataset_path ./amended_datasets/spc/sample_train.json \
    --test_dataset_path ./amended_datasets/spc/sample_valid.json \
    --output_file test_models/inference.json \
    --context_mode persona-tst
```
 
#### Calculate BPC/PPL
```bash
python src/grag/few-shot/incontext_inference_ppl.py \
    --task spc \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --seed 0 \
    --flag test \
    --n_shot 0 \
    --train_dataset_path ./amended_datasets/spc/sample_train.json \
    --test_dataset_path ./amended_datasets/spc/sample_valid.json \
    --metrics_output_file test_models/metrics.json \
    --context_mode persona-tst
```

### 2. Prefusion Models

#### Training
```bash
python src/grag/prefusion/train.py \
    --model_name GPT2 \
    --task spc \
    --train_dataset_path ./amended_datasets/spc/sample_train.json \
    --train_dataset_cache ./dataset_cache/spc_sample_train.tar.gz \
    --valid_dataset_path ./amended_datasets/spc/sample_validation.json \
    --valid_dataset_cache ./dataset_cache/spc_sample_valid.tar.gz \
    --n_epochs 5 \
    --model_dir test_models \
    --flag test \
    --train_batch_size 2 \
    --method GRAG
```

#### Inference
```bash
python src/grag/prefusion/inference.py \
    --model_name GPT2 \
    --task spc \
    --test_dataset_path ./amended_datasets/spc/sample_validation.json \
    --test_dataset_cache ./dataset_cache/spc_sample_valid.tar.gz \
    --model_checkpoint test_models/test \
    --method Hist \
    --inference_output_file test_models/inference.json \
    --metrics_output_file test_models/metrics.json
```

#### Calculate BPC/PPL
```bash
python src/grag/prefusion/evaluate_test_ppl.py \
    --model_name GPT2 \
    --task spc \
    --test_dataset_path ./amended_datasets/spc/sample_validation.json \
    --test_dataset_cache ./dataset_cache/spc_sample_valid.tar.gz \
    --model_checkpoint test_models/test \
    --method Hist
```

### 3. Postfusion Models
 
#### Training
```bash
python src/grag/postfusion/train.py \
    --model_name gpt2 \
    --method GRAG \
    --backward_weight 0.40 \
    --jepa_weight 0.01 \
    --task spc \
    --train_dataset_path ./amended_datasets/spc/sample_train.json \
    --train_dataset_cache ./dataset_cache/spc_sample_train.tar.gz \
    --valid_dataset_path ./amended_datasets/spc/sample_validation.json \
    --valid_dataset_cache ./dataset_cache/spc_sample_valid.tar.gz \
    --n_epochs 2 \
    --model_dir test_models \
    --flag test \
    --train_batch_size 4 \
    --seed 0
```
 
#### Inference
```bash
python src/grag/postfusion/inference.py \
    --model_name gpt2 \
    --task spc \
    --test_dataset_path ./amended_datasets/spc/sample_validation.json \
    --test_dataset_cache ./dataset_cache/spc_sample_valid.tar.gz \
    --model_checkpoint test_models/test \
    --method GRAG \
    --inference_output_file test_models/inference.json \
    --metrics_output_file test_models/metrics.json
```
 
#### Calculate BPC/PPL
```bash
python src/grag/postfusion/evaluate_test_ppl.py \
    --model_name gpt2 \
    --task spc \
    --test_dataset_path ./amended_datasets/spc/sample_validation.json \
    --test_dataset_cache ./dataset_cache/spc_sample_valid.tar.gz \
    --model_checkpoint test_models/test \
    --method GRAG
```

### 4. Common Evaluation

Evaluate the inference outputs using common metrics.

```bash
python src/grag/common_evaluations/evaluate_inference_output.py \
    --inference_output_file test_models/inference.json \
    --metrics_output_file test_models/metrics.json
```


## License

This project is intended for academic research purposes. The code is licensed under the **MIT License** (see [LICENSE](LICENSE) for details). 

The datasets (Synthetic Persona Chat, ConvAI2, PersuasionForGood and EmpatheticDialogues) used in this framework are subject to their own respective licenses. Users are responsible for complying with the original licenses and terms of use when downloading and using these datasets.


## References

[1] Emily Dinan, et al. "The second conversational intelligence challenge (convai2)." *arXiv preprint arXiv:2001.03467* (2020).

[2] Hannah Rashkin, et al. "Towards Empathetic Open-domain Conversation Models: A New Dataset and Step-by-Step Annotation Procedure." *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (2019).

[3] Xuewei Wang, et al. "Persuasion for good: Towards a personalized persuasive dialogue system for social good." *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (2019).

[4] Pegah Jandaghi, et al. "Faithful Persona-based Conversational Dataset Generation with Large Language Models." *arXiv preprint arXiv:2312.10007* (2023).
