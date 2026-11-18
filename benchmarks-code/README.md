# Sequence Benchmarking Tools (Seq Benchmarks)

A comprehensive framework for evaluating language models on bioinformatics sequence tasks.

## 📋 Project Overview

This project provides an end-to-end benchmarking platform for transformer-based language models on biological sequence analysis. It supports classification and regression tasks involving DNA, RNA, proteins, and other biological sequences.

## ✨ Key Features

- 🧬 **Broad dataset coverage**: More than 50 bioinformatics datasets
- 🔬 **Multiple task types**: Classification, regression, and multilabel prediction
- 🚀 **Efficient parallel processing**: Multi-GPU computation and task scheduling
- 📊 **End-to-end evaluation**: A complete workflow from embedding extraction to result analysis
- 📈 **Visual reports**: Automatic generation of performance reports and figures
- 🔧 **Flexible configuration**: Experiment settings can be customized through YAML files


## 📋 Supported Datasets

| Dataset | Task | Samples | Sequence Length | Classes / Outputs | Splits | Biological Significance |
|-----------|---------|--------|----------|--------|----------|------------|
| **Genomic Benchmark** | | | | | | |
| `demo_coding_vs_intergenomic_seqs` | Classification | 100,000 | 200 | 2 | train/test | Distinguishes coding sequences from intergenic regions and assesses recognition of functional genomic regions |
| `human_enhancers_cohn` | Classification | 27,791 | 500 | 2 | train/test | Identifies human enhancers and evaluates prediction of cis-regulatory elements |
| `human_ocr_ensembl` | Classification | 174,756 | 71–593 | 2 | train/test | Identifies open chromatin regions and predicts chromatin accessibility and transcriptionally active regions |
| **Nucleotide Transformer** | | | | | | |
| `splice_sites_all` | Classification | 30,000 | 400 | 3 | train/test | Predicts donor, acceptor, and non-splice sites to assess recognition of RNA splicing signals |
| `H3` | Classification | 14,965 | 290–500 | 2 | train/test | Predicts histone H3 modifications and assesses recognition of epigenetic marks |
| `H3K36me3` | Classification | 34,880 | 310–500 | 2 | train/test | Predicts H3K36me3 trimethylation and assesses recognition of a mark associated with transcriptional elongation |
| **Long Range Benchmark (LRB)** | | | | | | |
| `regulatory_element_promoter_8K` | Classification | 60,000 | 8,192 | 2 | train/test/eval | Identifies promoters in long sequences and evaluates prediction of transcription initiation regions |
| `regulatory_element_enhancer_8K` | Classification | 60,000 | 8,192 | 2 | train/test/eval | Identifies enhancers in long sequences and evaluates prediction of distal regulatory elements |
| `variant_effect_causal_eqtl_8K` | Classification | 38,862 | 8,192 | 2 | train/test/eval | Predicts causal eQTL variants from paired sequences and assesses recognition of variants that regulate gene expression |
| `variant_effect_pathogenic_clinvar_8K` | Classification | 31,018 | 8,192 | 2 | train/test/eval | Predicts pathogenic variants from paired sequences and assesses recognition of clinically relevant deleterious variants |
| `bulk_rna_expression_8K` | Regression | 23,817 | 8,192 | 218 outputs | train/test/eval | Predicts bulk RNA expression and evaluates quantitative gene-expression prediction |
| **Variant Hotspot** | | | | | | |
| `CPC_1024` | Classification | 828,039 | 1,024 | 2 | train/test/eval | Predicts coding potential in short sequences |
| `CPC_8192` | Classification | 62,729 | 8,192 | 2 | train/test/eval | Predicts coding potential in medium-length sequences and evaluates coding potential in long transcripts |
| `CPC_32768` | Classification | 15,391 | 32,768 | 2 | train/test/eval | Predicts coding potential in long sequences and evaluates recognition of complex transcripts |
| `CPC_131072` | Classification | 3,825 | 131,072 | 2 | train/test/eval | Predicts coding potential in very long sequences and evaluates coding analysis of full-length transcripts |
| `HPRC_1024` | Classification | 246,790 | 1,024 | 2 | train/test/eval | Assesses short-sequence variation in the human pangenome and population-level genetic variation patterns |
| `HPRC_8192` | Classification | 30,550 | 8,192 | 2 | train/test/eval | Assesses human pangenome variation in medium-length sequences and evaluates structural variant recognition |
| `HPRC_32768` | Classification | 7,375 | 32,768 | 2 | train/test/eval | Assesses human pangenome variation in long sequences and evaluates large-scale structural variants |
| `HPRC_131072` | Classification | 1,926 | 131,072 | 2 | train/test/eval | Assesses human pangenome variation in very long sequences and evaluates complex structural rearrangements |
| **Downstream Tasks** | | | | | | |
| `Human_classify_8192` | Classification | 28,994 | 8,192 | 3 | train/test/eval | Classifies European, African, and East Asian population groups from genomic sequences |
| `Human_classify_32768` | Classification | 29,045 | 32,768 | 3 | train/test/eval | Classifies European, African, and East Asian population groups from genomic sequences |
| `Human_classify_131072` | Classification | 29,410 | 131,072 | 3 | train/test/eval | Classifies European, African, and East Asian population groups from genomic sequences |
| `Human_classify_1048576` | Classification | 28,802 | 1,048,576 | 3 | train/test/eval | Classifies European, African, and East Asian population groups from genomic sequences |


## 🚀 Quick Start

### Environment Setup

**Install dependencies:**

```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Configure and run the benchmark:**
   ```bash
   python benchmarks.py --config config.yaml
   ```

2. **Configure `config.yaml`:**

- Model settings
   ```yaml
   # Path to the model used for embedding extraction. The model name defaults to the final directory name in this path.
   model_path: "/model/mode_path"
   # Some model classes may not be supported for embedding extraction by this framework. If the model cannot be loaded or its embeddings cannot be extracted, extract embeddings separately, remove model_path, and set model_name and num_layers. See the relevant section below for details.
   # If the model and its number of layers are detected correctly, remove model_name and num_layers.
   # Model name
   model_name: "model_name"
   # Number of model layers
   num_layers: 12
   ```
- Dataset settings
   ```yaml
   # Dataset directory
   dataset_path: "/data/dataset_path"
   # Dataset metadata file
   datasets_feature_path: "benchmarks/datasets_info.yaml"
   # Datasets to evaluate. For custom datasets, see the instructions below.
   eval_datasets:
     - demo_coding_vs_intergenomic_seqs
     - human_enhancers_cohn
     - splice_sites_all
   ```
- Task scheduling settings
    ```yaml
    # GPU indices to use. If omitted, all detected GPUs are used.
    gpu_list: [0,1]
    # Model layers to evaluate. If omitted, all layers are evaluated.
    layer_to_eval: [12]
    # Task allocation settings. See the scheduling section below for details.
    # Total task weight allowed on each GPU
    process_power_per_gpu: 5
    # Weight assigned to each embedding extraction task
    embedding_process_power: 4
    # Weight assigned to each evaluation task
    eval_process_power: 1
    # Number of samples per embedding extraction split
    embedding_extract_split: 50000
   ```
- Embedding extraction settings
    ```yaml
    # Directory for embeddings. Embeddings can require substantial storage; remove them after use when appropriate.
    embedding_output_dir: "/results/embeddings"
    # Batch size for embedding extraction
    batch_size: 8
   ```
- Classification and regression settings after embedding extraction

    ```yaml
    # Default classifier. Supported options include MLP, XGB (XGBoost), and RF (random forest). A dataset-specific classifier in datasets_info.yaml takes precedence.
    classifer_type: "MLP"
    # MLP settings
    mlp_dropout: 0.2
    mlp_lr: 0.0001
    mlp_epochs: 100
    save_last_epoch_model: False
    # Random forest settings
    rf_n_estimators: 100
    # XGBoost settings
    xgb_n_estimators: 100
    xgb_eval_metric: 'mlogloss'
    xgb_learning_rate: 0.1
    xgb_max_depth: 6
    # How to combine embeddings from multiple sequences in one sample (1: concatenate, e.g. 2x1024 -> 1x2048; 0: mean pooling, e.g. nx1024 -> 1024)
    pooled_embeddings_cat_dim: 1
   ```
- Report generation settings
    ```yaml
    # Output directory for evaluation results
    eval_result_path: "results_path" 
    # Classification or label metrics: ["accuracy", "roc_auc", "precision", "recall", "f1", "mcc"]
    # Regression metrics: ["r2", "pearsonr", "spearmanr"]
    # Select the best results for each dataset. With one metric, return all results from the layer with the best value. With multiple metrics, select the best value for each metric across layers. Generates a table.
    target_best_table:
    - "accuracy"
    - "roc_auc"
    - ["roc_auc", "accuracy"]
    - "pearsonr"

    # Plot metrics by layer. With one metric, plot that metric. With multiple metrics, the first is the primary metric; the others are reported in the legend at the layer with the best primary-metric value.
    target_layer_line_figure:
    - ["roc_auc", "accuracy"]
    - ["accuracy", "roc_auc"]
    - "pearsonr"
   ```
- Logging settings. Currently, W&B monitoring is supported only for the MLP workflow.
    ```yaml
    # Set wandb_report to False to limit the number of monitoring runs when evaluating many tasks.
    wandb_report: False
    wandb_project: "mlp"
    wandb_entity: "wandb_entity"
    wandb_online: False
   ```
- Other settings
    ```yaml
    seed: 42
   ```

## 🔧 Advanced Usage

### 1. Use Precomputed Embeddings

If the framework cannot load a model for any reason, extract embeddings separately, organize them in the required format, and update the configuration to continue using the framework.

#### 1.1 Update `config.yaml`

```yaml
# Comment out model_path and set model_name and num_layers.
# model_path: "/path/to/model"
# Model name, used mainly to locate result files
model_name: "your_model_name"
# Number of model layers, used to locate embedding data
num_layers: 12
```

#### 1.2 Embedding File Format

- File name: `model_name/datasetname-[x]layer_[train/eval/test].pt`
- Data format: `{'embeddings': pooled_embeddings, 'labels': all_labels}`


### 2. Custom Datasets

#### 2.1 Dataset Format

- File name: `model_name/[train/eval/test].jsonl`
- Each JSONL line should contain `{"seq": "ATCG", "label": 0}`. For samples with multiple sequences, use fields such as `{"seq1": "ATCG", "seq2": "ATCG", "label": 0}`. Classification labels should be integers, multilabel targets should be integer lists, and regression targets should be float lists.

#### 2.2 Add a Dataset Configuration to `benchmarks/datasets_info.yaml`

```yaml
  Human_classify_1048576:
    seq_for_item: 1   # Number of sequences in each sample
    seq_key: seq      # Sequence field; for multiple sequences, provide a list of keys such as ['seq1','seq2']
    label_key: label  # Label field in the dataset
    eval_task: classification  # Supported tasks: classification, multilabel, and regression
    data_split: ['train', 'test', 'eval']  # Dataset splits
    dataset_ratio:  [0.79, 0.09, 0.12]  # Split proportions, in the same order as data_split
    classifer_type: "XGB"   # Classifier for evaluation; defaults to classifer_type in config.yaml if omitted
    sample_num:  28802     # Number of samples
    min_length:  1048576   # Minimum sequence length
    max_length:  1048576   # Maximum sequence length
    dataset_ratio:  [0.79, 0.09, 0.12]  # Split proportions
    label_train_counter:  [(0, 4507), (1, 10582), (2, 7837)] # Number of samples per training label
    label_test_counter:  [(0, 392), (1, 1174), (2, 979)]     # Number of samples per test label
    label_eval_counter:  [(0, 784), (1, 1568), (2, 979)]     # Number of samples per evaluation label
```

### 3. Task Scheduling Logic

Task scheduling is controlled by four parameters in `config.yaml`: `process_power_per_gpu`, `embedding_process_power`, `eval_process_power`, and `embedding_extract_split`.

- `process_power_per_gpu`, `embedding_process_power`, and `eval_process_power` control task allocation and scheduling.
- `embedding_extract_split` controls how embedding extraction tasks are split.

#### 3.1 Task Allocation

- `process_power_per_gpu` is the task-weight capacity of each GPU.
- `embedding_process_power` is the weight assigned to each embedding extraction task.
- `eval_process_power` is the weight assigned to each downstream task after embeddings are available.

For example, suppose `process_power_per_gpu: 5`, `embedding_process_power: 2`, and `eval_process_power: 1`. If two GPUs must process three datasets (`dataset_A`, `dataset_B`, and `dataset_C`) and no embeddings have been precomputed, tasks are allocated as follows.

**Initial state:**
```
GPU-0: task load 0/5 [idle]
GPU-1: task load 0/5 [idle]
Embedding queue: [dataset_A, dataset_B, dataset_C]
Evaluation queue: []
```

**First allocation round** (embedding tasks have priority):
- GPU-0 receives the embedding extraction task for dataset_A, reaching a task load of 2/5.
- GPU-1 receives the embedding extraction task for dataset_B, reaching a task load of 2/5.
```
GPU-0: task load 2/5 [dataset_A embedding extraction]
GPU-1: task load 2/5 [dataset_B embedding extraction]
Embedding queue: [dataset_C]
Evaluation queue: []
```

**Second allocation round:**
- GPU-0 has capacity for three more task-weight units and can take another embedding task.
- GPU-0 receives the embedding extraction task for dataset_C, reaching a task load of 4/5.
```
GPU-0: task load 4/5 [dataset_A and dataset_C embedding extraction]
GPU-1: task load 2/5 [dataset_B embedding extraction]
Embedding queue: []
Evaluation queue: []
```

**Task completion and cascading allocation:**
- When embedding extraction for dataset_A finishes, GPU-0 releases two task-weight units and its load drops to 2/5.
- dataset_A is automatically added to the evaluation queue.
- GPU-0 takes the evaluation task for dataset_A, increasing its load to 3/5.
```
GPU-0: task load 3/5 [dataset_C embedding extraction, dataset_A evaluation]
GPU-1: task load 2/5 [dataset_B embedding extraction]
Embedding queue: []
Evaluation queue: []
```

**Final balanced state:**
As tasks finish, GPU resources remain in use:
```
GPU-0: task load 3/5 [dataset_B evaluation, dataset_C evaluation, other tasks]
GPU-1: task load 2/5 [new embedding tasks or multiple evaluation tasks]
```

This weighting scheme is designed to:
1. **Prevent overload**: The total task weight on each GPU stays within its capacity.
2. **Prioritize embedding extraction**: Embedding tasks, with a weight of 2, receive GPU resources first.
3. **Balance workloads**: New tasks are assigned to the GPU with the lowest load.
4. **Chain tasks**: Evaluation tasks are queued automatically when embedding extraction finishes.

#### 3.2 Splitting Embedding Tasks

To reduce memory use and improve parallel efficiency, large embedding extraction jobs are split into smaller tasks. The `embedding_extract_split` parameter controls the split size.

##### **Splitting Strategy**

- `embedding_extract_split: 50000` limits each subtask to 50,000 samples.
- Large datasets are automatically divided into smaller subtasks.
- Each subtask is assigned GPU resources independently to improve parallel throughput.

This strategy supports datasets of different sizes while promoting efficient resource use and stable execution.

---

**Note**: Ensure that sufficient compute and storage resources are available, particularly for large models and datasets. Evaluation results may vary slightly across platforms and environments.
