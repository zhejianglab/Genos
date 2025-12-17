# RNA-seq data generation
## 1. Overview
This case is based on the [Genos-1.2B](https://huggingface.co/ZhejiangLab/Genos-1.2B) and employs task-specific fine-tuning to directly predict single-nucleotide–resolution RNA-seq expression profiles from DNA sequences, covering multiple cell types and tissues. The scientific significance of this case lies in establishing a direct mapping between genomic sequences and transcriptomic expression, providing an innovative tool for elucidating gene regulatory mechanisms and accelerating transcriptomics research.

## 2. Data Analysis
### Data Sources
The training data are sourced from publicly available databases, including:

 - [ENCODE (ENCODE Consortium, 2012)](https://www.nature.com/articles/nature11247)
 - [GTEx (Kim-Hellmuth et al., 2020)](https://www.science.org/doi/10.1126/science.aaz8528)

After data integration and harmonization, the training dataset comprises 667 metadata groups of base-resolution RNA-seq bigWig files, paired with the hg38 reference genome. Model training uses all genomic positions across chromosomes 1–22, where each genomic window is paired with its corresponding average RNA-seq expression profile. The current released models are fine-tuned using:

 - 4 human B lymphocyte samples
 - 13 natural killer (NK) cell samples

To mitigate inter-individual variability, expression values within each cell type are averaged across samples and subsequently normalized, resulting in one aggregated expression profile per cell type. The models can support inference and prediction for human B lymphocytes and NK cells across chromosomes 1–22.

### Input & Output Data
The task is formulated as a regression problem. The model takes partial sequences from the hg38 reference genome as input, using a 32 kb sliding window, and outputs average normalized RNA-seq signal values at single-nucleotide resolution, covering multiple cell types and both forward and reverse genomic strands. The core objective is to learn the complex mapping from genomic sequence to gene expression, enabling accurate prediction of continuous transcriptomic expression levels at base-pair precision.

## 3. Model Design
### Downstream Model Architecture
The downstream model is built upon the pre-trained [Genos-1.2B](https://huggingface.co/ZhejiangLab/Genos-1.2B) backbone, with the original output head replaced by a task-specific convolutional module. This module comprises three 1D convolutional layers with progressively reduced channel dimensions (1024 → 256 → 64 → 1), each followed by batch normalization, GELU activation, and dropout regularization (dropout = 0.1). The final output is scaled by a learnable weight parameter and passed through a Softplus activation, ensuring non-negative continuous predictions consistent with RNA-seq signal characteristics. This design enhances the model’s ability to capture local sequence patterns while leveraging the translation invariance of convolution to improve computational efficiency.

### Fine-tuning Strategy and Training Optimization
The model is trained using full-parameter fine-tuning, with mean squared error (MSE) as the regression loss function. To address the skewed distribution of RNA-seq signal values, the training process incorporates square-root smoothing and clipping as well as power transformations for numerical compression, with inverse transformations applied during inference to restore the original signal scale.

The optimization uses the Adafactor optimizer, combined with a cosine annealing learning rate scheduler and linear warm-up (warm-up steps set to 5% of total training steps). Training is conducted with a global batch size of 256 over 60 epochs, ensuring stable convergence while mitigating gradient fluctuations associated with long-sequence training.

### Genomic Sequence Processing and Context Modeling
To balance long-range dependency modeling with computational efficiency, the input sequence window is set to 32 kb with 16 kb overlap between adjacent windows, enabling comprehensive coverage of all genomic positions across chromosomes 1–22.

## 4. Evaluation 
### Metrics
The primary evaluation metric is the **log1p-transformed Pearson correlation coefficient**, which quantifies the consistency between predicted RNA-seq profiles and experimentally measured data across different genomic scopes. Specifically, the evaluation includes::

 - Whole genome: Base-resolution correlation across the entire genome.
 - Gene expression: Correlation at the gene level, based on gene expression matrices.

The Pearson correlation coefficient is transformed using ```log1p``` (i.e., ```log(1 + r)```) to better capture global prediction performance and stabilize evaluation across signals with varying magnitudes.

### Evaluation Results
The model achieves a correlation coefficient exceeding 0.9 between simulated multi-cell-type RNA-seq expression profiles and experimental sequencing results, demonstrating high fidelity in capturing transcriptomic patterns. Performance was evaluated at both single-base resolution and gene-level expression, across different cell types and strands, as summarized in the table below.

| Cell Type | Gene Strand | Single-base Accuracy | Gene Expression Accuracy |
|:-----------:|:-------------:|:------------------------------------------------------:|:----------------------------------------------------------:|
| Human B Lymphocytes | + | **0.933467** | 0.8641 |
| Human B Lymphocytes | - | 0.918187 | 0.9081 |
| Natural Killer Cells | + | **0.908418** | 0.9267 |
| Natural Killer Cells | - | 0.856171 | 0.8969 |



### Output Example
The figure below shows that the model accurately predicted and identified differential expression of PLEKHG2.
<div align="center">
<img src="Figure\output example.png" width="90%" title="Architecture">
</div>
