<div align="center">
    <img src="Figure\Genos_LOGO.gif" width="99%" alt="Genos" />
</div>

<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.zero2x.org/genos" target="_blank">
      <img alt="Homepage" src="https://img.shields.io/badge/🌐%20Homepage-zero2x%20-536af5"/>
  </a>
  <a href="https://huggingface.co/ZhejiangLab/Genos" target="_blank">
      <img alt="Hugging Face" src="https://img.shields.io/badge/🤗%20Hugging%20Face-Genos%20-ffc107"/>
  </a>
  <a href="https://modelscope.cn/collections/zhejianglab/Genos" target="_blank">
      <img alt="modelscope" src="https://img.shields.io/badge/🤖%20ModelScope-Genos-FFC0CB"/>
  </a>
  <a href="https://cloud.stomics.tech/#/inferance-web?type=model" target="_blank">
      <img alt="DCS" src="https://img.shields.io/badge/☁️%20DCS-Inference Services%20-6f42c1"/>
  </a>
  <br>
  <a href="https://academic.oup.com/gigascience/advance-article/doi/10.1093/gigascience/giaf132/8296738?login=false" target="_blank">
      <img alt="Technical Report" src="https://img.shields.io/badge/📜%20Technical Report-GigaSience-brightgreen?logo=Linkedin&logoColor=white"/>
  </a>
  <a href="https://github.com/zhejianglab/Genos/blob/main/LICENSE" target="_blank">
       <img alt="License" src="https://img.shields.io/badge/📑%20License- Apache 2.0-f5de53"/> 
  <br>
</div>

## 1. Introduction
The Genos collection of models are foundation models for the human genome, achieving million-base-pair context modeling and single-nucleotide resolution learning capability for the human genome sequence. Genos adheres to the open science principles of collaboration, sharing, and co-construction, with a strong commitment to supporting the global genomics research community. To this end, we have openly released three models:

- [Genos-1.2B](https://huggingface.co/ZhejiangLab/Genos-1.2B): a 1.2 billion parameter model designed for high-efficiency genome sequence analysis.
- [Genos-10B](https://huggingface.co/ZhejiangLab/Genos-10B): a larger model with 10 billion parameters, offering enhanced performance for complex genomic tasks.
- [Genos-10B-v2](https://huggingface.co/ZhejiangLab/Genos-10B-v2): an enhanced version trained with additional human and non-human genomes to expand evolutionary context and sequence diversity.

 These models are openly released to facilitate research and innovation in genomics, enabling the broader scientific community to leverage and build upon this work. We also provide cloud inference services on [DCS Cloud](https://cloud.stomics.tech/#/inferance-web?type=model).

## 2. Model Information
The following figure illustrates the overall workflow of the model, including training data processing, model architecture, training process and downstream model inference and applications.

<div align="center">
<img src="Figure\Genos_model.png" width="90%" title="Architecture">
</div>

### Training Data
The training data for Genos are curated from large-scale, high-quality genomic resources, with a primary focus on human genomes. The core human corpus consists of haplotype-resolved and reference assemblies from internationally recognized consortia, including 231 assemblies from the Human Pangenome Reference Consortium (HPRC, Data Release 2), 65 assemblies from the Human Genome Structural Variation Consortium (HGSVC), 21 genomes from the Centre d’Étude du Polymorphisme Humain (CEPH) cohort, together with the GRCh38 and CHM13 reference genomes. After stringent quality control, this core dataset comprises 636 high-quality human genomes, totaling approximately 2,443.5B bases (corresponding to more than 1,500B tokens), and represents diverse global populations. For details on data preprocessing, please refer to our [Technical Report](https://doi.org/10.1093/gigascience/giaf132).

For [Genos-10B v2](https://huggingface.co/ZhejiangLab/Genos-10B-v2), to enhance evolutionary context and sequence diversity beyond the human lineage, its training corpus is further expanded using a phased data integration strategy. Additional datasets are introduced and mixed with the above human corpus at a 1:1 ratio during training. These datasets include approximately 60B bases of high-coverage East Asian human genomes generated using BGI’s CycloneSeq platform, 950.1B bases from RefSeq non-human primate genomes, and 484.85B bases from RefSeq non-primate mammalian genomes. This balanced and staged mixing approach broadens the taxonomic scope of the training data while preserving the central role of high-quality human genomic sequences and maintaining consistency in assembly quality and annotation standards.

The table below summarizes the genomic datasets used for Genos. We acknowledge these publicly available resources.

<div align="center">

| **Dataset** | **License/Data Use** | **Source** |
|:---:|:---:|:---:| 
| HPRC Data Release 2 | MIT License |🌐 [Official Website](https://humanpangenome.org/hprc-data-release-2/)|
| HGSVC | Public domain | 🌐 [Official Website](https://www.internationalgenome.org/data-portal/data-collection/structural-variation) |
| CEPH | Public domain |🌐 [Official Website](https://uofuhealth.utah.edu/center-genomic-medicine/research/ceph-resources)  |
| GRCh38 | Public domain |🌐 [Official Website](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/)|
| CHM13 | Public domain |🌐 [Official Website](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_009914755.1/)|
| High-coverage East Asian human genomes | Internal use | Internally generated | 
| RefSeq non-human primate genomes | Public domain |🌐 [Official Website](https://ftp.ncbi.nlm.nih.gov/refseq/release/)  | 
| RefSeq non-primate mammalian genomes| Public domain |🌐 [Official Website](https://ftp.ncbi.nlm.nih.gov/refseq/release/)|

</div>

### Model Architecture
The Genos models are build on the Transformer architecture and employs a Mixture-of-Experts (MoE) network. The main technical highlights include:

- Ultra-long sequence modeling: Leveraging ultra-long sequence parameterization, multi-dimensional tensor parallelism, and multi-scale attention, Genos can model sequences spanning millions of base pairs.

- Training stability optimization: A balanced expert load mechanism, combined with gradient clipping and expert selection strategies, mitigates imbalance in the expert modules caused by the small nucleotide vocabulary (4 bases).

- Dynamic expert activation: Both 1.2B and 10B parameter models support inference on sequences of up to millions of bases, with a dynamic routing algorithm that activates relevant expert modules in real time.


The following table provides an overview of the key specifications and architecture of the Genos models.

<div align="center">

| **Model Specification** | **Genos-1.2B** | **Genos-10B** | **Genos-10B-v2** |
|:---:|:---:|:---:|:---:|
| ++**Model Scale**++ |  |  | |
| Total Parameters | 1.2B | 10B | 10B |
| Activated Parameters | 0.33B | 2.87B | 2.87B |
| Trained Tokens | 1600B | 2200B | 6286B |
| ++**Architecture**++ |  |  ||
| Architecture | MoE | MoE | MoE |
| Number of Experts | 8 | 8 | 8 |
| Selected Experts per Token | 2 | 2 | 2 |
| Number of Layers | 12 | 12 | 12 |
| Attention Hidden Dimension | 1024 | 4096 | 4096 |
| Number of Attention Heads | 16 | 16 | 16 |
| MoE Hidden Dimension (per Expert) | 4096 | 8192 | 8192 |
| Vocabulary Size | 128 (padded) | 256 (padded) | 256 (padded) |
| Context Length | up to 1M | up to 1M | up to 1M |
</div>

### Training Process
The Genos models are trained using the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) framework across 256 GPUs, employing a sophisticated five-dimensional parallelism strategy that combines tensor parallelism, pipeline parallelism, context parallelism, data parallelism, and expert parallelism. 

- Key Features
  - **MoE**: 8 experts, Top-2 routing, 25% FFN sparsity
  - **GQA**: 50% KV cache reduction
  - **RoPE**: Base 50M for ultra-long context 
  - **Modern Stack**: RMSNorm, SwiGLU, Flash Attention

- Pre-training Strategy
  - **Objective**: Next Token Prediction (NTP) with self-supervised learning
  - **Progressive Context Scaling**: 8K → 32K → 128K → 1M tokens across training stages
  - **Data**:  high-quality, chromosome-scale de novo assemblies from publicly available resources
  - **Tokenizer**: One-hot optimized for DNA bases (A, T, C, G and N)

- Infrastructure
  - **Framework**: Megatron-LM on 256 GPUs
  - **Parallelism**: 5D strategy (TP, CP, DP, PP, EP)
  - **Batch**: Global 1024, Micro 1
  - **Optimizer**: AdamW (distributed sharded)
  - **Learning Rate**: up to 1e-4, cosine decay, 5-10% warmup
  - **Precision**: BF16 compute, FP32 for softmax/gradients/routing

- Key Optimizations
  - **MoE Load Balancing**: Aux loss (1e-3) + Z-loss (1e-3)
  - **Communication**: Grouped GEMM, AllToAll dispatch, overlapped gradient reduction
  - **Memory**: Flash Attention, sequence parallelism, distributed optimizer

- A representative fine-tuning setting:

  - **Task**: RNA-seq coverage prediction
  - **Data**: ENCODE + GTEx (667 samples, cell-type normalized)
  - **Architecture**: Genos backbone with a 3-layer CNN prediction head
  - **Hardware**: 64×H100 GPUs (bf16 precision)
  - **Training**: 1 epoch with LR = 5e-5, cosine learning-rate schedule


## 3. Performance Evaluation
### Evaluation Framework
We systematically evaluate the capabilities of the models in genome sequence analysis, transcription effect prediction, and biomedical downstream applications. Our evaluation focuses not only on the models’ performance on standard benchmark datasets but also on their potential to address real-world biomedical problems. In addition, these evaluations examine whether the models can capture biologically meaningful signals related to population differentiation and evolutionary history.

The evaluation is divided into three categories: 
  - Long sequence evaluation 
  - Short sequence evaluation
  - Mutation hot spot prediction

These tasks are designed to assess:
 - The model’s ability to identify and understand gene elements.
 - Its capacity to capture long-range regulatory interactions.
 - Its effectiveness in detecting local sequence variations that indicate susceptibility to mutations.

To evaluate the impact of incorporating new genomic data during training, [Genos-10B-v2](https://huggingface.co/ZhejiangLab/Genos-10B-v2) is trained with additional genome sequences from non-human primates and multiple mammalian species. To further assess the model’s understanding of this newly introduced data and its ability to generalize across diverse data distributions, we design the following two categories of evaluation tasks.

- **Cross-species Generalization**

    - Multi-species Sequence Classification: Based on the **NCBI RefSeq** database, we construct a multi-species sequence classification benchmark using genomic sequences from human, chimpanzee, Sumatran orangutan, common marmoset, mouse, cattle, and sheep. This task evaluates the model’s ability to capture low-level nucleotide sequence features and to represent both conserved and divergent patterns across species.

    - Genomic Element Classification: Using gene structure annotations from **Ensembl** for human, worm, fruit fly, mouse, zebrafish, and chicken, we define two sub-tasks:

      - Classification of different genomic elements within the same species (e.g., 5′UTR, 3′UTR, Exon, Gene)
      - Classification of the same genomic element across different species

      These tasks focus on evaluating the model’s ability to recognize structural characteristics of functional genomic regions and to generalize function-related sequence patterns across species.

- **Long-context Understanding**

  To comprehensively assess the model’s performance on long-range genomic context modeling, we adopt tasks from **[DNALongBench](https://github.com/ma-compbio/DNALONGBENCH)**. The evaluation selected includes the following tasks:

    - Enhancer–Target Gene Prediction: Determining whether a given enhancer–promoter pair is functionally associated, which evaluates the model’s capability to capture long-range regulatory interactions.

    - eQTL Prediction: Identifying whether a single-nucleotide polymorphism (SNP) acts as a positive expression quantitative trait locus (eQTL) for a target gene, testing the model’s ability to associate sequence-level regulatory variation with gene expression changes. 


### Evaluation Results
The figure below presents a comprehensive overview of the evaluation results across all tasks and models. Results are color-coded by rank for each evaluation task: **red** indicates the best-performing model (1st), **orange** indicates the second (2nd), and **green** indicates the third (3rd). As shown in the figure, the [Genos-10B-v2](https://huggingface.co/ZhejiangLab/Genos-10B-v2) model demonstrates competitive—and in several tasks, state-of-the-art—performance across diverse genomic benchmarks, highlighting its robustness in both long-context understanding and fine-grained sequence modeling.

<div align="center">
<img src="Figure\Evaluation_results.png" width="90%" title="Evaluation">
</div>


## 4. Quickstart
### Docker Deployment
We strongly recommend deploying Genos using Docker. 

Pull the Docker Image
```
docker pull zjlabgenos/mega:v1
```

Run the Container
```
docker run -it --gpus all --shm-size 32g zjlabgenos/mega:v1 /bin/bash
```

### Model Download
Genos models are available for download from [Hugging Face](https://huggingface.co/collections/ZhejiangLab/genos) and [ModelScope](https://modelscope.cn/collections/zhejianglab/Genos). Each model employs a hybrid Mixture-of-Experts (MoE) architecture and supports analysis at single-nucleotide resolution.

<div align="center">

| **Model** | **Total Params** | **Hugging Face** | **ModelScope** | **Megatron ckpt** |
|:---------:|:----------------:|:----------------:|:--------------:|:--------------:|
| Genos-1.2B | 1.2B | [🤗 Hugging Face](https://huggingface.co/ZhejiangLab/Genos-1.2B) |[🤖 ModelScope](https://modelscope.cn/models/zhejianglab/Genos-1.2B) | [Genos-1.2B](https://huggingface.co/ZhejiangLab/Genos-Megatron-1.2B) | 
| Genos-10B | 10B | [🤗 Hugging Face](https://huggingface.co/ZhejiangLab/Genos-10B) |[🤖 ModelScope](https://modelscope.cn/models/zhejianglab/Genos-10B) | [Genos-10B](https://huggingface.co/ZhejiangLab/Genos-Megatron-10B)  | 
| Genos-10B-v2 | 10B | [🤗 Hugging Face](https://huggingface.co/ZhejiangLab/Genos-10B-v2) |[🤖 ModelScope](https://modelscope.cn/models/zhejianglab/Genos-1.2B) | [Genos-10B-v2](https://huggingface.co/ZhejiangLab/Genos-Megatron-10B-v2)  | 

</div>

### API Calls
Install the Genos SDK
```
pip install genos-client
```

For detailed instructions on using the SDK, please refer to the [Genos SDK documentation](SDK).

### Usage Guide
Please refer to the tutorial notebooks for common usage scenarios:

- [Biological sequence embedding extraction](Notebooks/01.embedding_en.ipynb)
- [Variant pathogenicity prediction](Notebooks\02.ClinVar_variant_predict_en.ipynb)
- [RNA coverage track prediction](Notebooks\03.RNASeqConvTrack_en.ipynb)

## 5. Application Scenarios
To further illustrate the practical value, extensibility, and potential of Genos, we present two representative application cases.

- **Case 1: [RNA-Seq Data Generation](RNA-seq_data_generation/Case_1_RNA-Seq_Data_Generation.md)**  
  This case illustrates how Genos can be fine-tuned to generate transcriptomic profiles at single-nucleotide resolution directly from genomic sequences. This approach enables computational reconstruction of expression landscapes, reduces experimental costs, and provides a robust foundation for downstream functional genomics analyses.

- **Case 2: [Text-Genome Model Fusion](Text-genome_model_fusion/Case_2_Text_Genome_Model_Fusion.md)**  
  This case explores a multimodal framework that integrates genome-scale sequence encoders with large language models. It emphasizes the ability to jointly leverage biological prior knowledge, literature-based reasoning, and sequence-level representations, paving the way for more intelligent, interpretable, and knowledge-grounded bio-AI systems.

## 6. Model Inference Optimization and Adaptation
### vLLM Inference
We conduct inference optimization experiments on large language models using the vLLM framework. This initiative significantly enhances throughput and reduces inference latency. By leveraging vLLM’s innovative ```PagedAttention``` algorithm and efficient memory management mechanisms, we achieve a throughput improvement of over 7× compared with conventional inference approaches.

- Pull the Docker Image

```
docker pull zjlabgenos/vllm:v1
```

- Run the Container

```
docker run -it --entrypoint /bin/bash --gpus all --shm-size 32g zjlabgenos/vllm:v1
```

- For detailed test results and practical usage examples of vLLM, please refer to the [vllm example](https://github.com/zhejianglab/Genos/blob/main/Notebooks/04.vllm_example.ipynb).

### Other GPU adaptations
We also conduct compatibility tests on the following hardware accelerators. For detailed adaptation and deployment instructions, please refer to the [Adaptation](https://github.com/zhejianglab/Genos/tree/main/Adaptation) for more information.
- Huawei Ascend NPU
- MUXI GPU

## 7. License and Uses
**License**：The Genos collection of models are licensed under the  [Apache License 2.0](LICENSE).

**Primary intended use**：The primary use of Genos models is to support genomics research, providing researchers with advanced analytical capabilities and long-context modeling tools powered by large-scale foundation models for the human genome.

**Out-of-scope use**：Genos models are not intended for use in any manner that violates applicable laws or regulations, nor for any activities prohibited by the license agreement. 

**Ethical Considerations and Limitations**: Like other foundation models, Genos models may exhibit behaviors that carry potential risks. They may generate inaccurate outputs when interpreting genomic sequences or making inferences. Therefore, users should conduct rigorous validation and apply appropriate safeguards before using Genos in downstream research. Developers deploying applications based on Genos must carefully assess risks specific to their use cases, especially in contexts such as pharmaceutical development, clinical diagnosis, medical treatment, or any activities directly impacting human health.

## 8. Citation and Acknowledgements
We acknowledge the Human Pangenome Reference Consortium (HRPC; BioProject ID: PRJNA730823) and its funding agency, the National Human Genome Research Institute (NHGRI), for providing publicly available data. We also thank the BGI AI team for technical assistance.

If you use this work in your research, please cite the following paper:
```
@article{10.1093/gigascience/giaf132,
    author = {Genos Team, Hangzhou, China},
    title = {Genos: A Human-Centric Genomic Foundation Model},
    journal = {GigaScience},
    pages = {giaf132},
    year = {2025},
    month = {10},
    issn = {2047-217X},
    doi = {10.1093/gigascience/giaf132},
    url = {https://doi.org/10.1093/gigascience/giaf132},
    eprint = {https://academic.oup.com/gigascience/advance-article-pdf/doi/10.1093/gigascience/giaf132/64848789/giaf132.pdf},
}
```

## 9. Contact
If you have any questions, please raise an issue or contact us at [genos@zhejianglab.org](mailto:genos@zhejianglab.org).

