# Text-genome model fusion
## 1. Overview
This case presents a DNA sequence analysis system combining DNA sequence encoders with large language models, primarily designed for gene variant effect prediction and disease association analysis. By integrating DNA and text modalities, the system enables deep understanding and analysis of genomic sequences.

## 2. Data Analysis
### Input Data
We use a KEGG-based dataset derived from the KEGG task in the [Bioreason](https://arxiv.org/abs/2505.23579) study . This task compiles data through a multi-stage integration of KEGG pathway information and variant data from clinical databases. Molecular interactions within the biological network are represented using a standardized symbolic representation, and both reference and variant DNA sequences are provided to enable comparative analysis.

The dataset contains 1,449 entries covering 37 diseases, and is split into training, validation, and test sets with an 8:1:1 ratio. Each input sample includes a problem description, a reference DNA sequence, and a variant DNA sequence. The data are provided in JSON format, where each entry contains the following fields:

  ```json
  {
    "question": "Problem description defined by chromosome information and pathway networks",
    "answer": "Disease name (e.g., cushing syndrome, parkinson's disease, amyotrophic lateral sclerosis)",
    "reasoning": "Detailed reasoning steps, including 10 steps of biological analysis",
    "reference_sequence": "Reference DNA sequence (uppercase letters, spaces removed)",
    "variant_sequence": "Variant DNA sequence (uppercase letters, spaces removed)"
  }
  ```

### Output Data
The model outputs consist of textual generations that include both the explicit reasoning process and the final disease classification result. Specifically, the generated output contains a structured reasoning trace detailing the biological analysis steps, followed by the predicted disease label. The output is formatted as a single assistant response, as shown below:

  ```
  <|im_start|>assistant
  [Reasoning content]
  Answer: [Final answer]<|im_end|>
  ```
This output format ensures that the model’s decision-making process is transparent and interpretable, supporting detailed inspection of the reasoning steps alongside the final prediction.

## 3. Data Preprocessing Pipeline
### DNA Sequence Preprocessing
 - Sequence Normalization  
All DNA sequences undergo a standardized normalization process prior to model input. Specifically, all characters are converted to **uppercase**, and any **whitespace characters are removed** to ensure sequence consistency. To control sequence length and reduce boundary noise, sequences are truncated from both ends using a custom ```truncate_dna``` function, which removes **1,024 base pairs from each side**. For sequences shorter than the truncation threshold, the **central region** is retained to preserve the most informative content.  


   ```python
      def truncate_dna(example, truncate_dna_per_side=1024):
        # Truncate 1024 base pairs from each end of the sequence
        # If sequence is too short, return the middle portion
   ```


 - Sequence Tokenization  
Normalized DNA sequences are tokenized using the **OURGEN character-level tokenizer**, which operates directly at the nucleotide level. Special tokens, including ```<|dna_start|>```, ```<|dna_pad|>```, and ```<|dna_end|>```, are added to explicitly mark sequence boundaries and padding regions. The maximum sequence length is capped at **2,048 tokens** to ensure compatibility with model input constraints.

### Text Preprocessing
 - Dialogue Format Conversion  
To support multimodal reasoning, input data are converted into a dialogue-style format that combines DNA sequences with natural language descriptions. Each dialogue explicitly defines two roles: the ```user```, which contains the DNA sequences and problem description, and the ```assistant```, which contains the model’s reasoning process and final answer.

 - Template Application  
 A custom chat template is applied to format the dialogue input consistently. During this process, special control tokens such as ```<|im_start|>``` and ```<|im_end|>``` are handled explicitly to ensure correct segmentation of user and assistant content and proper alignment with the model’s generation mechanism.

### Data Loading and Batching
 - Dataset Split  
 The dataset is divided into training, validation, and test sets using an 80% / 10% / 10% split, ensuring balanced evaluation while maximizing training data availability.

 - Batching Function  
Data batching is performed using a custom ```qwen_dna_collate_fn```, specifically designed for the Qwen DNA model architecture. During batching, **loss computation is restricted to the assistant response segment** through label masking, ensuring that only generated reasoning and answers contribute to the training objective. A **left-padding strategy** is adopted, with appropriate insertion of special padding tokens to maintain sequence alignment across batch elements.

## 4. Model Architecture
### Core Components
As illustrated in the figure, the model adopts a multimodal architecture that integrates DNA sequence modeling with text-based reasoning. 

<div align="center">
<img src="Figure\text_gLM.png" width="90%" title="Architecture">
</div>

The core components include:
 - Multimodal Architecture Design: The framework combines a DNA encoder and a text model to jointly process genomic sequences and natural language inputs, enabling unified reasoning over sequence-level variation and biological knowledge.

 - Projection Layer: A projection layer is introduced to map DNA-derived representations into the text embedding space, allowing seamless fusion and interaction between genomic features and textual tokens.

 - LoRA Adaptation: LoRA is employed to enable parameter-efficient fine-tuning, reducing training cost while preserving model expressiveness.

### Training Strategy
 - Freezing Strategy: The training process supports optional freezing of the DNA encoder or the text model, allowing flexible trade-offs between computational efficiency and task adaptation.

 - Mixed Precision Training: DeepSpeed-based mixed precision training is used to improve memory efficiency and accelerate convergence.

 - Gradient Accumulation: Gradient accumulation is applied to support large effective batch sizes under hardware constraints.

## 5. Evaluation
### Metrics
The primary evaluation metric is Accuracy, defined as the proportion of correctly predicted samples over the total number of samples in the evaluation set. Accuracy is computed using the ```scikit-learn``` library. In addition to Accuracy, macro-averaged Precision, Recall, and F1-score can be optionally reported to provide a more comprehensive assessment across multiple classes.

  ```python
  # Use sklearn's classification_report
  report_dict = classification_report(
     y_true, y_pred, 
     labels=labels, 
     output_dict=True, 
     zero_division=1
  )

  # Extract macro-average metrics
  macro_metrics = report_dict['macro avg']
  Accuracy = accuracy_score(ground_truth, pred_label)
  # Precision = macro_metrics['precision'] # optional
  # Recall = macro_metrics['recall']  # optional
  # F1_score = macro_metrics['f1-score'] # optional
  ```

### Special Handling
 - Answer Extraction  
    - Rule-based Matching: The ```extract_single_entry``` function is used to extract predicted answers from the generated text.
    - Format Processing: Outputs containing ```<think>``` prefixes or intermediate reasoning are properly processed.
    - Error Handling: If a valid answer cannot be extracted, the prediction is recorded as ```NaN```.
 - Multi-class Support
    - Dynamic Label Inference: Class labels are dynamically determined based on ground-truth annotations.
    - Zero Division Handling: The parameter ```zero_division=1``` is applied to handle classes with no predicted samples.
    - Macro-average: Macro-averaged metrics are used to ensure equal weighting of all disease classes, regardless of class imbalance.

### Evaluation Results
The evaluation results of different models on the KEGG dataset are shown in the table below. Text–genome multimodal models significantly outperform unimodal models. In particular, the [021-8B](https://www.zero2x.org/021) and [Genos-1.2B](https://huggingface.co/ZhejiangLab/Genos-1.2B) fusion model achieves an accuracy of **98.28%**, which is 7% higher than the unimodal Genos-1.2B model (91.72%).

<div align="center">
<table>
  <thead>
    <tr><th>Text Model</th><th>Genome Model</th><th>Accuracy</th></tr>
  </thead>
  <tbody>
    <tr><td colspan="3" align="center"><strong>Unimodal Model</strong></td></tr>
    <tr><td>/</td><td>Genos-10B</td><td>0.9207</td></tr>
    <tr><td>/</td><td>Genos-1.2B</td><td><strong style="color: red;">0.9172</strong></td></tr>
    <tr><td>/</td><td><a href="https://huggingface.co/arcinstitute/evo2_1b_base" target="_blank">Evo2-1.2b</a></td><td>0.8828</td></tr>
    <tr><td>/</td><td><a href="https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen" target="_blank">HyenaDNA-1m</a></td><td>0.5000</td></tr>
    <tr><td>/</td><td><a href="https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species" target="_blank">NT-2.5b-multi</a></td><td>0.8655</td></tr>
    <tr><td colspan="3" align="center"><strong>Text–genome Multimodal Model</strong></td></tr>
    <tr><td>021-8B</td><td>Genos-1.2B</td><td><strong style="color: red;">0.9828</strong></td></tr>
    <tr><td>021-8B</td><td><a href="https://huggingface.co/arcinstitute/evo2_1b_base" target="_blank">Evo2-1.2b</a></td><td>0.9759</td></tr>
    <tr><td>Qwen3-8B</td><td><a href="https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen" target="_blank">HyenaDNA-1m</a></td><td>0.9758</td></tr>
    <tr><td>Qwen3-4B</td><td><a href="https://huggingface.co/arcinstitute/evo2_1b_base" target="_blank">Evo2-1.2b</a></td><td>0.9724</td></tr>
    <tr><td>021-8B</td><td>Genos-10B</td><td>0.9723</td></tr>
    <tr><td>Qwen3-4B</td><td>Genos-1.2B</td><td>0.9690</td></tr>
    <tr><td>Qwen3-4B</td><td><a href="https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species" target="_blank">NT-2.5b-multi</a></td><td>0.9690</td></tr>
    <tr><td>021-8B</td><td><a href="https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen" target="_blank">HyenaDNA-1m</a></td><td>0.9655</td></tr>
    <tr><td>Qwen3-8B</td><td><a href="https://huggingface.co/arcinstitute/evo2_1b_base" target="_blank">Evo2-1.2b</a></td><td>0.9621</td></tr>
    <tr><td>Qwen3-4B</td><td>Genos-10B</td><td>0.9621</td></tr>
    <tr><td>Qwen3-4B</td><td><a href="https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen" target="_blank">HyenaDNA-1m</a></td><td>0.9621</td></tr>
    <tr><td>Qwen3-1B</td><td><a href="https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen" target="_blank">HyenaDNA-1m</a></td><td>0.9345</td></tr>
    <tr><td>Qwen3-1B</td><td>Genos-1.2B</td><td>0.9138</td></tr>
    <tr><td>Qwen3-1B</td><td><a href="https://huggingface.co/arcinstitute/evo2_1b_base" target="_blank">Evo2-1.2b</a></td><td>0.9042</td></tr>
    <tr><td>Qwen3-1B</td><td>Genos-10B</td><td>0.8897</td></tr>
    <tr><td>Qwen3-1B</td> <td><a href="https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species" target="_blank">NT-2.5b-multi</a></td><td>0.8842</td></tr>
  </tbody>
</table>
</div>


> [!Note]  
> [021-8B](https://www.zero2x.org/021) is a science foundation model trained on extensive scientific corpora covering 174 scientific domains. It is designed to enhance scientific understanding and reasoning, and to promote interdisciplinary innovation and discovery. It will be open-sourced soon.


## 6. Usage
### Environment Setup
This case requires running on a machine with GPU. Please ensure that CUDA is installed on your machine first.

```python
# Check environment
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```
We provide a ```requirements.txt``` file. Users can install dependencies using:

```shell
pip install -r requirements.txt
```

### Model Training
There are two ways to run model training: via shell script or Jupyter notebook.
 - Shell Script  
    Start training by running:
      ```
      sh_train.sh
      ```
 - Jupyter Notebook  
    Alternatively, run the notebook:
      ```
      user_case.ipynb`
      ```

### Model Deployment
The case can use ```FastAPI``` for deployment.
 - Start the service by running:
  ```
    sh start_simple.sh
  ```

 - Test the service by running:
  ```
    python test_simple_api.py
  ```


