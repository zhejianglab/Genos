# Genos Python SDK Documentation


We provides a Python SDK that enables programmatic access to **Genos** models on the GeneOS platform. The SDK offers a unified interface for the following tasks: 

- 🧬 **Variant Pathogenicity Prediction**: Evaluate the potential pathogenic impact of genetic variants.
- 🔬 **DNA Sequence Embedding Extraction**: Generate high-dimensional embeddings from raw genomic sequences for downstream analysis.
- 🧪 **RNA-seq Coverage Track Prediction**: Predict RNA-seq coverage tracks from genomic coordinates.



In addition to support these tasks, the SDK also provides:
- 📊 **Genomic Visualization**: Plot and analyze genomic tracks with built-in visualization utilities.
- 🚀 **Easy-to-Use API**: A clean, Pythonic interface for seamless integration into bioinformatics workflows.
- 🔐 **Secure Authentication**: Automatic token validation and payment checking.
- ⚠️ **Comprehensive Error Handling**: Fine-grained exceptions types for clearer diagnostics and troubleshooting.



## Quick Start
### Get API Key
To use Genos, you need an API key from the DCS Cloud：

- Log in to [DCS Clould](https://cloud.stomics.tech/#/login) and navigate to **Personal Center → API Key Management**.
-  Click **“Create API Key”**.
- Read the **“API Usage Notice”** and confirm your agreement.
- The system will automatically generate your exclusive API Key — **copy and store it securely**.

> ⚠️NOTE: Please keep your API key confidential and avoid any unauthorized disclosure. The API key is for personal use only; sharing, transferring, or publishing it is strictly prohibited. If the key is leaked, misused, or used for illegal purposes, the platform reserves the right to immediately disable it. You can also manually deactivate your key at any time via the control panel.

### Installation
- Install from Source
  ```bash
  git clone https://github.com/BGI-HangzhouAI/Genos.git
  cd sdk
  pip install -e .
  ```

- Install from PyPI
  ```bash
  pip install genos-client
  ```

- Requirements
  -  Python 3.8 or higher
  - pip package manager

### Create a Client

```python
from genos import create_client

# Use GENOS_API_TOKEN environment variable
client = create_client()

# Or provide token explicitly
client = create_client(token="your_api_token_here")
```

### Run a Task

- Variant pathogenicity prediction

  ```python
  # Predict whether a genetic variant is pathogenic or benign

  result = client.variant_predict("hg19", "chr6", 51484075, "T", "G")['result']

  print(f"Variant: {result['variant']}")
  print(f"Prediction: {result['prediction']}")
  print(f"Pathogenic Score: {result['score_Pathogenic']:.4f}")
  print(f"Benign Score: {result['score_Benign']:.4f}")
  ```

- DNA sequence embedding extraction

  ```python
  # Generate deep learning embeddings from DNA sequences
  sequence = "ATCGATCGATCGATCGATCGATCGATCG"
  result = client.get_embedding(sequence, model_name="Genos-1.2B")['result']

  print(f"Sequence Length: {result['sequence_length']}")
  print(f"Embedding Dimension: {result['embedding_dim']}")
  print(f"Embedding Shape: {result['embedding_shape']}")

  # Access the embedding vector
  embedding_vector = result['embedding']  # List of floats
  ```
  You can choose different models and pooling methods depending on your analysis scenario:
  - **Available Models:**
    - `Genos-1.2B`: 1.2 billion parameters, lightweight and fast
    - `Genos-10B`: 10 billion parameters, more expressive but computationally heavier

  - **Pooling Methods:**
    - `mean`: Average pooling across sequence
    - `max`: Max pooling
    - `last`: Use last token embedding
    - `none`: Return all token embeddings


- RNA-seq coverage track prediction

  ```python
  # Predict RNA-seq coverage tracks from genomic coordinates
  result = client.rna_coverage_track_pred(chrom="chr6", start_pos=51484075)['result']

  print(f"Predicted coverage track: {result}")
  ```


## Advanced Configuration

### Custom Embedding Service

GenosClient allows users to configure a **custom embedding API endpoint**. This is useful if you want to deploy your own embedding service locally or within your organization.  

> Note: Variant prediction and RNA-seq coverage track prediction models are not open-source yet and cannot be self-hosted. Only the embedding service can be customized.

```python
from genos import GenosClient

# Initialize the client with a custom embedding endpoint
client = GenosClient(
    token="your_custom_token",  # Your token for authenticating with your own embedding service
    api_map={
        # Only the embedding service can be customized
        "embedding": "https://custom-embed-api.example.com/predict"
    }
)

# Calls to variant and RNA APIs will still use the official hosted services
```

### Timeout Configuration
You can adjust the request timeout for long-running operations:

```python
# Set a 60-second timeout
client = create_client(token="your_token", timeout=60)
```

## Error Handling
Genos provides comprehensive error handling with specific exception types for different scenarios. All API responses follow a consistent format.

### Error Response Format

```json
{
  "result": {},
  "status": "<HTTP_STATUS_CODE>",
  "messages": "<ERROR_MESSAGE>"
}
```

### Common Error Codes

| Status Code | Error Message | Description |
|-------------|---------------|-------------|
| 400 | Insufficient balance | Your account balance is insufficient for the requested operation |
| 401 | Invalid API Key | The provided API key is invalid or expired |
| 500 | Internal server error | An unexpected error occurred on the server side |

## Examples

Complete examples are available in the [`examples/`](examples/) directory:

- [`predict_variant.py`](examples/predict_variant.py): Variant pathogenicity prediction
- [`embedding_extract.py`](examples/embedding_extract.py): DNA sequence embedding extraction
- [`rna_generator.py`](examples/rna_generator.py): RNA-seq coverage track prediction
- [`error_handling_demo.py`](examples/error_handling_demo.py): Comprehensive error handling examples

## Acknowledgements
The Genos SDK is built on top of the open-source ecosystems of

- **PyTorch**
- **Hugging Face Transformers**
- **Megatron-LM**

We sincerely thank these projects and their communities for their invaluable contributions to open science.