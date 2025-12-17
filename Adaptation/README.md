# Genos Server Adaptation
## 1. Introduction
Genos is a deep learning model designed for DNA sequence analysis. This server adaptation is specifically optimized for Huawei Ascend NPUs and Muxi MetaX accelerators, providing high-performance DNA sequence embedding extraction and nucleotide-level base prediction services via a RESTful API.

The directory is organized as follows:

```
Adaptation/
├── Dockerfile.npu      # Docker image configuration for NPU version
├── Dockerfile.metax    # Docker image configuration for MetaX version
├── genos_server.py     # Main server program providing API interfaces
└── README.md           # This documentation
```

## 2. Features

- **DNA Sequence Embedding Extraction**: Supports multiple pooling methods (mean, max, last, none)
- **DNA Nucleotide Prediction**: Predicts downstream nucleotides of DNA sequences
- **Multi-Device Support**: Prioritizes NPU usage while supporting GPU and CPU
- **Multi-Model Support**: Default support for 1.2B and 10B models
- **RESTful API**: Provides simple and user-friendly HTTP interfaces


## 3. Huawei Ascend NPU

###  Requirements

- Huawei Ascend NPU device (e.g., Atlas series)
- Docker environment
- NPU drivers properly installed

###  Build Docker Image

Navigate to the directory containing the Dockerfile and execute the following command to build the image:

```bash
cd Adaptation/
docker build --network=host -t genos-npu-image -f Dockerfile.npu .
```

###  Run Docker Container

After building, use the following command to run the container:

```bash
docker run --rm -d \
--name genos-npu-server \
--network=host \
--device=/dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--shm-size=2g \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /Path/To/Model/:/AI_models/zhejianglab/ \
-p 8000:8000 \
-it genos-npu-image
```

**Parameter Description**
- `--device=/dev/davinci0`: Specifies the NPU device to use (adjust according to actual conditions)
- `-v /Path/To/Model/:/AI_models/zhejianglab/`: Mounts the model file directory into the container
- `-p 8000:8000`: Port mapping. Adjust if the host port 8000 is occupied, e.g., `-p 8888:8000`
- `-d`: Runs the container in detached mode


## 4. Muxi MetaX

###  Requirements
- Muxi GPU devices (such as the Xiyun C500 series)  
- Docker environment  
- GPU drivers are correctly installed

###  Build Docker Image

Navigate to the directory containing the Dockerfile and execute the following command to build the image:

```bash
cd Adaptation/
docker build --network=host -t genos-metax-image -f Dockerfile.metax .
```

###  Run Docker Container

After building, use the following command to run the container:
```bash
docker run --rm -it -d \
--name genos-metax-server \
--network=host \
--device=/dev/dri \
--device=/dev/mxcd \
--privileged=true \
--group-add video \
--device=/dev/mem \
--device=/dev/infiniband \
--security-opt seccomp=unconfined \
--security-opt apparmor=unconfined \
--shm-size '100gb' \
--ulimit memlock=-1 \
-v /Path/To/Model/:/AI_models/zhejianglab/ \
-p 8000:8000 \
genos-metax-image
```
**Parameter Description**
- `--device=/dev/mxcd --device=/dev/dri`: Specifies the Muxi GPU device (adjust according to actual conditions)
- `-v /Path/To/Model/:/AI_models/zhejianglab/`: Mounts the model file directory into the container
- `-p 8000:8000`: Port mapping. Adjust if the host port 8000 is occupied, e.g., `-p 8888:8000`
- `-d`: Runs the container in detached mode

## 5. Invoke API Interface

After the container starts, you can call the service via HTTP API.

### Extract DNA Sequence Embedding

```bash
curl -X POST http://localhost:8000/extract \
-H "Content-Type: application/json" \
-d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "10B", "pooling_method": "max"}'
```

### Predict Downstream Nucleotides of DNA Sequence

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "10B", "predict_length": 25}'
```

## 6. API Interface Documentation

###  Embedding Extraction Interface

**URL**: `/extract`  
**Method**: `POST`  
**Content-Type**: `application/json`

**Request Parameters**:
- `sequence` (required): Input DNA sequence
- `model_name` (required): Model name ("1.2B" or "10B")
- `pooling_method` (optional): Pooling method ("mean", "max", "last", "none", default: "mean")

**Response Example**:
```json
{
  "success": true,
  "message": "Sequence embedding extraction successful",
  "result": {
    "sequence": "GGATCCGGATCCGGATCCGGATCC",
    "sequence_length": 24,
    "token_count": 5,
    "embedding_shape": [1, 1024],
    "embedding_dim": 1024,
    "pooling_method": "max",
    "model_type": "flash",
    "device": "npu:0",
    "embedding": [0.123, 0.456, ...]
  }
}
```

###  Nucleotide Prediction Interface

**URL**: `/predict`  
**Method**: `POST`  
**Content-Type**: `application/json`  

**Request Parameters**:
- `sequence` (required): Input DNA sequence
- `model_name` (required): Model name ("1.2B" or "10B")
- `predict_length` (optional): Number of nucleotides to predict (default: 10, maximum: 1000)

**Response Example**:
```json
{
  "success": true,
  "message": "Nucleotide prediction successful",
  "result": {
    "original_sequence": "GGATCCGGATCCGGATCCGGATCC",
    "predicted_sequence": "GGATCCGGATCCGGATCCGGATCCATCGATCGATCGATCGAT",
    "predicted_bases": "ATCGATCGATCGATCGAT",
    "predict_length": 16,
    "total_length": 40
  }
}
```

## 7. Custom Configuration

### Command-Line Arguments

`genos_server.py` supports the following command-line arguments:

| Argument | Description | Default Value |
|----------|-------------|---------------|
| `--host` | Server listening address | 0.0.0.0 |
| `--port` | Server listening port | 8000 |
| `--force_cpu` | Force CPU usage | False |
| `--device` | Specify running device (single device: npu:0, cuda:0, cpu; multiple devices: comma-separated, e.g., "cuda:0,cuda:1" or "npu:0,npu:1") | None |
| `--device_map` | Device mapping strategy (auto, balanced, sequential) | None |
| `--memory_ratio` | Memory allocation ratio | 0.9 |
| `--model_path_prefix` | Model storage path prefix | /AI_models/zhejianglab/ |
| `--log_level` | Logging level | INFO |

### Example: Custom Path and Device

```bash
docker run --rm -d \
--name genos-npu-server \
--network=host \
--device=/dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--shm-size=2g \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /DW/AI_models/modelscope/hub/models/zhejianglab/:/AI_models/zhejianglab/ \
-it genos-npu-image \
python genos_server.py --device npu:0
```


## 8. Viewing Logs

You can view container logs using the following command:

```bash
docker logs ${server_name}
```

## 9. Stopping the Service

Use the following command to stop and remove the container:

```bash
docker stop ${server_name}
```

## 10. Technical Support

For any questions or suggestions, please raise an issue or contact us at genos@zhejianglab.org.

---
