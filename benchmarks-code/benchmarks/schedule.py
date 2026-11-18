import os
import torch
import multiprocessing as mp
from collections import deque
from transformers import AutoTokenizer, AutoModel
from benchmarks.evaluation import evaluate_model_on_dataset_layer, results_exist
from benchmarks.embedding_extract import need_extract_layer, load_dataset_class_jsonl_split, save_embedding, JSONLDataset
from benchmarks.analysis_reprot import analysis_reprot
import pandas as pd
from benchmarks.initation import set_seed
torch.use_deterministic_algorithms(True)


def embedding_worker(gpu_id, submit_communication, return_communication, config):
    set_seed(config._config['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'], trust_remote_code=True)
    model = AutoModel.from_pretrained(config['model_path'], device_map="auto",
                                      torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    while True:
        task = submit_communication.get()
        if task is None:
            break
        if len(task) == 3:
            dataset, split, dataset_class = task
            i = None
        elif len(task) == 4:
            dataset, split, i, dataset_class = task
        else:
            raise ValueError(f"Invalid task: {task}")
        save_embedding(
            dataset, gpu_id, model, tokenizer, split, i, dataset_class, config)
        result_message = (dataset, split, i)
        return_communication.put(result_message)


def result_writter(results, config):
    df = pd.DataFrame(results)
    result_path = f"{config['eval_result_path']}/{config['model_name']}/{results[0]['task']}.tsv"
    # Load and update the results file if it exists
    if os.path.exists(result_path):
        existing_df = pd.read_csv(result_path, sep="\t")
        # Update or append each new result in the existing DataFrame
        for _, row in df.iterrows():
            mask = (existing_df['task'] == row['task']) & (
                existing_df['layer'] == row['layer'])
            # Match the column order of the existing DataFrame
            row_df = pd.DataFrame([row])[existing_df.columns]

            if mask.sum() == 1:
                # Update when exactly one row matches
                existing_df.loc[mask] = row_df.iloc[0]
            elif mask.sum() > 1:
                # If multiple rows match, remove them and append the new row
                existing_df = existing_df[~mask]
                existing_df = pd.concat(
                    [existing_df, row_df], ignore_index=True)
            else:
                # Append the row when there is no match
                existing_df = pd.concat(
                    [existing_df, row_df], ignore_index=True)
        # Use the updated DataFrame
        df = existing_df

    # Sort by layer
    df = df.sort_values('layer')
    # Save the results
    df.to_csv(result_path, sep="\t", index=False)


def splited_embedding_concat(dataset, config, split):
    for i, sp in enumerate(config.dataset_info[dataset]["data_split"]):
        if sp == split:
            data_num = config.dataset_info[dataset]["dataset_ratio"][i] * \
                config.dataset_info[dataset]["sample_num"]
            break
    split_part_number = int(data_num // config['embedding_extract_split'] +
                            1 if data_num % config['embedding_extract_split'] > 0 else data_num // config['embedding_extract_split'])
    split_output_dir = f"{config['embedding_output_dir']}/{config['model_name']}/split-{config['embedding_extract_split']}"
    for layer in config['layer_to_eval']:
        data_output = {"embeddings": [], "labels": []}
        for i in range(split_part_number):
            data = torch.load(
                f"{split_output_dir}/{dataset}-{layer}layer_{split}-{i}.pt")
            data_output["embeddings"].extend(data["embeddings"])
            data_output["labels"].extend(data["labels"])
        data_output = {"embeddings": torch.stack(
            data_output["embeddings"]), "labels": torch.stack(data_output["labels"])}
        torch.save(
            data_output, f"{config['embedding_output_dir']}/{config['model_name']}/{dataset}-{layer}layer_{split}.pt")
        file2delete = [
            f"{split_output_dir}/{dataset}-{layer}layer_{split}-{i}.pt" for i in range(split_part_number)]
        for file in file2delete:
            os.remove(file)


def eval_worker(dataset, layer, gpu_id, result_queue, config):
    set_seed(config._config['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model_name = config['model_name']
    # model_layers = config.model_config['num_hidden_layers']
    results = evaluate_model_on_dataset_layer(
        dataset, gpu_id, model_name, layer, config)
    result_queue.put((gpu_id, results))


def embedding_task_split(tasks_to_do, config) -> list:
    embedding_split_path = f"{config['embedding_output_dir']}/{config['model_name']}/split-{config['embedding_extract_split']}"
    os.makedirs(embedding_split_path, exist_ok=True)
    embedding_extract_split = config['embedding_extract_split']
    embedding_task_list = []
    embedding_subdataset_task_view = {dataset: {} for dataset in tasks_to_do}
    for dataset in tasks_to_do:
        embedding_output_dir = f"{config['embedding_output_dir']}/{config['model_name']}"
        result_dir = f"{config['eval_result_path']}/{config['model_name']}"
        result_path = f"{result_dir}/{dataset}.tsv"
        layer2eval = results_exist(
            result_path, config['layer_to_eval'], config.model_config['num_hidden_layers'], config.dataset_info[dataset]["eval_task"])
        layer2extract = need_extract_layer(
            embedding_output_dir, layer2eval, dataset, config)
        for i, split in enumerate(config.dataset_info[dataset]["data_split"]):
            if len(layer2extract[split]) == 0:
                continue
            item_number = config.dataset_info[dataset]["dataset_ratio"][i] * \
                config.dataset_info[dataset]["sample_num"]
            if item_number <= embedding_extract_split:
                embedding_task_list.append((dataset, split, JSONLDataset(
                    dataset, split, config)))
            else:
                splited_dataset = load_dataset_class_jsonl_split(
                    dataset, config, split)
                embedding_subdataset_task_view[dataset][split] = splited_dataset[0]
                embedding_task_list.extend(splited_dataset[1])
                # An empty embedding_subdataset_task_view[dataset][split] indicates that this split was divided and all sub-splits are complete. Merge them in the main process.
    return embedding_task_list, embedding_subdataset_task_view


def task_view(config):
    tasks_to_do = {"embedding": [], "eval": []}
    for dataset in config['eval_datasets']:
        result_dir = f"{config['eval_result_path']}/{config['model_name']}"
        result_path = f"{result_dir}/{dataset}.tsv"
        layer2eval = results_exist(
            result_path, config['layer_to_eval'], config.model_config['num_hidden_layers'], config.dataset_info[dataset]["eval_task"])
        if len(layer2eval) == 0:
            continue
        else:
            embedding_output_dir = f"{config['embedding_output_dir']}/{config['model_name']}"
            layer2extract = need_extract_layer(
                embedding_output_dir, layer2eval, dataset, config)
            if layer2extract is None:
                for layer in layer2eval:
                    tasks_to_do["eval"].append((dataset, layer))
            else:
                for split in config.dataset_info[dataset]["data_split"]:
                    if len(layer2extract[split]) > 0:
                        tasks_to_do["embedding"].append(dataset)
                        break
    # Calculate priorities for embedding and evaluation tasks
    embedding_priority = [config.dataset_info[dataset]["sample_num"]*config.dataset_info[dataset]
                          ["seq_for_item"]*config.dataset_info[dataset]["max_length"] for dataset in tasks_to_do["embedding"]]
    eval_priority = [config.dataset_info[dataset]["sample_num"] *
                     config.dataset_info[dataset]["dataset_ratio"][0] for dataset, layer in tasks_to_do["eval"]]
    tasks_to_do['embedding'] = sorted(
        tasks_to_do['embedding'], key=lambda x: embedding_priority[tasks_to_do['embedding'].index(x)], reverse=True)
    tasks_to_do['embedding'], embedding_subdataset_task_view = embedding_task_split(
        tasks_to_do['embedding'], config)
    tasks_to_do['eval'] = sorted(
        tasks_to_do['eval'], key=lambda x: eval_priority[tasks_to_do['eval'].index(x)], reverse=True)
    return tasks_to_do, embedding_subdataset_task_view


def scheduler(config):
    """
    Task scheduler. Embeddings can be extracted for all layers at once, so embedding tasks are scheduled by dataset and evaluation tasks by (dataset, layer).
    """
    gpu_list = config['gpu_list']
    gpu_num = len(gpu_list)
    os.makedirs(
        f"{config['eval_result_path']}/{config['model_name']}", exist_ok=True)

    # Get the initial task lists, ordered by priority. Embedding tasks are queued by dataset and evaluation tasks by (dataset, layer).
    tasks_to_do, embedding_subdataset_task_view = task_view(config)

    # Count evaluation tasks to determine when the loop should end
    # Number of initial evaluation tasks
    task_num2eval = len(tasks_to_do['eval'])
    # Count evaluation tasks added after embedding extraction
    ds2embedding = set([task[0] for task in tasks_to_do['embedding']])
    for dataset in ds2embedding:
        task_num2eval += len(results_exist(
            f"{config['eval_result_path']}/{config['model_name']}/{dataset}.tsv", config['layer_to_eval'], config.model_config['num_hidden_layers'], config.dataset_info[dataset]["eval_task"]))

    # Ranks are used for task assignment and GPU IDs identify devices. These dictionaries map between ranks and GPU IDs.
    rank_to_gpu_id = {rank: gpu_id for rank, gpu_id in enumerate(gpu_list)}
    gpu_id_to_rank = {gpu_id: rank for rank, gpu_id in rank_to_gpu_id.items()}
    ranks = list(rank_to_gpu_id.keys())
    # Create queues for pending embedding and evaluation tasks
    embedding_queue = deque(tasks_to_do['embedding'])
    eval_queue = deque(tasks_to_do['eval'])
    # Track remaining embedding tasks to shut down excess embedding processes
    excess_embedding_task = len(embedding_queue)

    # Track the task weight assigned to each rank as a measure of GPU load
    rank_power = [0]*len(ranks)

    # Create queues for passing embedding and evaluation status from child processes to the main process
    completed_eval = 0
    eval_statue = {task: mp.Queue() for task in tasks_to_do['eval']}

    print(
        f"[SCHEDULER] Starting benchmarks for {task_num2eval} dataset-layers ({len(list(embedding_subdataset_task_view.keys()))} datasets need to extract embedding) on {gpu_num} GPUs")
    # Create dictionaries for managing embedding and evaluation processes
    submitted_eval_process = {}

    # Initialize the first batch of embedding tasks
    # Calculate the number of embedding tasks each GPU can handle
    embedding_process_per_gpu = config['process_power_per_gpu']//config['embedding_process_power']
    # Determine the initial number of embedding tasks
    init_embedding_tasks_num = min(
        len(embedding_queue), gpu_num*embedding_process_per_gpu)
    init_rank = 0
    process_idx_on_gpu = 0
    submitted_embedding_process = {}
    embedding_process_status = {}
    embedding_process_submit_communication = {}
    embedding_process_return_communication = {}
    # Initialize the required number of embedding processes
    for _ in range(init_embedding_tasks_num):
        if init_rank == len(ranks):
            init_rank = 0
            process_idx_on_gpu += 1
        embedding_process_submit_communication[(
            init_rank, process_idx_on_gpu)] = mp.Queue()
        embedding_process_return_communication[(
            init_rank, process_idx_on_gpu)] = mp.Queue()
        submitted_embedding_process[(init_rank, process_idx_on_gpu)] = mp.Process(target=embedding_worker, args=(
            rank_to_gpu_id[init_rank], embedding_process_submit_communication[(init_rank, process_idx_on_gpu)], embedding_process_return_communication[(init_rank, process_idx_on_gpu)], config))
        submitted_embedding_process[(init_rank, process_idx_on_gpu)].start()
        embedding_process_status[(init_rank, process_idx_on_gpu)] = 0
        rank_power[init_rank] += config['embedding_process_power']
        init_rank += 1

    # Concatenate datasets whose embeddings are complete
    for dataset in list(embedding_subdataset_task_view.keys()):
        for split in list(embedding_subdataset_task_view[dataset].keys()):
            if len(embedding_subdataset_task_view[dataset][split]) == 0:
                splited_embedding_concat(dataset, config, split)
                del embedding_subdataset_task_view[dataset][split]
                # Check whether embeddings are complete for every split of this dataset. If so, add its evaluation tasks to the queue.
                for sub_split in config.dataset_info[dataset]["data_split"]:
                    embedding_complete = True
                    if embedding_complete is False:
                        break
                    for layer in config['layer_to_eval']:
                        embedding_result_path = f"{config['embedding_output_dir']}/{config['model_name']}/{dataset}-{layer}layer_{sub_split}.pt"
                        if not os.path.exists(embedding_result_path):
                            embedding_complete = False
                            break
                if embedding_complete:
                    layer2eval = results_exist(
                        f"{config['eval_result_path']}/{config['model_name']}/{dataset}.tsv", config['layer_to_eval'], config.model_config['num_hidden_layers'], config.dataset_info[dataset]["eval_task"])
                    for layer in layer2eval:
                        eval_queue.append((dataset, layer))
                        eval_statue[(dataset, layer)] = mp.Queue()
                    # Reorder eval_queue by priority
                    eval_priority = [config.dataset_info[dataset]["sample_num"] *
                                     config.dataset_info[dataset]["dataset_ratio"][0] for dataset, layer in eval_queue]
                    eval_queue = deque(sorted(
                        eval_queue, key=lambda x: eval_priority[eval_queue.index(x)], reverse=True))

    # Process results and assign new tasks
    # Stop when the number of submitted evaluation tasks reaches task_num2eval
    while completed_eval < task_num2eval:
        # Assign a new embedding extraction task if the queue is not empty
        if embedding_queue:
            for idx, status in embedding_process_status.items():
                if status == 0:
                    rank, process_idx_on_gpu = idx
                    task = embedding_queue.popleft()
                    embedding_process_submit_communication[(
                        rank, process_idx_on_gpu)].put(task)
                    embedding_process_status[(rank, process_idx_on_gpu)] = 1

        # Assign a new evaluation task
        if eval_queue:
            min_rank_power = min(rank_power)
            if min_rank_power+config['eval_process_power'] <= config['process_power_per_gpu']:
                dataset, layer = eval_queue.popleft()
                easiest_rank_id = rank_power.index(
                    min_rank_power)
                submitted_eval_process[(dataset, layer)] = mp.Process(target=eval_worker, args=(
                    dataset, layer, rank_to_gpu_id[easiest_rank_id], eval_statue[dataset, layer], config))
                submitted_eval_process[(dataset, layer)].start()
                rank_power[easiest_rank_id] += config['eval_process_power']

        # Check the status of embedding tasks
        for idx in list(embedding_process_return_communication.keys()):
            result_queue = embedding_process_return_communication[idx]
            if result_queue.empty():
                continue
            result = result_queue.get()
            rank, process_idx_on_gpu = idx
            dataset, split, i = result
            embedding_process_status[(rank, process_idx_on_gpu)] = 0
            # Shut down processes when fewer tasks remain than allocated resources
            excess_embedding_task -= 1
            if excess_embedding_task < len(submitted_embedding_process):
                embedding_process_submit_communication[(
                    rank, process_idx_on_gpu)].put(None)
                embedding_process_submit_communication[(
                    rank, process_idx_on_gpu)].close()
                embedding_process_return_communication[(
                    rank, process_idx_on_gpu)].close()
                submitted_embedding_process[(rank, process_idx_on_gpu)].join()
                embedding_process_submit_communication[(
                    rank, process_idx_on_gpu)].join_thread()
                embedding_process_return_communication[(
                    rank, process_idx_on_gpu)].join_thread()
                del submitted_embedding_process[(rank, process_idx_on_gpu)]
                del embedding_process_status[(rank, process_idx_on_gpu)]
                del embedding_process_submit_communication[(
                    rank, process_idx_on_gpu)]
                del embedding_process_return_communication[(
                    rank, process_idx_on_gpu)]
                rank_power[rank] -= config['embedding_process_power']

            if i is None:
                embedding_complete = True
                for split in config.dataset_info[dataset]["data_split"]:
                    if embedding_complete is False:
                        break
                    for layer in config['layer_to_eval']:
                        embedding_result_path = f"{config['embedding_output_dir']}/{config['model_name']}/{dataset}-{layer}layer_{split}.pt"
                        if not os.path.exists(embedding_result_path):
                            embedding_complete = False
                            break
            else:
                # Remove the entry where embedding_subdataset_task_view[dataset][split] equals i
                embedding_subdataset_task_view[dataset][split].remove(i)
                if len(embedding_subdataset_task_view[dataset][split]) == 0:
                    splited_embedding_concat(dataset, config, split)
                    del embedding_subdataset_task_view[dataset][split]
                    embedding_complete = True
                    for split in config.dataset_info[dataset]["data_split"]:
                        if embedding_complete is False:
                            break
                        for layer in config['layer_to_eval']:
                            embedding_result_path = f"{config['embedding_output_dir']}/{config['model_name']}/{dataset}-{layer}layer_{split}.pt"
                            if not os.path.exists(embedding_result_path):
                                embedding_complete = False
                                break
                else:
                    embedding_complete = False

            if embedding_complete:
                # Add evaluation tasks (dataset, layer) when embedding extraction is complete
                layer2eval = results_exist(
                    f"{config['eval_result_path']}/{config['model_name']}/{dataset}.tsv", config['layer_to_eval'], config.model_config['num_hidden_layers'], config.dataset_info[dataset]["eval_task"])
                for layer in layer2eval:
                    eval_queue.append((dataset, layer))
                    eval_statue[(dataset, layer)] = mp.Queue()
                # Reorder eval_queue by priority
                eval_priority = [config.dataset_info[dataset]["sample_num"] *
                                 config.dataset_info[dataset]["dataset_ratio"][0] for dataset, layer in eval_queue]
                eval_queue = deque(sorted(
                    eval_queue, key=lambda x: eval_priority[eval_queue.index(x)], reverse=True))

        # Check the status of evaluation tasks
        dataset_layer_completed_eval = []
        for dataset_layer, result_queue in eval_statue.items():
            if result_queue.empty():
                continue
            gpu_id, result = result_queue.get()
            rank_power[gpu_id_to_rank[gpu_id]] -= config['eval_process_power']
            result_writter(result, config)
            completed_eval += 1
            dataset_layer_completed_eval.append(dataset_layer)
        for dataset_layer in dataset_layer_completed_eval:
            eval_statue[(dataset_layer)].close()
            submitted_eval_process[dataset_layer].join()
            eval_statue[(dataset_layer)].join_thread()
            del submitted_eval_process[dataset_layer]
            del eval_statue[dataset_layer]

    print("✅ All datasets embedding extracted and evaluated!")
    analysis_reprot(config)
    print(
        f'Report exported to {config["eval_result_path"]}/{config["model_name"]}/reports')
