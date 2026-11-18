import pandas as pd
import matplotlib.pyplot as plt
import os


def best_table(target, dataset_list, config, report_path):
    result_path = f"{report_path}/best_{target}.tsv"
    result_df_path = [
        f"{config['eval_result_path']}/{config['model_name']}/{dataset}.tsv" for dataset in dataset_list]
    results_df = [pd.read_csv(path, sep="\t") for path in result_df_path]
    results_df = [result_df.loc[result_df['layer'].isin(
        config['layer_to_eval'])] for result_df in results_df]

    if len(results_df) > 0:
        best_result_df = pd.concat([result_df.loc[[result_df[target].idxmax()]]
                                   for result_df in results_df], ignore_index=True)
        best_result_df.to_csv(result_path, sep="\t", index=False)
    else:
        print(f"No results found for {target} to create best table")


def last_layer_table(dataset_list, config, report_path):
    result_path = f"{report_path}/last_layer.tsv"
    result_df_path = [
        f"{config['eval_result_path']}/{config['model_name']}/{dataset}.tsv" for dataset in dataset_list]
    results_df = [pd.read_csv(path, sep="\t") for path in result_df_path]
    results_df = [result_df.loc[result_df['layer'].isin(
        config['layer_to_eval'])] for result_df in results_df]
    results_df = [result_df.loc[[result_df['layer'].idxmax()]]
                  for result_df in results_df]
    results_df = pd.concat(results_df, ignore_index=True)
    results_df.to_csv(result_path, sep="\t", index=False)


def multi_best_table(targets, dataset_list, config, report_path):
    result_path = f"{report_path}/best_{'-'.join(targets)}.tsv"
    result_df_path = [
        f"{config['eval_result_path']}/{config['model_name']}/{dataset}.tsv" for dataset in dataset_list]
    results_df = [pd.read_csv(path, sep="\t") for path in result_df_path]
    results_df = [result_df.loc[result_df['layer'].isin(
        config['layer_to_eval'])] for result_df in results_df]
    table_index = ["task"]
    for target in targets:
        table_index.extend([target, f"{target}_layer"])
    best_result = {index: [] for index in table_index}
    if len(results_df) > 0:
        for result in results_df:
            best_result["task"].append(result["task"][0])
            for target in targets:
                best_result[target].append(
                    result.loc[result[target].idxmax()][target])
                best_result[f"{target}_layer"].append(
                    result.loc[result[target].idxmax()]['layer'])
        best_result_df = pd.DataFrame(best_result)
        best_result_df.to_csv(result_path, sep="\t", index=False)
    else:
        print(f"No results found for {targets} to create best table")


def target_layer_line_plot(target_main, target_sub, dataset_list, config, report_path):
    if len(dataset_list) > 0:
        result_path = f"{report_path}/line_plot_{target_main}.png"
        if target_sub is not None:
            result_path = f"{report_path}/line_plot_{target_main}({'_'.join(target_sub)}).png"
        plt.figure(figsize=(14, 6))
        for dataset in dataset_list:
            result_df_path = f"{config['eval_result_path']}/{config['model_name']}/{dataset}.tsv"
            result_df = pd.read_csv(result_df_path, sep="\t")
            y = result_df[target_main].values
            x = result_df['layer'].values
            max_idx = y.argmax()
            best_value = y[max_idx]
            best_layer = x[max_idx]
            if target_sub is not None:
                sub_target_str = ','
                for target in target_sub:
                    value_sub = result_df.loc[result_df['layer']
                                              == best_layer, target].values[0]
                    sub_target_str += f" {target}: {value_sub:.4f},"
                legend_label = f"{dataset} ({target_main}: {best_value:.4f}{sub_target_str} layer: {best_layer})"
            else:
                legend_label = f"{dataset} ({target_main}: {best_value:.4f}, layer: {best_layer})"
            plt.plot(x, y, marker='o', label=legend_label)
            plt.plot(x[max_idx], y[max_idx], marker='*',
                     color=plt.gca().lines[-1].get_color(), markersize=18, label=None, zorder=10)
            plt.annotate(f"{best_value:.4f} (layer {best_layer})",
                         xy=(x[max_idx], y[max_idx]),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', va='bottom', fontsize=12,
                         color=plt.gca().lines[-1].get_color())
        plt.title(f"{target_main} vs Layer for {config['model_name']}")
        plt.xlabel("Layer")
        plt.ylabel(target_main)
        plt.ylim(0, 1)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(result_path)
        plt.close()
    else:
        print(
            f"No results found for {target_main} vs Layer for {config['model_name']} to create line plot")


def analysis_reprot(config):
    model_name = config['model_name']
    classification_label_targets = [
        "accuracy", "roc_auc", "precision", "recall", "f1", "mcc"]
    regression_targets = ["mse", "mae", "r2", "pearsonr", "spearmanr"]
    classification_label_dataset_list = [dataset for dataset in config['eval_datasets'] if config.dataset_info[dataset]
                                         ["eval_task"] == "classification" or config.dataset_info[dataset]["eval_task"] == "labels"]
    regression_dataset_list = [dataset for dataset in config['eval_datasets']
                               if config.dataset_info[dataset]["eval_task"] == "regression"]
    report_path = f"{config['eval_result_path']}/{model_name}/reports"
    os.makedirs(report_path, exist_ok=True)
    for target in config['target_best_table']:
        if type(target) == str:
            if target in classification_label_targets:
                best_table(target, classification_label_dataset_list,
                           config, report_path)
            elif target in regression_targets:
                best_table(target, regression_dataset_list,
                           config, report_path)
            else:
                raise ValueError(f"Cannot report the target: {target}")
        elif type(target) == list:
            if target[0] in classification_label_targets:
                multi_best_table(
                    target, classification_label_dataset_list, config, report_path)
            elif target[0] in regression_targets:
                multi_best_table(
                    target, regression_dataset_list, config, report_path)
            else:
                raise ValueError(f"Cannot report the target: {target}")

    for target in config['target_layer_line_figure']:
        if type(target) == str:
            target_main = target
            target_sub = None
        elif type(target) == list:
            target_main = target[0]
            target_sub = target[1:]
        else:
            raise ValueError(
                f"Invalid element type in target_layer_line_figure: {type(target)}")
        if target_main in classification_label_targets:
            target_layer_line_plot(target_main, target_sub, classification_label_dataset_list,
                                   config, report_path)
        elif target_main in regression_targets:
            target_layer_line_plot(
                target_main, target_sub, regression_dataset_list, config, report_path)
    last_layer_table(config['eval_datasets'], config, report_path)
