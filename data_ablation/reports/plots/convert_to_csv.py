import re
import csv
import math

log_file = "/home/gowrav8849/V2/refactored/reports/plots/graphsage-upto_cornell.txt"
out_file = "/home/gowrav8849/V2/refactored/reports/plots/graphsage-upto_cornell.csv"

data = []

with open(log_file, "r") as f:
    for line in f:
        if " | I | runner | Run " in line:
            parts = line.split(" | I | runner | Run ")
            config_and_metrics = parts[1].strip()
            config_str, metrics_str = config_and_metrics.split(" | ")
            
            # Parse metrics
            acc_match = re.search(r"acc=([0-9.]+)", metrics_str)
            if acc_match:
                acc = float(acc_match.group(1))
            else:
                continue
            
            # Parse config
            seed_match = re.search(r"_s(\d+)$", config_str)
            if seed_match:
                seed = int(seed_match.group(1))
                config_str = config_str[:seed_match.start()]
            else:
                continue
                
            parts = config_str.split("_")
            dataset = parts[0]
            model = parts[1]
            
            feature_setting = parts[2]
            feature_selector = ""
            topk = ""
            
            if feature_setting == "topk":
                k_match = re.search(r"_k(\d+)$", config_str)
                if k_match:
                    topk = int(k_match.group(1))
                    prefix_len = len(dataset) + len(model) + len(feature_setting) + 3
                    feature_selector = config_str[prefix_len:k_match.start()].strip("_")
            
            data.append({
                "dataset": dataset,
                "model": model,
                "feature_setting": feature_setting,
                "feature_selector": feature_selector,
                "topk": topk,
                "acc": acc,
                "seed": seed
            })

grouped = {}
for row in data:
    key = (row["dataset"], row["model"], row["feature_setting"], row["feature_selector"], row["topk"])
    if key not in grouped:
        grouped[key] = []
    grouped[key].append(row["acc"])

# Sorting
def sort_key(k):
    dataset, model, feature_setting, feature_selector, topk = k
    fs_order = {"full": 0, "none": 1, "random": 2, "topk": 3}
    fs_score = fs_order.get(feature_setting, 4)
    fsel_order = {"": 0, "correlation": 1, "random_forest": 2, "variance": 3}
    fsel_score = fsel_order.get(feature_selector, 4)
    tk = topk if topk != "" else 0
    return (dataset, model, fs_score, fsel_score, tk)

sorted_keys = sorted(grouped.keys(), key=sort_key)

with open(out_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "model", "feature_setting", "feature_selector", "topk", "test_accuracy_mean", "test_accuracy_std", "count"])
    for k in sorted_keys:
        accs = grouped[k]
        count = len(accs)
        mean = sum(accs) / count
        if count > 1:
            variance = sum((x - mean) ** 2 for x in accs) / (count - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        
        row = list(k) + [mean, std, count]
        writer.writerow(row)

print(f"Successfully converted to {out_file}")
