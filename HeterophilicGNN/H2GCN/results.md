# Results Summary

## Overview

This document summarizes the performance of the implemented H2GCN-based model across multiple homophilic and heterophilic datasets.

---

## Experimental Setup

* Framework: PyTorch + PyTorch Geometric
* Training: Full-batch
* Optimizer: Adam
* Early Stopping: Enabled
* Splits:

  * WebKB, Actor, WikipediaNetwork → 10 splits
  * Planetoid datasets → single split

---

## Results

### Actor (Heterophilic)

| Split | Accuracy |
| ----- | -------- |
| 0     | 0.3895   |
| 1     | 0.3546   |
| 2     | 0.3618   |
| 3     | 0.3605   |
| 4     | 0.3645   |
| 5     | 0.3684   |
| 6     | 0.3461   |
| 7     | 0.3493   |
| 8     | 0.3572   |
| 9     | 0.3618   |

* Average Accuracy: **0.3614**

---

### Wisconsin (Heterophilic)

| Split | Accuracy |
| ----- | -------- |
| 0     | 0.7451   |
| 1     | 0.8824   |
| 2     | 0.8824   |
| 3     | 0.8627   |
| 4     | 0.8039   |
| 5     | 0.8431   |
| 6     | 0.8431   |
| 7     | 0.8039   |
| 8     | 0.8235   |
| 9     | 0.8824   |

* Average Accuracy: **0.8373**

---

### Squirrel (Heterophilic)

| Split | Accuracy |
| ----- | -------- |
| 0     | 0.3622   |
| 1     | 0.3929   |
| 2     | 0.3823   |
| 3     | 0.3939   |
| 4     | 0.4131   |
| 5     | 0.3756   |
| 6     | 0.3660   |
| 7     | 0.3900   |
| 8     | 0.3939   |
| 9     | 0.3573   |

* Average Accuracy: **0.3827**

---

### Chameleon (Heterophilic)

| Split | Accuracy |
| ----- | -------- |
| 0     | 0.4934   |
| 1     | 0.5088   |
| 2     | 0.4737   |
| 3     | 0.5461   |
| 4     | 0.4825   |
| 5     | 0.5022   |
| 6     | 0.4496   |
| 7     | 0.4737   |
| 8     | 0.5175   |
| 9     | 0.5197   |

* Average Accuracy: **0.4967**

---

## Observations

1. The model performs strongly on the Wisconsin dataset, achieving high accuracy despite low homophily.
2. Performance on Actor and Squirrel is moderate, reflecting the difficulty of these datasets.
3. Chameleon shows intermediate performance, indicating partially exploitable multi-hop structure.
4. Significant variation across splits is observed, especially in smaller datasets.
5. Larger and noisier datasets (e.g., Squirrel) exhibit lower performance despite similar homophily levels.

---

## Conclusion

* The model is effective on certain heterophilic graphs where multi-hop information is structured.
* Performance degrades on datasets with noisy features and complex graph topology.
* Homophily alone is insufficient to explain model behavior; feature quality and graph structure are equally important factors.

---
