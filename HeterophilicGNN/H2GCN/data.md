# 📊 Dataset Overview for Experiments

This project evaluates GNN models on both **homophilic** and **heterophilic** graph datasets.

---

# 🧠 Key Terms

* **Nodes** → entities (web pages, papers, actors, etc.)
* **Edges** → relationships (links, citations, co-occurrence)
* **Features** → input attributes of nodes
* **Classes** → labels for node classification
* **Homophily** → neighbors share same label
* **Heterophily** → neighbors have different labels

---

# 📐 Homophily Rate

Homophily is defined as:

[
h = \frac{|{(u,v) \in E : y_u = y_v}|}{|E|}
]

👉 Fraction of edges connecting nodes of the **same class**

---

# 🌐 WebKB Datasets (Cornell, Texas, Wisconsin)

## 📊 Stats

| Dataset   | Nodes | Edges | Features | Classes | Homophily |
| --------- | ----- | ----- | -------- | ------- | --------- |
| Cornell   | 183   | 295   | 1703     | 5       | 0.30      |
| Texas     | 183   | 309   | 1703     | 5       | 0.11      |
| Wisconsin | 251   | 499   | 1703     | 5       | 0.21      |

## 🧠 Meaning

* Nodes → Web pages
* Edges → hyperlinks
* Features → bag-of-words
* Classes → page categories

## ⚠️ Properties

* Strong **heterophily**
* Small graphs → high variance

---

# 🎭 Actor Dataset

## 📊 Stats

| Nodes | Edges  | Features | Classes | Homophily |
| ----- | ------ | -------- | ------- | --------- |
| 7600  | ~30000 | 932      | 5       | 0.22      |

## 🧠 Meaning

* Nodes → actors
* Edges → co-occurrence
* Features → keywords
* Classes → categories

## ⚠️ Properties

* Noisy + heterophilic

---

# 🌍 WikipediaNetwork (Chameleon, Squirrel)

## 📊 Stats

| Dataset   | Nodes | Edges | Features | Classes | Homophily |
| --------- | ----- | ----- | -------- | ------- | --------- |
| Chameleon | 2277  | ~36k  | 2325     | 5       | 0.23      |
| Squirrel  | 5201  | ~217k | 2089     | 5       | 0.22      |

## 🧠 Meaning

* Nodes → Wikipedia pages
* Edges → hyperlinks
* Features → TF-IDF

## ⚠️ Properties

* Strong heterophily
* Hard benchmarks

---

# 📚 Planetoid Datasets (Cora, Citeseer, Pubmed)

## 📊 Stats

| Dataset  | Nodes | Edges | Features | Classes | Homophily |
| -------- | ----- | ----- | -------- | ------- | --------- |
| Cora     | 2708  | 10556 | 1433     | 7       | 0.81      |
| Citeseer | 3327  | 9104  | 3703     | 6       | 0.74      |
| Pubmed   | 19717 | 88648 | 500      | 3       | 0.80      |

## 🧠 Meaning

* Nodes → research papers
* Edges → citations
* Features → bag-of-words

## ⚠️ Properties

* Strong **homophily**
* Easy for GCN

---

# 📚 Cora Full

## 📊 Stats

| Nodes | Edges | Features | Classes | Homophily |
| ----- | ----- | -------- | ------- | --------- |
| 19793 | ~126k | 8710     | 70      | ~0.60     |

## 🧠 Meaning

* Larger citation graph
* Fine-grained classification

---

# 🔥 Summary

| Type                         | Datasets                                     |
| ---------------------------- | -------------------------------------------- |
| Strong Heterophily (h < 0.3) | Texas, Wisconsin, Actor, Squirrel, Chameleon |
| Moderate                     | Cornell                                      |
| Strong Homophily (h > 0.7)   | Cora, Citeseer, Pubmed                       |
| Mixed                        | Cora Full                                    |

---

# 🎯 Key Insight

* **GCN works well when h is high**
* **Fails when h is low**
* Models like:

  * MultiHop
  * MixHop
  * H2GCN

👉 are designed for **low homophily graphs**

---

# ⚠️ Important Notes

* WebKB / Actor / Wikipedia → **10 splits**
* Planetoid → **1 split**
* Feature quality varies heavily

---

# 📌 References

* H2GCN Paper (NeurIPS 2020)
* Geom-GCN (Pei et al.)
* Planetoid (Kipf & Welling)

---
