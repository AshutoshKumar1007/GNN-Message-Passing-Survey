
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


feat_path = "data/Elliptic_data/elliptic_bitcoin_dataset/elliptic_txs_features.csv"
edge_path = "data/Elliptic_data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"
classes_path = "data/Elliptic_data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
#paths - ['feath_path' , 'edge_path', 'classes_path']
 
def load_data(paths):
    features = pd.read_csv(paths[0], header=None)
    edges = pd.read_csv(paths[1])
    classes = pd.read_csv(paths[2])


    # =========================
    # 3. Rename columns
    # =========================
    features.rename(columns={0: "txId", 1: "time_step"}, inplace=True)
    classes.rename(columns={"txId": "txId", "class": "label"}, inplace=True)

    # =========================
    # 4. Merge features + labels
    # =========================
    df = features.merge(classes, on="txId")

    # =========================
    # 5. Map labels to integers
    # =========================
    label_map = {
        "unknown": 0,
        "1": 1,   # illicit
        "2": 2    # licit
    }
    df["label"] = df["label"].astype(str).map(label_map)

    # =========================
    # 6. Create node index mapping
    # =========================
    tx_ids = df["txId"].values
    id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}

    # =========================
    # 7. Build feature matrix
    # =========================
    feature_cols = df.columns[2:-1]   # skip txId, time_step, label
    x = df[feature_cols].values

    # Normalize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x = torch.tensor(x, dtype=torch.float)

    # =========================
    # 8. Labels
    # =========================
    y = torch.tensor(df["label"].values, dtype=torch.long)

    # =========================
    # 9. Time steps
    # =========================
    time = torch.tensor(df["time_step"].values)

    # =========================
    # 10. Build edge_index
    # =========================
    edge_list = []

    for _, row in edges.iterrows():
        src = row["txId1"]
        dst = row["txId2"]

        if src in id_map and dst in id_map:
            edge_list.append([id_map[src], id_map[dst]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # =========================
    # 11. Create masks (temporal split)
    # =========================
    # (y != 0) is used to remove the unknown nodes from mask as they dont involve in loss calculation
    train_mask = (time <= 34) & (y != 0) # this is for static models -----------------------
    val_mask   = (time >= 35) & (time <= 39) & (y != 0) # this is for static models -----------------------
    test_mask  = (time >= 40) & (y != 0) # this is for static models -----------------------

    # =========================
    # 12. Build PyG Data object
    # =========================
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data

