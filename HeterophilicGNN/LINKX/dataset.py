from torch_geometric.datasets import WebKB, WikipediaNetwork, Planetoid, Actor, CoraFull

def load_dataset(name):
    print("Inside load_dataset")
    name = name.lower()
    print("name:",name)

    if name in ["cornell", "texas", "wisconsin"]:
        dataset = WebKB(root=f"data/{name}", name=name.capitalize())

    elif name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root=f"data/{name}", name=name)

    elif name == "actor":
        dataset = Actor(root="data/actor")

    elif name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root=f"data/{name}", name=name.capitalize())

    elif name == "cora_full":
        dataset = CoraFull(root="data/cora_full")

    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    print("feature shape:", dataset[0].x.shape)
    return dataset[0], dataset.num_classes
