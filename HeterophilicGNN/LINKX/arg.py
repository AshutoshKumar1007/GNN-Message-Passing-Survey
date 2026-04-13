import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=64)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    print(args.lr)
    print(args.epochs)
