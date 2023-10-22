import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for Grid Search")
    parser.add_argument('--learning-rates', nargs='+', type=float, default=[1e-2, 1e-4, 1e-6])
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--epochs', nargs='+', type=int, default=[50, 100, 150])
    args = parser.parse_args()
    return args  
