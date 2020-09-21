#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

root = Path("../../data")

def main():
    df = pd.read_csv(root / "flickr8k/captions.txt")
    df = df.sample(frac=1).reset_index(drop=True)   

    mask = np.random.rand(len(df)) < 0.8
    df_train = df[mask]
    df_test = df[~mask]

    mask = np.random.rand(len(df_test)) < 0.8
    df_val = df_test[mask]
    df_test = df_test[~mask]

    df_train.to_csv(root / "flickr8k/captions_train.txt", index=False)
    df_val.to_csv(root / "flickr8k/captions_val.txt", index=False)
    df_test.to_csv(root / "flickr8k/captions_test.txt", index=False)

if __name__ == "__main__":
    main()
