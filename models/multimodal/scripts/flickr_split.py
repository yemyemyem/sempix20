#!/usr/bin/env python3
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("flickr8k/captions.txt")
    df = df.sample(frac=1).reset_index(drop=True)   

    mask = np.random.rand(len(df)) < 0.8
    df_train = df[mask]
    df_test = df[~mask]

    df_train.to_csv("flickr8k/captions_train.txt", index=False)
    df_test.to_csv("flickr8k/captions_test.txt", index=False)

if __name__ == "__main__":
    main()