import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("flickr8k/captions.txt")
    mask = np.random.rand(len(df)) < 0.8
    df_train = df[mask]
    df_test = df[~mask]
    df_train.to_csv("flickr8k/captions_train.txt")
    df_test.to_csv("flickr8k/captions_test.txt")

if __name__ == "__main__":
    main()
