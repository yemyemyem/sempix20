#!/usr/bin/env python3
import pandas as pd
import random

def main():
    captions = pd.read_csv("flickr8k/captions.txt")
    sample_df = captions.sample(n=10)
    sample_df.to_csv("sample.txt")
    
if __name__ == "__main__":
    main()
