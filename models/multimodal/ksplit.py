import pandas as pd
import random

def main():
    df = pd.read_json("flickr8k/karpathy_split.json")
    splits = {"train": [], "test": [], "val": []}

    for entry in df["images"]:
        path = entry["filename"]
        sentences = entry["sentences"]
        sentences = [x["tokens"] for x in sentences]
        split = entry["split"]

        for sentence in sentences:
            x = (path, sentence)
            splits[split].append(x)
 
    for split in splits.keys():
        print(split, len(splits[split]))

        with open(f"flickr8k/captions_k{split}.txt", "w") as f:
            f.write("image,caption\n")
            items = splits[split]
            random.shuffle(items)

            for x in items:
                f.write(x[0]+","+ " ".join(x[1]) + "\n")

if __name__ == "__main__":
    main()
