# Libraries

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class HateSpeech(Dataset):
    def __init__(
        self,
        root_dir,
        data_file,
        tokenizer,
        split="given",
        way_split=None,
        part="train",
    ):
        """Constructor method for hate speech data.
        Args:
            root_dir (str): path to the root directory
            data_file (str): filename of the csv file with examples labels
        """
        self.rootdir = root_dir
        self.data_file = data_file

        self.label2id = {"noHate": 0, "hate": 1, "offensive": 2}

        self.df = pd.read_csv(os.path.join(root_dir, data_file), sep="\t", header=0)

        # standard split (separate files given)
        if split == "given":
            self.df = pd.read_csv(os.path.join(root_dir, data_file), sep="\t", header=0)

        # train test split (dev set given)
        elif split == "train_test":
            if isinstance(way_split, int):
                random_seed = way_split

                train_data, test_data = train_test_split(
                    self.df,
                    random_state=random_seed,
                    test_size=1.0 / 9,
                    train_size=8.0 / 9,
                )

            else:
                raise NotImplementedError(
                    "Split with indices only possible for train, dev, test!"
                )

            if part == "train":
                self.df = train_data
            else:
                self.df = test_data

        # train dev test split
        elif split == "train_dev_test":
            if isinstance(way_split, int):
                random_seed = way_split

                train_test_data, dev_data = train_test_split(
                    self.df, random_state=random_seed, test_size=0.1, train_size=0.9
                )
                train_data, test_data = train_test_split(
                    train_test_data,
                    random_state=random_seed,
                    test_size=1.0 / 9,
                    train_size=8.0 / 9,
                )

            else:  # use index file
                train_idx, test_idx = self.get_df_from_idx_file(way_split)
                test_data = self.df.iloc[test_idx]

                train_data, dev_data = train_test_split(
                    self.df.iloc[train_idx],
                    random_state=42,
                    test_size=1.0 / 9,
                    train_size=8.0 / 9,
                )

            if part == "train":
                self.df = train_data
            elif part == "dev":
                self.df = dev_data
            else:
                self.df = test_data

        else:
            raise NotImplementedError(f"way of splitting {split} is not known!")

        self.labels = [self.label2id[label] for label in self.df["label"]]
        self.texts = [text for text in self.df["example"]]
        self.encoded_texts = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            for text in self.df["example"]
        ]

    def __len__(self):
        """Returns length based on the data frame length (= number of data points)
        Returns (int):  length based on the data frame length (= number of data points)
        """
        return len(self.encoded_texts)

    def get_batch_labels(self, idx):
        """ Fetch a batch of labels """
        return np.array(self.labels[idx])

    def get_batch_enocoded_texts(self, idx):
        """ Fetch a batch of inputs """
        return self.encoded_texts[idx]

    def get_batch_original_texts(self, idx):
        """ Fetch a batch of inputs """
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_encoded_texts = self.get_batch_enocoded_texts(idx)
        batch_original_texts = self.get_batch_original_texts(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_encoded_texts, batch_labels, idx, batch_original_texts

    def classes(self):
        return self.labels

    def get_df_from_idx_file(self, idx_file):
        def get_list(line):
            line = line.split("[")[-1].split("]")[0].strip().split(",")
            return [int(i) for i in line]

        with open(idx_file, "r", encoding="UTF8") as index_file:
            for line in index_file:
                indices = get_list(line)
                if line.startswith("test"):
                    test_indices = indices
                elif line.startswith("train"):
                    train_indices = indices

                elif line.startswith("dev"):
                    raise NotImplementedError("dev indices not supported")

        return train_indices, test_indices
