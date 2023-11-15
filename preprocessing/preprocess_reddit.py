import csv

import pandas as pd

FILE = "../data/reddit/raw/reddit_raw.csv"
CSV_NAME = "../data/reddit/raw/reddit.csv"
TRAIN_NAME = "../data/reddit/train_test.csv"
DEV_NAME = "../data/reddit/dev.csv"


def write_to_csv(file, csv_name, header):
    """
    Create dataset with columns ["example", "label", "conv"] based on orginal data frame. 
    """

    hate_counter = 0
    no_hate_counter = 0

    with open(csv_name, "w", encoding="UTF8", newline="") as data_csv:
        writer = csv.writer(data_csv, delimiter="\t")
        writer.writerow(header)

        df = pd.read_csv(file)
        for i, (text, hate_speech_idx) in enumerate(
            zip(df["text"], df["hate_speech_idx"])
        ):

            # get labels
            if pd.isna(hate_speech_idx):
                hate_speech_idx = []
            else:
                hate_speech_idx = (
                    hate_speech_idx.replace("[", "").replace("]", "").split(",")
                )
                hate_speech_idx = [int(x) for x in hate_speech_idx]

            # get examples
            for idx, row in enumerate(text.split("\n")[:-1]):
                example = row[2:].strip()

                if example == "":
                    continue

                if (idx + 1) in hate_speech_idx:
                    label = "hate"
                    hate_counter += 1

                else:
                    label = "noHate"
                    no_hate_counter += 1

                # write example to file
                writer.writerow([example, label, i])

    return hate_counter / (hate_counter + no_hate_counter)


def do_dev_train_data(source_data, dev_dir, train_dir, header, hate_ratio):
    """
    Generate a train and development set randomly (ratio 90/10) from a data 
    frame and write a new data frame.
    """

    hate_counter = 0
    no_hate_counter = 0

    with open(dev_dir, "w", encoding="UTF8", newline="") as dev_csv:
        dev_writer = csv.writer(dev_csv, delimiter="\t")
        dev_writer.writerow(header)

        with open(train_dir, "w", encoding="UTF8", newline="") as train_csv:
            train_writer = csv.writer(train_csv, delimiter="\t")
            train_writer.writerow(header)

            df = pd.read_csv(source_data, sep="\t")

            # shuffle data
            df = df.sample(frac=1)

            for example, label, c in zip(df["example"], df["label"], df["conv"]):

                if label == "hate":
                    hate_counter += 1

                    if hate_counter <= int(0.1 * len(df) * hate_ratio):
                        dev_writer.writerow([example, label, c])
                    else:
                        train_writer.writerow([example, label, c])

                else:
                    no_hate_counter += 1

                    if no_hate_counter <= (0.1 * len(df) * (1 - hate_ratio)) + 1:
                        dev_writer.writerow([example, label, c])
                    else:
                        train_writer.writerow([example, label, c])


if __name__ == "__main__":

    header = ["example", "label", "conv"]

    hate_ratio = write_to_csv(FILE, CSV_NAME, header)

    do_dev_train_data(CSV_NAME, DEV_NAME, TRAIN_NAME, header, hate_ratio)
