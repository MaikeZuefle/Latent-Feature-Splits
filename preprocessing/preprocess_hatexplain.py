import csv
import json

from sklearn.model_selection import train_test_split

DATA_DIR = "data/hatexplain"
ORIGINAL_DATA = DATA_DIR + "/raw/dataset.json"
IDS = DATA_DIR + "/raw/post_id_divisions.json"

RANDOM_DIR = DATA_DIR
STANDARD_DIR = DATA_DIR + "/standard"
TRAIN_TEST_PATH = "train_test.csv"
TRAIN_PATH = "train.csv"
DEV_PATH = "dev.csv"
TEST_PATH = "test.csv"


def read_hatexplain(path):
    """
    Read data from json file.
    """
    with open(path, "r", encoding="UTF8") as f:
        data = json.load(f)

    hate, offensive, normal = [], [], []

    for dataset in data:

        text = " ".join(data[dataset]["post_tokens"])
        final_label = []

        for i in range(1, 4):
            final_label.append(data[dataset]["annotators"][i - 1]["label"])
        final_label_id = max(final_label, key=final_label.count)

        target = {x for i in data[dataset]["annotators"] for x in i["target"]}

        if final_label.count(final_label_id) == 1:  # undecided
            continue

        # rename 'hatespeech' label to 'hate' and 'normal' label to 'noHate'
        if final_label_id == "hatespeech":
            final_label_id = "hate"
        elif final_label_id == "normal":
            final_label_id = "noHate"

        data_dict = {
            "label": final_label_id,
            "target": target,
            "text": text,
            "dataset": dataset,
        }

        if final_label_id == "offensive":
            offensive.append(data_dict)
        elif final_label_id == "hate":
            hate.append(data_dict)
        elif final_label_id == "noHate":
            normal.append(data_dict)

    data_size = len(hate) + len(offensive) + len(normal)
    all_data = hate + offensive + normal

    hate_ratio = len(hate) / data_size
    offensive_ratio = len(offensive) / data_size
    normal_ratio = len(normal) / data_size

    return (
        hate,
        offensive,
        normal,
        hate_ratio,
        offensive_ratio,
        normal_ratio,
        data_size,
        all_data,
    )


def write_random__split_to_df(
    dev_path, train_test_path, hate, offensive, normal, data_size, header
):
    """
    Generate a train and development set randomly (ratio 90/10) from a data 
    frame and write a new data frame.
    """
    train_off, dev_off = train_test_split(offensive, test_size=0.1, random_state=42)
    train_hate, dev_hate = train_test_split(hate, test_size=0.1, random_state=42)
    train_norm, dev_norm = train_test_split(normal, test_size=0.1, random_state=42)

    assert (
        0.1 * data_size - 3
        < len(dev_off) + len(dev_hate) + len(dev_norm)
        < 0.1 * data_size + 3
    )
    assert (
        0.9 * data_size - 3
        < len(train_off) + len(train_hate) + len(train_norm)
        < 0.9 * data_size + 3
    )

    with open(dev_path, "w", encoding="UTF8", newline="") as dev_csv:
        dev_writer = csv.writer(dev_csv, delimiter="\t")
        dev_writer.writerow(header)
        for i in dev_off + dev_hate + dev_norm:
            dev_writer.writerow(
                [i["text"], i["label"], i["target"], i["dataset"].split("_")[-1]]
            )

    with open(train_test_path, "w", encoding="UTF8", newline="") as train_test_csv:
        train_test_writer = csv.writer(train_test_csv, delimiter="\t")
        train_test_writer.writerow(header)
        for i in train_off + train_hate + train_norm:
            train_test_writer.writerow(
                [i["text"], i["label"], i["target"], i["dataset"].split("_")[-1]]
            )


def get_ids(path):
    """
    Get the (standard) ids of train, validation and test set examples from the dataset.
    """
    with open(path, "r", encoding="UTF8") as f:
        ids = json.load(f)
    return ids["train"], ids["val"], ids["test"]


def write_standard_split_to_df(
    train_path, dev_path, test_path, all_data, data_size, header, ids
):
    """
    Write the standard data split to a data frame given the train/val/test ids.
    """
    dev_path = dev_path.replace(".csv", "") + "_standard.csv"

    train_ids, eval_ids, test_ids = get_ids(ids)
    test_set = [d for d in all_data if d["dataset"] in test_ids]
    train_set = [d for d in all_data if d["dataset"] in train_ids]
    dev_set = [d for d in all_data if d["dataset"] in eval_ids]

    with open(train_path, "w", encoding="UTF8", newline="") as train_csv:
        train_writer = csv.writer(train_csv, delimiter="\t")
        train_writer.writerow(header)
        for i in train_set:
            train_writer.writerow(
                [i["text"], i["label"], i["target"], i["dataset"].split("_")[-1]]
            )

    with open(dev_path, "w", encoding="UTF8", newline="") as dev_csv:
        dev_writer = csv.writer(dev_csv, delimiter="\t")
        dev_writer.writerow(header)
        for i in dev_set:
            dev_writer.writerow(
                [i["text"], i["label"], i["target"], i["dataset"].split("_")[-1]]
            )

    with open(test_path, "w", encoding="UTF8", newline="") as test_csv:
        test_writer = csv.writer(test_csv, delimiter="\t")
        test_writer.writerow(header)
        for i in test_set:
            test_writer.writerow(
                [i["text"], i["label"], i["target"], i["dataset"].split("_")[-1]]
            )


if __name__ == "__main__":
    (
        hate,
        offensive,
        normal,
        hate_ratio,
        offensive_ratio,
        normal_ratio,
        data_size,
        all_data,
    ) = read_hatexplain(ORIGINAL_DATA)

    header = ["example", "label", "target", "dataset"]

    write_random__split_to_df(
        RANDOM_DIR + DEV_PATH,
        RANDOM_DIR + TRAIN_TEST_PATH,
        hate,
        offensive,
        normal,
        data_size,
        header,
    )
    write_standard_split_to_df(
        STANDARD_DIR + TRAIN_PATH,
        STANDARD_DIR + DEV_PATH,
        STANDARD_DIR + TEST_PATH,
        all_data,
        data_size,
        header,
        IDS,
    )
