import logging

from datasets import HateSpeech


def split_data_into_train_dev_test(
    path, tokenizer, train_file, dev_file, test_file, split_seed, only_test
):
    """
    Splits the dataset into train, dev and test. Three cases are supported:
    1. Standard Split: Train, dev and test set is given as a file
    2. Random Split: Randomly split the data into train, dev and test, or
                     if dev set is given, split into train and test
    3. Split based on indices: Split the data based on a file providing the data
                                indices for the train and test set.
    """
    if test_file and split_seed:
        raise Exception(
            f"Test file {test_file} and the split {split_seed} is given. Unsure which one to use!"
        )

    # use standard split
    if train_file and dev_file and test_file:

        logging.info("Using standard split! (train, dev and test is given)")
        # load data
        train_hate_data = HateSpeech(
            root_dir=path, data_file=train_file, tokenizer=tokenizer, split="given"
        )
        dev_hate_data = HateSpeech(
            root_dir=path, data_file=dev_file, tokenizer=tokenizer, split="given"
        )
        test_hate_data = HateSpeech(
            root_dir=path, data_file=test_file, tokenizer=tokenizer, split="given"
        )

    # only testing - use given test set
    elif not split_seed and only_test and test_file:
        logging.info("Using standard split! (test is given)")
        test_hate_data = HateSpeech(
            root_dir=path, data_file=test_file, tokenizer=tokenizer, split="given"
        )
        train_hate_data, dev_hate_data = None, None

    # use random split with a random seed
    elif isinstance(split_seed, int):

        # randomly split train and test (dev file given)
        if dev_file:
            logging.info("Using random split for train and test, dev is given!")
            train_hate_data = HateSpeech(
                root_dir=path,
                data_file=train_file,
                tokenizer=tokenizer,
                split="train_test",
                way_split=split_seed,
                part="train",
            )
            dev_hate_data = HateSpeech(
                root_dir=path, data_file=dev_file, tokenizer=tokenizer
            )
            test_hate_data = HateSpeech(
                root_dir=path,
                data_file=train_file,
                tokenizer=tokenizer,
                split="train_test",
                way_split=split_seed,
                part="test",
            )

        # randomly split train, dev and test
        else:

            if test_file and split_seed:
                raise Exception(
                    f"Dev file {dev_file} and the split {split_seed} is given. Unsure which one to use!"
                )
            logging.info("Using random split for train, dev and test!")

            train_hate_data = HateSpeech(
                root_dir=path,
                data_file=train_file,
                tokenizer=tokenizer,
                split="train_dev_test",
                way_split=split_seed,
                part="train",
            )
            dev_hate_data = HateSpeech(
                root_dir=path,
                data_file=train_file,
                tokenizer=tokenizer,
                split="train_dev_test",
                way_split=split_seed,
                part="dev",
            )
            test_hate_data = HateSpeech(
                root_dir=path,
                data_file=train_file,
                tokenizer=tokenizer,
                split="train_dev_test",
                way_split=split_seed,
                part="test",
            )

    # using index files to split
    elif isinstance(split_seed, str):

        if split_seed.endswith("indices.txt"):
            logging.info(f"Using index file for split:{split_seed}")
            train_hate_data = HateSpeech(
                root_dir=path,
                data_file=train_file,
                tokenizer=tokenizer,
                split="train_dev_test",
                way_split=split_seed,
                part="train",
            )
            dev_hate_data = HateSpeech(
                root_dir=path,
                data_file=train_file,
                tokenizer=tokenizer,
                split="train_dev_test",
                way_split=split_seed,
                part="dev",
            )
            test_hate_data = HateSpeech(
                root_dir=path,
                data_file=train_file,
                tokenizer=tokenizer,
                split="train_dev_test",
                way_split=split_seed,
                part="test",
            )

    else:
        raise NotImplementedError(
            f"Split seed '{split_seed}' is not a valid argument! Try 'train_only', an integer or None!"
        )

    return train_hate_data, dev_hate_data, test_hate_data
