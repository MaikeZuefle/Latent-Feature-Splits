import logging
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from checkpoints import load_checkpoint
from classifiers import Classifier
from evaluation import evaluate
from plot_loss import plot_loss
from split_data import split_data_into_train_dev_test
from training import train
from transformers import AutoTokenizer
from transformers import logging as transformers_logging
from transformers.optimization import get_linear_schedule_with_warmup

# Cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers_logging.set_verbosity_error()

# deterministic behaviour
def set_seed(seed):
    """Sets all random seeds for deterministic behaviour.
    Args:
        seed (int): integer defining the random seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_cl(embeddings, model, data, only_test=False):
    """Main function that runs the classifier model, trains and evaluates it.
    """

    set_seed(model["random_seed"])

    # model parameter
    epochs = model["epochs"]
    batch_size = model["batch_size"]
    destination_path = model["destination_path"]
    early_stopping = model["early_stopping"]
    save_hiddens = model.get("save_hiddens", False)
    bottleneck = model.get("bottleneck", False)

    # check destination path and create directory
    if os.path.exists(destination_path):
        if len(os.listdir(destination_path)) > 1:
            raise FileExistsError(f"Model directory {destination_path} exists.")

    os.makedirs(destination_path)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(destination_path, "run.log")),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"{time.strftime('%H:%M:%S', time.localtime())} -- Welcome :)\n")
    logging.info(f"Device used: {device}\n")

    # pretrained models
    embeds = embeddings["embeddings"]
    tok = embeddings["tokenizer"]

    parameter_path = embeddings.get("load_from", None)

    # data
    path = data["path"]
    train_file = data.get("train_file", None)
    dev_file = data.get("dev_file", None)
    test_file = data.get("test_file", None)
    split_seed = data.get("split_seed", None)

    # model and tokenizer
    classifier = Classifier(embeds, bottleneck=bottleneck, data_path=path).to(device)
    if parameter_path:
        load_checkpoint(parameter_path, classifier)
    tokenizer = AutoTokenizer.from_pretrained(tok)

    # splitting data
    train_hate_data, dev_hate_data, test_hate_data = split_data_into_train_dev_test(
        path, tokenizer, train_file, dev_file, test_file, split_seed, only_test
    )

    # loading data
    train_dataloader = (
        torch.utils.data.DataLoader(
            train_hate_data, batch_size=batch_size, shuffle=True
        )
        if not only_test
        else None
    )
    val_dataloader = (
        torch.utils.data.DataLoader(dev_hate_data, batch_size=batch_size)
        if not only_test
        else None
    )
    test_dataloader = torch.utils.data.DataLoader(test_hate_data, batch_size=batch_size)

    # training
    if not only_test:
        optimizer = optim.AdamW(classifier.parameters(), lr=1e-6)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * epochs,
        )

        logging.info(
            f"{time.strftime('%H:%M:%S', time.localtime())} -- Starting Training!"
        )

        train(
            model=classifier,
            optimizer=optimizer,
            train_loader=train_dataloader,
            valid_loader=val_dataloader,
            num_epochs=epochs,
            file_path=destination_path,
            early_stopping=early_stopping,
            scheduler=scheduler,
            dataset=path,
        )

        logging.info(
            f"{time.strftime('%H:%M:%S', time.localtime())} -- Finished Training!\n"
        )

        # Evaluation
        plot_loss(destination_path, "metrics.pt")
        acc_checkpoint = os.path.join(destination_path, "model_best_acc.pt")

        logging.info("\nEvaluation Model with best ACC")
        best_model_acc = Classifier(embeds, bottleneck=bottleneck, data_path=path).to(
            device
        )

        load_checkpoint(acc_checkpoint, best_model_acc)
        evaluate(
            best_model_acc,
            test_dataloader,
            destination_path,
            save_hiddens=save_hiddens,
            bottleneck=bottleneck,
            data_path=path,
        )

    # only test the model
    else:
        if not parameter_path:
            logging.info(
                "Only testing and no checkpoint given at 'load_from' - testing the pretrained model!"
            )
        else:
            logging.info("Only testing, no training!")

        logging.info(f"{time.strftime('%H:%M:%S', time.localtime())} -- Loading Model")
        loaded_model = Classifier(embeds, bottleneck=bottleneck, data_path=path).to(
            device
        )

        if parameter_path:
            load_checkpoint(parameter_path, loaded_model)

        evaluate(
            loaded_model,
            test_dataloader,
            destination_path,
            save_hiddens=save_hiddens,
            bottleneck=bottleneck,
            data_path=path,
        )

    logging.info(f"{time.strftime('%H:%M:%S', time.localtime())} -- Goodbye :)")
