import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_results_reddit(y_true, y_pred, y_probs):
    logging.info(
        "\n"
        + classification_report(
            y_true, y_pred, labels=[1, 0], target_names=["Hate", "No-Hate"], digits=4,
        )
    )
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
    roc_auc = roc_auc_score(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    logging.info(f"Acc: {round(acc, 4)}")
    logging.info(f"F1: {round(f1, 4)}")
    logging.info(f"Roc Auc: {round(roc_auc, 4)}")
    logging.info(f"Precision-Recall Auc: {round(pr_auc, 4)}")


def log_results_hatexplain(y_true, y_pred, y_probs_all):
    logging.info(
        "\n"
        + classification_report(
            y_true,
            y_pred,
            labels=[2, 1, 0],
            target_names=["Offensive", "Hate", "Normal"],
            digits=4,
            zero_division=0.0,
        )
    )
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        labels=[2, 1, 0],
        average="macro",
        zero_division=0.0,
    )
    nohatef1, hatef1, offensivef1 = f1_score(
        y_true=y_true, y_pred=y_pred, average=None, zero_division=0.0
    )
    roc_auc = roc_auc_score(
        F.one_hot(torch.tensor(y_true)),
        y_probs_all,
        multi_class="ovo",
        average="macro",
    )

    logging.info(f"Acc: {round(acc, 4)}")
    logging.info(f"F1: {round(f1, 4)}")
    logging.info(f"Roc Auc: {round(roc_auc, 4)}")
    logging.info(f"Hate F1: {round(hatef1, 4)}")
    logging.info(f"Offensive F1: {round(offensivef1, 4)}")
    logging.info(f"noHate F1: {round(nohatef1, 4)}")


def evaluate(
    model,
    test_loader,
    destination_path,
    save_hiddens=False,
    bottleneck=False,
    data_path="reddit",
):
    """Evaluation function for testing purposes (single models only).
    """

    y_pred = []
    y_true = []
    y_probs = []

    model.eval()
    if save_hiddens:
        hiddens_folder = f"{destination_path}/hiddens/"

        if bottleneck:
            hiddens_folder = hiddens_folder + f"_bottleneck_{bottleneck}"

        if not os.path.exists(hiddens_folder):
            os.makedirs(hiddens_folder)
        logging.info(f"Hidden representations are saved to '{hiddens_folder}'!")

        hiddens_file = open(f"{hiddens_folder}/hiddens.pkl", "wb")
        labels_file = open(f"{hiddens_folder}/hiddens_labels.pkl", "wb")

    with open(
        os.path.join(destination_path, "predictions.txt"), "w+", encoding="UTF8"
    ) as file:
        with torch.no_grad():

            for batch, batch_labels, idx, original_texts in test_loader:

                labels = batch_labels.to(device)
                input_ids = batch["input_ids"].squeeze(1).to(device)

                output = model(input_ids, label=labels)
                logits = output.logits

                if save_hiddens:

                    hidden_cls = (
                        output.bottleneck_cls if bottleneck else output.last_hidden_cls
                    )

                    for cls, i, l in zip(hidden_cls, idx, labels):
                        pickle.dump(cls, hiddens_file)
                        pickle.dump([i, l], labels_file)

                probs = F.softmax(logits, dim=1)

                trues = labels.tolist()
                preds = torch.argmax(logits, 1).tolist()

                for t, pred, true in zip(original_texts, preds, trues):
                    file.write(f"{t}\t{pred}\t{true}\n")

                y_probs.extend(probs.tolist())
                y_pred.extend(preds)
                y_true.extend(trues)

    if save_hiddens:
        hiddens_file.close()
        labels_file.close()

    y_probs_all = np.array(y_probs)
    y_probs = np.array([prob[1] for prob in y_probs])
    y_true = np.array(y_true)

    # print classification report
    logging.info("Classification Report:")

    if "reddit" in data_path:
        log_results_reddit(y_true, y_pred, y_probs)

    elif "hatexplain" in data_path:
        log_results_hatexplain(y_true, y_pred, y_probs_all)

    else:
        raise NotImplementedError(f"Unknown dataset {data_path}")
