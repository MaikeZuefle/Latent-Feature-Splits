# Libraries

import logging

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):
    """
    Classifier Model for predictions, using pretrained BertForSequenceClassification or RobertaForSequenceClassification.
    """

    def __init__(
        self, embedding_model, bottleneck=False, data_path="reddit",
    ):
        """Init method for the classifier model.
        """
        super(Classifier, self).__init__()

        self.loss = nn.CrossEntropyLoss()

        if "reddit" in data_path:
            num_labels = 2
        elif "hatexplain" in data_path:
            num_labels = 3
        else:
            raise NotImplementedError(f"Unknown dataset '{data_path}'")

        self.config = AutoConfig.from_pretrained(
            embedding_model,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        self.bottleneck = bottleneck

        if "roberta" in embedding_model:
            self.encoder = RobertaForSequenceClassification.from_pretrained(
                embedding_model, config=self.config
            )

        elif "bert" in embedding_model or "BERT" in embedding_model:
            self.encoder = BertForSequenceClassification.from_pretrained(
                embedding_model, config=self.config
            )

        else:
            raise NotImplementedError(f"Architecture {embedding_model} not supported!")

        if self.bottleneck:
            self.linear1 = nn.Linear(self.config.hidden_size, self.bottleneck)
            self.linear2 = nn.Linear(self.bottleneck, self.config.hidden_size)
            self.get_logits = nn.Linear(self.config.hidden_size, self.config.num_labels)
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
            logging.info(
                f"Model used: {embedding_model} + a bottleneck with dimension {bottleneck}"
            )

        else:
            logging.info(f"Model used: {embedding_model}")

    def forward(self, input_ids=None, label=None):
        """Forward pass of the model of given text.

        Args:
            input_ids (input_ids): tensor containing the input_ids of the text to classify
            label (int): labels for the given text, default: None

        Returns:
            out (Attributes): Attributes wrapper class with information important for further processing
        """

        input_ids.to(device)

        if label != None:
            label.to(device)

        out = self.encoder(input_ids, labels=label)

        if self.bottleneck:  # make new head (only calculated with CLS)

            last_hiddens_cls = out["hidden_states"][-1][:, 0]
            last_hiddens_cls = self.dropout(last_hiddens_cls)

            bottleneck = self.linear1(last_hiddens_cls)
            linear2_output = self.linear2(bottleneck)

            linear2_output = torch.tanh(linear2_output)
            linear2_output = self.dropout(linear2_output)

            logits = self.get_logits(linear2_output)

        else:
            logits = out["logits"]
            bottleneck = None

        l = self.loss(logits, label)
        out = Attributes(output=out, loss=l, logits=logits, bottleneck=bottleneck)

        return out


class Attributes:
    """Wrapper class containing information about the classifer output.
    """

    def __init__(self, output=None, loss=None, logits=None, bottleneck=None):
        """Contains the loss, logits, attentions, all hidden states, 
        last hidden states and the last hidden cls state.
        """
        self.loss = loss
        self.logits = logits
        self.attentions = output["attentions"]

        self.all_hidden_states = output["hidden_states"]
        self.last_hidden_states = output["hidden_states"][-1]
        self.last_hidden_cls = output["hidden_states"][-1][:, 0]

        if bottleneck is not None:
            self.bottleneck_cls = bottleneck
