import gc
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sumeval.metrics.rouge import RougeCalculator

import torch
from transformers import T5Tokenizer
import transformers
from transformers import T5ForConditionalGeneration
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
from datasets import load_dataset, Dataset
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utilities import *

print('Pytorch version: %s'  % torch.__version__)


def main():
    dataset = load_dataset("samsum")

    train = dataset["train"]
    valid = dataset["validation"]
    test = dataset["test"]

    train = train.remove_columns(["id"])
    valid = valid.remove_columns(["id"])
    test = test.remove_columns(["id"])

    train, valid, test = add_prefix(train, valid, test)

    print(len(train), len(valid), len(test))
    print("dataset has features: ", train)
    print("sample input and output is")
    print(train[0]["dialogue"])
    print(train[0]["summary"])

    parameters = {"model": "t5-small",  # model_type: t5-base/t5-large
    "train_bs": 8,  # training batch size
    "val_bs": 8,  # validation batch size
    "epochs": 5,  # number of training epochs
    "lr": 1e-4,  # learning rate
    "wd": 0.01,  # learning rate
    "max_source_length": 512,  # max length of source text
    "max_target_length": 80,  # max length of target text
    "SEED": 42,
    "out_dir": "./T5-samsum/"}

    train_model(parameters, train, valid)



if __name__ == "__main__":
    main()