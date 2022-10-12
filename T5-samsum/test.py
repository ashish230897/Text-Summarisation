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
from torch.utils.data import Dataset

from utilities import *

def compute_rogue(model, device, tokenizer, parameters):
    dataset = load_dataset("samsum")

    train = dataset["train"]
    valid = dataset["validation"]
    test = dataset["test"]

    train = train.remove_columns(["id"])
    valid = valid.remove_columns(["id"])
    test = test.remove_columns(["id"])

    train, valid, test = add_prefix(train, valid, test)

    val_obj = LoadData(
            valid,
            tokenizer,
            parameters["max_source_length"],
            parameters["max_target_length"]
        )
    test_obj = LoadData(
            test,
            tokenizer,
            parameters["max_source_length"],
            parameters["max_target_length"]
        )

    valid_loader = DataLoader(val_obj, shuffle=False, batch_size=parameters["val_bs"])
    test_loader = DataLoader(test_obj, shuffle=False, batch_size=parameters["val_bs"])

    evaluate(model, valid_loader, tokenizer, device)
    evaluate(model, test_loader, tokenizer, device)

def inference(model, device, tokenizer, parameters):
    dataset = load_dataset("samsum")
    test = dataset["test"]
    test = test.remove_columns(["id"])
    
    sent = test[1]["dialogue"]  # taking an example sentence
    sent = "summarize: " + sent
    sent = " ".join(sent.split())
    
    source = tokenizer.__call__(
            [sent],
            max_length=parameters["max_source_length"],
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
    
    ids = source["input_ids"]
    mask = source["attention_mask"]
    
    model.eval()
    with torch.no_grad():
        ids = ids.to(device, dtype = torch.long)
        mask = mask.to(device, dtype = torch.long)
        
        generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5,  # there is a research paper for this
              #length_penalty=1.0,  # > 0 encourages to generate short sentences, < 0 to generate long sentences
              early_stopping=True  # stops beam search when number of beams sentences are generated per batch
              )
        
        preds = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        print("Input dialogue is: ", sent)
        print()
        print("Output summary is: ", preds)


def main():
    cuda =  torch.cuda.is_available()
    device = torch.device("cuda") if cuda else torch.device("cpu")

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

    tokenizer = T5Tokenizer.from_pretrained(parameters["out_dir"], do_lower_case=False)
    model = T5ForConditionalGeneration.from_pretrained(parameters["out_dir"])
    model.to(device)

    # compute rogue score on validation and test sets
    #compute_rogue(model, device, tokenizer, parameters)

    # perform inference on a sentence
    inference(model, device, tokenizer, parameters)






if __name__ == "__main__":
    main()