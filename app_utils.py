import traceback
import torch
from datasets import Dataset
import transformers
from transformers import (
  AdamW,
  T5Tokenizer,
  T5ForConditionalGeneration)
from torch.utils.data import DataLoader
import torch.nn as nn
import os

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib as mpl
from matplotlib.colors import Normalize, rgb2hex
import pandas as pd

parameters = {"model": "t5-small",  # model_type: t5-base/t5-large
    "train_bs": 8,  # training batch size
    "val_bs": 8,  # validation batch size
    "epochs": 5,  # number of training epochs
    "lr": 1e-4,  # learning rate
    "wd": 0.01,  # learning rate
    "max_source_length": 512,  # max length of source text
    "max_target_length": 80,  # max length of target text
    "SEED": 42,
    "out_dir": "./T5-samsum-alltrain/"}


def get_max_attn(c_atten):
    lst1 = []
    for target,i in enumerate(c_atten):
        lst2 = []
        for ipword in range(512):
            max_head = 0.0
            for layer in range(6):
                max_ = 0
                for head in range(8):
                    if(max_ < c_atten[target][layer][0][head][0][ipword].tolist()):
                        max_ = c_atten[target][layer][0][head][0][ipword].tolist()
                max_head += max_
            avg = max_head/6
            lst2.append(avg)
        lst1.append(lst2)
    return lst1

def colorize(attrs, cmap='PiYG'):

    cmap_bound = max([abs(attr) for attr in attrs])

    norm = Normalize(vmin=-cmap_bound, vmax=cmap_bound)

    cmap = mpl.cm.get_cmap(cmap)
    colors = list(map(lambda x: rgb2hex(cmap(norm(x))), attrs))

    return colors

def  hlstr(string, color='white'):
    return f"<mark style=background-color:{color}>{string} </mark>"

def color(word_scores, words):
    colors = colorize(word_scores)
    colored_input = []
    lis = list(map(hlstr, words, colors))
    
    return lis


def generate_(sent, model, tokenizer, device):
    
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
              #num_beams=2,
              repetition_penalty=2.5,  # there is a research paper for this
              #length_penalty=1.0,  # > 0 encourages to generate short sentences, < 0 to generate long sentences
              #early_stopping=True  # stops beam search when number of beams sentences are generated per batch
              output_attentions=True,
              return_dict_in_generate=True
              )
        


        preds = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    c_atten = generated_ids["cross_attentions"]
    target_input_attn = get_max_attn(c_atten)

    max_atten_per_ipword = []
    for ipword in range(512):
        max_ = 0.0
        for target in range(len(target_input_attn)):
            if(max_ <= target_input_attn[target][ipword]):
                max_ = target_input_attn[target][ipword]
        max_atten_per_ipword.append(max_)
    input_tokens = tokenizer.convert_ids_to_tokens(ids[0])
    input_tokens = [token for token in input_tokens if token != '<pad>']

    print("Generated summary is : ", preds)
    
    colored_inputs = color(max_atten_per_ipword , input_tokens)
    print("colored inputs: ", " ".join(colored_inputs[2:]))

    return preds, colored_inputs[2:]



if __name__ == "__main__":

    cuda =  torch.cuda.is_available()
    device = torch.device("cuda") if cuda else torch.device("cpu")
    
    tokenizer = T5Tokenizer.from_pretrained(parameters["out_dir"], do_lower_case=False)
    model = T5ForConditionalGeneration.from_pretrained(parameters["out_dir"])
    model.to(device)
    print ('Model loaded')

    text = "This is really helpful to point out!!"
    sent = "summarize: " + text
    sent = " ".join(sent.split())

    summary, text_colors = generate_(sent, model, tokenizer, device)