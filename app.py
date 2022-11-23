# Dependencies
from flask import Flask, request, jsonify
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
import json

from flask_cors import CORS

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib as mpl
from matplotlib.colors import Normalize, rgb2hex
import pandas as pd

from app_utils import *

# Your API definition
app = Flask(__name__)
CORS(app)

model = None
device = None
tokenizer = None

parameters = {
    "model": "t5-small",  # model_type: t5-base/t5-large
    "train_bs": 8,  # training batch size
    "val_bs": 8,  # validation batch size
    "epochs": 5,  # number of training epochs
    "lr": 1e-4,  # learning rate
    "wd": 0.01,  # learning rate
    "max_source_length": 512,  # max length of source text
    "max_target_length": 80,  # max length of target text
    "SEED": 42,
    "out_dir": "./T5-samsum-alltrain/"
}


@app.route('/generate', methods=['POST'])
def generate():
    try:
        json_ = request.json
        print(json_)
        text = json_["text"]
        sent = "summarize: " + text
        sent = " ".join(sent.split())   

        summary, text_colors = generate_(sent, model, tokenizer, device)

        return jsonify({'summary': summary, 'attention': text_colors})

    except:

        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    cuda =  torch.cuda.is_available()
    device = torch.device("cuda") if cuda else torch.device("cpu")
    
    tokenizer = T5Tokenizer.from_pretrained(parameters["out_dir"], do_lower_case=False)
    model = T5ForConditionalGeneration.from_pretrained(parameters["out_dir"])
    model.to(device)
    print ('Model loaded')

    app.run(port=port, debug=True)