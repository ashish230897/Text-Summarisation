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

rouge = RougeCalculator(stopwords=True, lang="en")

class LoadData(Dataset):
    """
    Using this since dataloader expects map-style dataset objects
    
    """
    
    def __init__(
        self, dataset, tokenizer, source_length, target_length):
        """
        Initializes a Dataset class

        Args:
            dataset (Dataset object): Input Dataset
            tokenizer (Tokenizer object): Transformer tokenizer
            source_length (int): Max length of source text
            target_length (int): Max length of target text
        """
        
        self.tokenizer = tokenizer
        self.data = dataset
        self.source_length = source_length
        self.summary_length = target_length
        self.target_text = self.data["summary"]
        self.source_text = self.data["dialogue"]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        """
        return input ids, attention masks and target ids
        
        """
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.__call__(
            [source_text],
            max_length=self.source_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        target = self.tokenizer.__call__(
            [target_text],
            max_length=self.summary_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def add_prefix(train, valid, test):
    # adding summarize in front of data
    from datasets import Dataset as Dataset_Hugging

    train_df = pd.DataFrame(train)
    train_df["dialogue"] = "summarize: " + train_df["dialogue"]
    train = Dataset_Hugging.from_pandas(train_df)

    valid_df = pd.DataFrame(valid)
    valid_df["dialogue"] = "summarize: " + valid_df["dialogue"]
    valid = Dataset_Hugging.from_pandas(valid_df)

    test_df = pd.DataFrame(test)
    test_df["dialogue"] = "summarize: " + test_df["dialogue"]
    test = Dataset_Hugging.from_pandas(test_df)

    return train, valid, test

def rouge_calc(preds, targets):
    rouge_1 = [rouge.rouge_n(summary=preds[i],references=targets[i],n=1) for i in range(len(preds))]
    rouge_2 = [rouge.rouge_n(summary=preds[i],references=targets[i],n=2) for i in range(len(preds))]
    rouge_l = [rouge.rouge_l(summary=preds[i],references=targets[i]) for i in range(len(preds))]

    return {"Rouge_1": np.array(rouge_1).mean(),
            "Rouge_2": np.array(rouge_2).mean(),
            "Rouge_L": np.array(rouge_l).mean()}

def evaluate(model, eval_dataloader, tokenizer, device):
    predictions = []
    ground_truths = []
    losses = []
    
    with torch.no_grad():
        for eval_batch in eval_dataloader:
            y = eval_batch['target_ids'].to(device, dtype = torch.long)
            ids = eval_batch['source_ids'].to(device, dtype = torch.long)
            mask = eval_batch['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5,  # there is a research paper for this
              #length_penalty=1.0,  # > 0 encourages to generate short sentences, < 0 to generate long sentences
              early_stopping=True  # stops beam search when number of beams sentences are generated per batch
              )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            
            predictions += preds
            ground_truths += target
            
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

            loss = model(
                input_ids=ids,
                attention_mask=mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )[0]
            losses.append(loss.item())

    scores = rouge_calc(predictions, ground_truths)
    avg_loss = sum(losses)/len(losses)
    print("Data score and losses are", scores, avg_loss)
    return avg_loss

def train_(model, train_loader, valid_loader, device, tokenizer, optimizer, parameters):
    steps = 0
    last_loss = 1000
    
    checkpoint_path = parameters["out_dir"] + "best_checkpoint/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    for epoch in range(parameters["epochs"]):
        print("Epoch: ", epoch)    
        for batch in train_loader:
            model.train()
            y = batch["target_ids"].to(device, dtype=torch.long)

            y_ids = y[:, :-1].contiguous()  # inputs passed to decoder, start token for decoder is pad token
            lm_labels = y[:, 1:].clone().detach()  # since it is lm task
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100  # so that loss can ignore the padded tokens

            ids = batch["source_ids"].to(device, dtype=torch.long)
            mask = batch["source_mask"].to(device, dtype=torch.long)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            loss = outputs[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps % 400 == 0: print("Train loss on {}th step is {}".format(steps, loss.item()))
            """
            if steps % 400 == 0:
                model.eval()
                print("Train loss on {}th step is {}".format(steps, loss.item()))
                loss = evaluate(model, valid_loader, tokenizer, device)
                if loss < last_loss: # save model parameters
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
                    last_loss = loss"""
            steps += 1
    
    """loss = evaluate(model, valid_loader, tokenizer)
    if loss < last_loss: # save model parameters
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        last_loss = loss"""
    
    # save the last model weights
    model.save_pretrained(parameters["out_dir"])
    tokenizer.save_pretrained(parameters["out_dir"])
    torch.save(optimizer.state_dict(), os.path.join(parameters["out_dir"], "optimizer.pt"))

def train_model(parameters, train_dataset, valid_dataset):
    cuda =  torch.cuda.is_available()
    device = torch.device("cuda") if cuda else torch.device("cpu")
    
    tokenizer = T5Tokenizer.from_pretrained(parameters["model"])    
    model = T5ForConditionalGeneration.from_pretrained(parameters["model"])  # has 60M parameters
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=parameters["lr"], weight_decay=parameters["wd"])
    
    train_obj = LoadData(
        train_dataset,
        tokenizer,
        parameters["max_source_length"],
        parameters["max_target_length"]
    )
    
    val_obj = LoadData(
        valid_dataset,
        tokenizer,
        parameters["max_source_length"],
        parameters["max_target_length"]
    )
    
    train_loader = DataLoader(train_obj, shuffle=True, batch_size=parameters["train_bs"])
    valid_loader = DataLoader(val_obj, shuffle=False, batch_size=parameters["val_bs"])
    
    num_training_steps = parameters["epochs"] * len(train_loader)
    print("Training steps are", num_training_steps)

    train_(model, train_loader, valid_loader, device, tokenizer, optimizer, parameters)