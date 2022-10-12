import gc
import random
import warnings
import numpy as np
import pandas as pd

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
from datasets import load_dataset, Dataset
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

    dialogue_lengths = [len(text.split()) for text in train["dialogue"]]
    summary_lengths = [len(text.split()) for text in train["summary"]]
    print("average length of dialogue is", sum(dialogue_lengths)/len(dialogue_lengths))
    print("average length of summary is", sum(summary_lengths)/len(summary_lengths))

    comment_words = ''
    stopwords = set(STOPWORDS)
    
    for text in train["dialogue"]:
        tokens = text.split()
        
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens)+" "

    for text in train["summary"]:
        tokens = text.split()
        
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_words)
                        
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()




if __name__ == "__main__":
    main()