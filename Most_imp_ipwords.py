from IPython.display import display, HTML
import matplotlib as mpl
from matplotlib.colors import Normalize, rgb2hex
import pandas as pd
from IPython.display import HTML
import tensorflow as tf

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

def predict(model,tokenizer,parameters,sent, device):
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
#               num_beams=2,
              repetition_penalty=2.5,  # there is a research paper for this
              #length_penalty=1.0,  # > 0 encourages to generate short sentences, < 0 to generate long sentences
#               early_stopping=True,  # stops beam search when number of beams sentences are generated per batch
              output_attentions=True,
              return_dict_in_generate=True
              )
        
        
        preds = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(preds)
    c_atten = generated_ids["cross_attentions"]
    
    return c_atten, generated_ids, ids

def predict_(model,tokenizer,parameters,sent, device):
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
#               num_beams=2,
              repetition_penalty=2.5,  # there is a research paper for this
              #length_penalty=1.0,  # > 0 encourages to generate short sentences, < 0 to generate long sentences
#               early_stopping=True,  # stops beam search when number of beams sentences are generated per batch
              output_attentions=True,
              return_dict_in_generate=True
              )
        
        
        preds = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(preds)
        enumerated_preds = tokenizer.convert_ids_to_tokens(generated_ids.sequences[0])
        print("enumerated predictions in token format: ")
        for i,token in enumerate(enumerated_preds):
            print(i,":",enumerated_preds[i])
    c_atten = generated_ids["cross_attentions"]
    
    return c_atten, generated_ids, ids


def colorize(attrs, cmap='PiYG'):

    cmap_bound = tf.reduce_max(tf.abs(attrs))

    norm = Normalize(vmin=-cmap_bound, vmax=cmap_bound)

    cmap = mpl.cm.get_cmap(cmap)
    colors = list(map(lambda x: rgb2hex(cmap(norm(x))), attrs))

    return colors

def  hlstr(string, color='white'):

    return f"<mark style=background-color:{color}>{string} </mark>"



def color_(max_atten_per_ipword , input_tokens):
    colors = colorize(max_atten_per_ipword)
    colored_input=[]
    display(HTML("".join(list(map(hlstr, input_tokens, colors)))))
   

def cross_atten(model,tokenizer,parameters,sent, device):
    
    c_atten, generated_ids, input_ids = predict(model,tokenizer,parameters,sent, device)
    
    target_input_attn = get_max_attn(c_atten)
    
    max_atten_per_ipword = []
    for ipword in range(512):
        max_ = 0.0
        for target in range(len(target_input_attn)):
            if(max_ <= target_input_attn[target][ipword]):
                max_ = target_input_attn[target][ipword]
        max_atten_per_ipword.append(max_)
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    input_tokens = [token for token in input_tokens if token != '<pad>']
    
    color_(max_atten_per_ipword , input_tokens)

def cross_atten_per_word(model,tokenizer,parameters,sent, device):
    
    c_atten, generated_ids, input_ids = predict_(model,tokenizer,parameters,sent, device)
    tarid = (int)(input("Enter the input id of the target word to be analysed :"))
    target_input_attn = get_max_attn(c_atten)
    
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    input_tokens = [token for token in input_tokens if token != '<pad>']
    
    color_(target_input_attn[tarid] , input_tokens)
    
    
    
    
