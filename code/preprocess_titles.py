from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import os
import requests
from tqdm import tqdm

def getDataset():
    url = "https://www.dropbox.com/s/qjmj4wq9ywz5tb7/clean_data.csv?dl=1"
    fname = "temp_data.csv"

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fname, 'wb') as f:
            pbar = tqdm(total=int(r.headers['Content-Length']))
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    aita_data = pd.read_csv("temp_data.csv")
    os.remove("temp_data.csv")    
    return aita_data


def get_ids_and_attn(title, tokenizer, batch_size):	

	input_ids,attention_masks = [],[]
	
	for i,post in enumerate(title):
		if not(isinstance(post,str)):
			print(i,type(post))
		encoded_post = tokenizer.encode_plus(
            text=post,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
			max_length=512,
            pad_to_max_length=True,         # Pad sentence to max length
            return_attention_mask=True      # Return attention mask
            )
		input_ids.append(encoded_post.get('input_ids'))
		attention_masks.append((encoded_post.get('attention_mask')))
	input_ids,attention_masks = torch.tensor(input_ids),torch.tensor(attention_masks)
	return input_ids,attention_masks

def preprocess_text(aita_data):
	aita_data["title"].str.lower()
	aita_data["title"].str.replace(r'\\n',' ', regex=True) 
	aita_data["title"].str.replace(r"\'t", " not")
	aita_data["title"].str.replace(r'([\;\:\|«\n])', ' ')
	aita_data["title"].str.strip()

def create_dataloader(inputs,masks,labels,category,BATCH_SIZE):
	data = TensorDataset(inputs,masks,labels)
	sampler = RandomSampler(data) if category=='train' else SequentialSampler(data)
	dataloader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)
	return dataloader


if __name__=="__main__":
	RANDOM_SEED = 40
	loss_fn = nn.CrossEntropyLoss()
	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	BATCH_SIZE = 16
	PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
	tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

	aita_data = getDataset()
	preprocess_text(aita_data)

	df_train, df_test = train_test_split(aita_data,test_size=0.1,random_state=RANDOM_SEED)
	df_train, df_val = train_test_split(df_train,test_size=0.5,random_state=RANDOM_SEED)

	X_train,y_train,X_val,y_val = (df_train["title"].astype(str).tolist(),torch.tensor(df_train["verdict"].values),
									df_val["title"].astype(str).tolist(),torch.tensor(df_val["verdict"].values))
	train_labels = torch.tensor(y_train)
	val_labels = torch.tensor(y_val)

	train_inputs, train_masks = get_ids_and_attn(X_train, tokenizer, BATCH_SIZE)
	val_inputs, val_masks = get_ids_and_attn(X_val, tokenizer, BATCH_SIZE)
	print("Data tokenized")

	train_dataloader = create_dataloader(train_inputs,train_masks,train_labels,"train",BATCH_SIZE)
	val_dataloader = create_dataloader(val_inputs,val_masks,val_labels,"val",BATCH_SIZE)
	
	torch.save(train_dataloader, '../../dataloaders/train_dataloader.pth')
	torch.save(val_dataloader, '../../dataloaders/val_dataloader.pth')
	#torch.save(test_data_loader, '../dataloaders/test_dataloader.pth')

	print("Dataloaders created!")
