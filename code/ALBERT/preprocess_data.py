from transformers import AlbertModel, AlbertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset, DataLoader, RandomSampler
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
import sys
import requests
from tqdm import tqdm
import pickle

sys.path.append("../../dataloaders/")

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


def get_ids_and_attn(body, tokenizer, batch_size):	

	input_ids,attention_masks = [],[]
	
	for i,post in enumerate(body):
		if not(isinstance(post,str)):
			print(i,type(post))
		encoded_post = tokenizer.encode_plus(
            text=post,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
			max_length=64,
            pad_to_max_length=True,         # Pad sentence to max length
            truncation=True,
            return_attention_mask=True      # Return attention mask
            )
		input_ids.append(encoded_post.get('input_ids'))
		attention_masks.append((encoded_post.get('attention_mask')))
	input_ids,attention_masks = torch.tensor(input_ids),torch.tensor(attention_masks)
	return input_ids,attention_masks

def preprocess_text(aita_data):
	aita_data["body"].str.lower()
	aita_data["body"].str.replace(r'\\n',' ', regex=True) 
	aita_data["body"].str.replace(r"\'t", " not")
	aita_data["body"].str.strip()

def create_dataloader(inputs,masks,labels,BATCH_SIZE):
	data = TensorDataset(inputs,masks,labels)
	sampler = RandomSampler(data) 
	dataloader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE, num_workers=22)
	return dataloader

def generate_dataloader(X,y,tokenizer,BATCH_SIZE,category):
	labels = torch.tensor(y)
	inputs,masks = get_ids_and_attn(X, tokenizer, BATCH_SIZE)
	dataloader = create_dataloader(inputs,masks,labels,BATCH_SIZE)
	filename = category+'_dataloader_'+str(BATCH_SIZE)+'.pth'	
	torch.save(dataloader, '../../../dataloaders/ALBERT/'+filename)

if __name__=="__main__":
	RANDOM_SEED = 40
	loss_fn = nn.CrossEntropyLoss()
	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	BATCH_SIZE = (64,32)
	PRE_TRAINED_MODEL_NAME = 'albert-base-v1'
	tokenizer = AlbertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
	aita_data = getDataset()
	preprocess_text(aita_data)

	df_train, df_test = train_test_split(aita_data,test_size=0.1,random_state=RANDOM_SEED)
	df_train, df_val = train_test_split(df_train,test_size=0.2,random_state=RANDOM_SEED)

	X_train,y_train,X_val,y_val = (df_train["body"].astype(str).tolist(),torch.tensor(df_train["verdict"].values),
									df_val["body"].astype(str).tolist(),torch.tensor(df_val["verdict"].values))
	X_test,y_test = (df_test["body"].astype(str).tolist(),torch.tensor(df_test["verdict"].values))

	data_dict = dict(X_train=X_train,y_train=y_train,X_val=X_val,y_val=y_val,X_test=X_test,y_test=y_test)

	with open('../../../dataloaders/all_data.pkl','wb') as file:
		pickle.dump(data_dict,file)

	for size in BATCH_SIZE:
		generate_dataloader(X_train,y_train,tokenizer,size,"train")
		print("Training dataloader created!")
		generate_dataloader(X_val,y_val,tokenizer,size,"val")
		print("Validation dataloader created!")
		generate_dataloader(X_test,y_test,tokenizer,size,"test")
		print("Testing dataloader created!")

	print("Dataloaders created!")