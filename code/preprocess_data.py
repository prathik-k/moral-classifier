from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
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

from getData import getDataset


class AITADataset(Dataset):

	def __init__(self, posts, targets, tokenizer, max_len):

		self.posts = posts
		self.targets = targets
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):
		return len(self.reviews)

	def __getitem__(self, item):

		post = str(self.posts[item])
		target = self.targets[item]
		encoding = self.tokenizer.encode_plus(
		add_special_tokens=True,
		max_length=self.max_len,
		return_token_type_ids=False,
		pad_to_max_length=True,
		return_attention_mask=True,
		return_tensors='pt')

		return {
		'post_text': post,
		'input_ids': encoding['input_ids'].flatten(),
		'attention_mask': encoding['attention_mask'].flatten(),
		'targets': torch.tensor(target, dtype=torch.long)
	}

if __name__=="__main__":
	def create_data_loader(df, tokenizer, max_len, batch_size):
		ds = AITADataset(
		posts=df.body.to_numpy(),
		targets=df.verdict.to_numpy(),
		tokenizer=tokenizer,
		max_len=max_len
		)
		return DataLoader(ds,batch_size=batch_size,num_workers=4)


	RANDOM_SEED = 42
	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	BATCH_SIZE = 16
	PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
	tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

	aita_data = getDataset()

	MAX_LEN	= aita_data.body.str.len().max()

	df_train, df_test = train_test_split(aita_data,test_size=0.1,random_state=RANDOM_SEED)
	df_val, df_test = train_test_split(df_test,test_size=0.5,random_state=RANDOM_SEED)

	train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
	val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
	test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

	torch.save(train_data_loader, '../dataloaders/train_dataloader.pth')
	torch.save(val_data_loader, '../dataloaders/val_dataloader.pth')
	torch.save(test_data_loader, '../dataloaders/test_dataloader.pth')

	print("Dataloaders created")
