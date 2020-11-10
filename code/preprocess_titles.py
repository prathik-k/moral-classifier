from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
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

class AITADataset(Dataset):

	def __init__(self, titles, targets, tokenizer, max_len):

		self.titles = titles
		self.targets = targets
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):

		return len(self.reviews)

	def __getitem__(self, item):

		title = str(self.titles[item])
		target = self.targets[item]
		encoding = self.tokenizer.encode_plus(
		title,
		add_special_tokens=True,
		max_length=self.max_len,
		return_token_type_ids=False,
		pad_to_max_length=True,
		return_attention_mask=True,
		return_tensors='pt')

		return {
		'title_text': title,
		'input_ids': encoding['input_ids'].flatten(),
		'attention_mask': encoding['attention_mask'].flatten(),
		'targets': torch.tensor(target, dtype=torch.long)
	}


def create_data_loader(df, tokenizer, max_len, batch_size):

	ds = AITADataset(

	title=df.title.to_numpy(),
	targets=df.is_asshole.to_numpy(),
	tokenizer=tokenizer,
	max_len=max_len

	)

	return DataLoader(ds,batch_size=batch_size,num_workers=4)

cur_path = os.getcwd()
file_path_aita = os.path.relpath('..\\data\\aita_clean.csv', cur_path)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

aita_data = pd.read_csv(file_path_aita)

MAX_LEN	= aita_data.title.str.len().max()

df_train, df_test = train_test_split(aita_data,test_size=0.1,random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test,test_size=0.5,random_state=RANDOM_SEED)

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
