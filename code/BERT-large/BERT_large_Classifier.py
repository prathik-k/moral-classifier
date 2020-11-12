import torch
import torch.nn as nn
from transformers import BertModel

class BertLargeClassifier(nn.Module):
    def __init__(self):
        super(BertLargeClassifier, self).__init__()
        D_in, H, D_out = 1024, 50, 2
        self.bert = nn.DataParallel(BertModel.from_pretrained('bert-large-uncased'))

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(H, D_out)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

