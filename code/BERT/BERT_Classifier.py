import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2
        self.bert = nn.DataParallel(BertModel.from_pretrained('bert-base-uncased'))

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

