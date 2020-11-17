import torch
import torch.nn as nn
from transformers import RobertaModel

class RobertaClassifier(nn.Module):
    def __init__(self):
        super(RobertaClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2
        self.roberta = nn.DataParallel(RobertaModel.from_pretrained('roberta-base'))

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

