import torch.nn as nn
from transformers import BertModel
import copy
import numpy as np

def deleteLayers(model, layers_to_remove):
    oldModuleList = model.encoder.layer
    newModuleList = nn.ModuleList()

    # loop through 12 layers
    for i in range(0, 12):
        # if layer i+1 is in remove list, then remove it
        if i+1 in layers_to_remove:
            continue
        else:
            newModuleList.append(oldModuleList[i])

    copyModel = copy.deepcopy(model)
    copyModel.encoder.layer = newModuleList
    return copyModel

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False, D_out=3):
        super(BertClassifier, self).__init__()

        D_in, H = 768, 50
        # Bert layer
        self.bert = deleteLayers(BertModel.from_pretrained('bert-base-uncased'), [12])
        # Linear layer with ReLU
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        first_hidden_state_cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(first_hidden_state_cls)
        return logits