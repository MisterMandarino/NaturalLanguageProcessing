import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig

class IntentBERT(nn.Module):

    def __init__(self, out_slot, out_int):
        super(IntentBERT, self).__init__()

        # Bert Layer
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Fine-tuning Layer
        self.slot_out = nn.Linear(self.bert.config.hidden_size, out_slot)
        self.intent_out = nn.Linear(self.bert.config.hidden_size, out_int)

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, ids, mask):

        output = self.bert(input_ids=ids, attention_mask=mask)

        # Get the last hidden state
        last_hidden = output.last_hidden_state #slot filling
        pool_output = output.pooler_output #intent classification

        # Compute slot logits
        slots = self.dropout(self.slot_out(last_hidden))

        # Compute intent logits
        intent = self.dropout(self.intent_out(pool_output))

        # Slot size: seq_len, batch size, calsses
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len

        return slots, intent