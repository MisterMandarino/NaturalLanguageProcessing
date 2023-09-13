from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from transformers import logging

logging.set_verbosity_error()

class Subjectivity_BERT(nn.Module):
    def __init__(self, dropout_prob=0):
        super(Subjectivity_BERT, self).__init__()

        # Bert Layer
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Fine-tuning Layer
        self.subj_out = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid() # sigmoid activation to squeeze values between [0, 1]

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, ids, mask):

        output = self.bert(input_ids=ids, attention_mask=mask)

        # Get the pooling layer
        last_hidden = output.pooler_output 

        # Compute subjectivity classification
        subjectivity = self.dropout(self.subj_out(last_hidden))

        return self.sigmoid(subjectivity)
    
class Polarity_BERT(nn.Module):
    def __init__(self, dropout_prob=0):
        super(Polarity_BERT, self).__init__()

        # Bert Layer
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Fine-tuning Layer
        self.pol_out = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, ids, mask):

        output = self.bert(input_ids=ids, attention_mask=mask)

        # Get the pooling layer
        last_hidden = output.pooler_output

        # Compute sentiment classification
        sentiment = self.dropout(self.pol_out(last_hidden))

        return self.sigmoid(sentiment)