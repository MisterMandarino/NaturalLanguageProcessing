import nltk
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
from nltk.corpus import subjectivity
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# Global variables
import os

# Get the english stopwords from nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
POL_TRAIN_BATCH_SIZE = 16
POL_TEST_BATCH_SIZE = 8
SUB_TRAIN_BATCH_SIZE = 64
SUB_TEST_BATCH_SIZE = 32
PAD_TOKEN = 0
MAX_LEN = 122
STOP_WORDS = set(stopwords.words('english'))

def get_subjectivity_data():

    #nltk.download("subjectivity")
    from nltk.corpus import subjectivity

    subj = subjectivity.sents(categories='subj')
    obj = subjectivity.sents(categories='obj')

    subj_data = [(sent, 0) for sent in subj]
    obj_data = [(sent, 1) for sent in obj]

    dataset = subj_data + obj_data
    dataset = random.sample(dataset, len(dataset))

    return dataset

def get_polarity_data():

    #nltk.download('movie_reviews')
    #nltk.download('punkt')
    from nltk.corpus import movie_reviews

    neg_reviews = movie_reviews.paras(categories='neg')
    pos_reviews = movie_reviews.paras(categories='pos')

    neg_data = [(sent, 0) for sent in neg_reviews]
    pos_data = [(sent, 1) for sent in pos_reviews]

    dataset = neg_data + pos_data
    dataset = random.sample(dataset, len(dataset))

    return dataset

def get_tokenizer_max_length(tokenizer, dataset):
    token_lens = []
    for x in dataset:
        tokens = tokenizer.encode(x[0], max_length=512)
        token_lens.append(len(tokens))
    max_len = max(token_lens)
    return max_len

class Lang():
    def __init__(self, words, cutoff=0):
        self.word2id = self.vocabolary(words, cutoff=cutoff, unknown=True)
        self.id2word = {v:k for k, v in self.word2id.items()}

    def vocabolary(self, tokens, cutoff=0, unknown=True):
        vocab = {'PAD': 0}
        if unknown:
            vocab['UNK'] = 1
        count = Counter(tokens)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

class Subjectivity_Dataset(data.Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.samples = [sample for sample, label in data]
        self.labels = [label for sample, label in data]
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.samples[index]

        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            #return_tensors='pt'
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = self.labels[index]

        sample = {'label': label, 'ids': ids, 'mask': mask}
        return sample

    def __len__(self):
        return len(self.samples)
    
class Polarity_Dataset(data.Dataset):

    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.docs = [self.todoc(sample, max_len) for sample, label in data]
        self.labels = [label for sample, label in data]
        self.max_len = max_len

    def __getitem__(self, index):
        doc = self.docs[index]
        
        inputs = self.tokenizer.encode_plus(
            doc,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            #return_tensors='pt'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = self.labels[index]

        sample = {'label': label, 'ids': ids, 'mask': mask}
        return sample

    def __len__(self):
        return len(self.docs)
    
    def todoc(self, sentences, max_len):
        doc = []
        for sent in sentences:
            for token in sent:
                if ( len(doc) < (max_len - 2) )  and (token not in STOP_WORDS):
                    doc.append(token)
        return doc
    
def collate_fn(data):

    def merge_label(labels, max_len):
        padded_seqs = torch.FloatTensor(len(labels),max_len).fill_(PAD_TOKEN)
        for i, label in enumerate(labels):
            padded_seqs[i, :1] = label
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs

    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    label = merge_label(new_item['label'], MAX_LEN)

    ids = torch.LongTensor(new_item['ids'])
    mask = torch.LongTensor(new_item['mask'])

    label = label.to(device)
    ids = ids.to(device)
    mask = mask.to(device)

    sample = {'label': label, 'ids': ids, 'mask': mask}
    return sample

def polarity_collate_fn(data):

    def merge_label(labels, max_len):
        padded_seqs = torch.FloatTensor(len(labels),max_len).fill_(PAD_TOKEN)
        for i, label in enumerate(labels):
            padded_seqs[i, :1] = label
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs

    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    label = merge_label(new_item['label'], 512)

    ids = torch.LongTensor(new_item['ids'])
    mask = torch.LongTensor(new_item['mask'])

    label = label.to(device)
    ids = ids.to(device)
    mask = mask.to(device)

    sample = {'label': label, 'ids': ids, 'mask': mask}
    return sample

def remove_objective_sents(dataset, tokenizer, model):
    subjective_dataset = dataset.copy()

    for element in subjective_dataset:
        for sent in element[0]:
            
            inputs = tokenizer.encode_plus(
                sent,
                max_length=512,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt'
            )
            ids = inputs['input_ids'].to(device)
            mask = inputs['attention_mask'].to(device)
            objectivity = model(ids, mask)
            if torch.round(objectivity) == 1.: ## Obj sentence           
                element[0].remove(sent)
 
    return subjective_dataset