# Add functions or classes used for data loading and preprocessing
import torch.nn as nn
import torch
import json
import torch.optim as optim
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader

import os
device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0
MAX_LEN = 52

def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def get_data():

    tmp_train_raw = load_data(__file__ + "\\..\\dataset\\ATIS\\train.json")
    test_raw = load_data(__file__ + "\\..\\dataset\\ATIS\\test.json")

    # Firt we get the 10% of dataset, then we compute the percentage of these examples
    # on the training set which is around 11%
    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10)/(len(tmp_train_raw)),2)

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    Y = []
    X = []
    mini_Train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occure once only, we put them in training
            X.append(tmp_train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])

    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=portion, random_state=42, shuffle=True, stratify=Y)
    X_train.extend(mini_Train)
    train_raw = X_train
    dev_raw = X_dev

    return train_raw, dev_raw, test_raw

# we can't use the set structure because it doesn't preserve the ordering
def get_slots_n(corpus):
    slots = []
    for line in corpus:
        for x in line['slots'].split():
            if x not in slots:
                slots.append(x)
    return slots

def get_intents_n(corpus):
    intents = []
    for line in corpus:
        if line['intent'] not in intents:
            intents.append(line['intent'])
    return intents

# return the max token length for the bert tokenizer
def get_max_token_len(corpus, tokenizer):
    token_lens = []
    for x in corpus:
        tokens = tokenizer.encode(x['utterance'], max_length=512)
        token_lens.append(len(tokens))
    return max(token_lens)

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    
class IntentDataset(data.Dataset):

    def __init__(self, dataset, lang, tokenizer, max_len, unk='unk'):
        self.tokenizer = tokenizer
        self.utterances = []
        self.intents = []
        self.slots = []
        self.max_len = max_len
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):

        text = self.utterances[idx]

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

        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]

        sample = {'utterance': utt, 'slots': slots, 'intent': intent, 'ids': ids, 'mask': mask}
        return sample

    # Auxiliary methods

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    
def collate_fn(data):

    def merge(sequences, max_len):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'], MAX_LEN)
    y_slots, y_lengths = merge(new_item["slots"], MAX_LEN)
    intent = torch.LongTensor(new_item["intent"])

    ids = torch.LongTensor(new_item['ids'])
    mask = torch.LongTensor(new_item['mask'])

    src_utt = src_utt.to(device) # We load the Tensor on our seleceted device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    ids = ids.to(device)
    mask = mask.to(device)

    sample = {'utterance': src_utt, 'slots': y_slots, 'slots_len': y_lengths, 'intent': intent, 'ids': ids, 'mask': mask}
    return sample