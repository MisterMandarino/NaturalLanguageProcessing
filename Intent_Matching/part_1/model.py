import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, bidirectional=False, dropout=0.1):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size of the LSTM network
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (ouput size for intent class)
        # emb_size = word embedding size

        # set the bidirectionality lstm
        self.bidirectional = bidirectional

        # create the word embeddings
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # define the lstm module
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=self.bidirectional)

        if self.bidirectional:
            # double the input features since now, with the bidirectional LSTM we are accounting also for future sequences
            self.slot_out = nn.Linear(2*hid_size, out_slot)
            self.intent_out = nn.Linear(2*hid_size, out_int)
        else:
            # only past sequences are stored in the hidden cell
            self.slot_out = nn.Linear(hid_size, out_slot)
            self.intent_out = nn.Linear(hid_size, out_int)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, utterance, seq_lengths):

        utt_emb = self.embedding(utterance)
        utt_emb = utt_emb.permute(1,0,2) # we need seq len first -> seq_len X batch_size X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy())

        # Process the batch
        #packed_output, (last_hidden, cell) = self.dropout(self.utt_encoder(packed_input))
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)

        if self.bidirectional:
            # Get the first and the last hidden state
            last_hidden = torch.cat((last_hidden[0,:,:],last_hidden[-1,:,:]), dim=1)
        else:
            # Get the last hidden state
            last_hidden = last_hidden[-1,:,:]

        # Compute slot logits
        slots = self.dropout(self.slot_out(utt_encoded))
        # Compute intent logits
        intent = self.dropout(self.intent_out(last_hidden))

        #  (seq_len, batch size, calsses) to (batch_size, classes, seq_len)
        slots = slots.permute(1,2,0) # We need this for computing the loss

        return slots, intent
    
