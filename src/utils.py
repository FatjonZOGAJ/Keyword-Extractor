import torch
import unicodedata
import re

from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


from settings_configuration import MAX_LENGTH, root_path, USE_CUDA, TRAIN, EOS_token, SOS_token
# Language class to create own embedding
# saves all the words with their amount of occurences
# creates two dictionaries
#                          to allow us to access a word based on an index
#                          to allow us to access an index based on a word

class Lang:

    def __init__(self, name):
        self.name       = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words    = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word]          = self.n_words
            self.word2count[word]          = 1
            self.index2word[self.n_words]  = word
            self.n_words                  += 1
        else:
            self.word2count[word] += 1

# preprocessing of input sentences
def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalize_string(s):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-z_AZ.!?]+", r" ", s)

    return s

# read data into Language / dictionary
# reverse theoretically allows us to match from a keyword to a sentence
# data must be of form: <sentence> \t <keyword>^+ \n
def read_langs(path, lang1, lang2, reverse=False):
    lines = open('%s/%s-%s.txt' % (path, lang1, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    #if reverse:
    #    pairs       = [list(reverse(p)) for p in pairs]
    #    input_lang  = Lang(lang2)
    #    output_lang = Lang(lang1)
    #else:
    input_lang  = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


# creates sentence and keyword Language class with the respective data
def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(root_path + '/data/',lang1, lang2, reverse)
    pairs                          = filter_pairs(pairs)

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    return input_lang, output_lang, pairs

# returns a List containing the indexes of every word
def indexes_from_sentence(lang, sentence):
    if TRAIN:
        return [lang.word2index[word] for word in sentence.split(' ')]
    else:
        indexes_list = []
        for word in sentence.split(' '):
            try:
                #print('WORD', word, 'Index', lang.word2index[word])                                                         #TODO: CHANGED ORIGINAL
                indexes_list.append(lang.word2index[word])
            except KeyError:
                print('Word not yet in dictionary:', word)
        return indexes_list
        #return [lang.word2index[word] for word in sentence.split(' ')]

# wraps the Tensor in a Variable to allow us to compute gradients more easily
# Variables are able to remember graph state -> automatic calculation of backwards gradients
def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)

    var = Variable(torch.LongTensor(indexes).view(-1, 1))

    if USE_CUDA:
        var = var.cuda()

    return var

def variables_from_pair(input_lang, output_lang, pair):
    input_variable  = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])

    return (input_variable, target_variable)

# removes the words that are not within input_lang dictionary
def filter_sentence_containing_only_train_index(pair, input_lang, train_input_lang_size):
    filtered_sentence = []
    for word in pair[0].split(' '):
        try:
            if input_lang.word2index[word] < train_input_lang_size:
                filtered_sentence.append(word)
        except KeyError:        # word that has no index
            pass
    return ' '.join(filtered_sentence)

#_______________________________________________________________________________________________________________________
# Models
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size= input_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding   = nn.Embedding(input_size, hidden_size)
        self.gru         = nn.GRU(hidden_size, hidden_size)
        #self.gru         = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input, hidden):
        embedded       = self.embedding(input).view(1, 1, -1)
        # output         = embedded
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def init_hidden(self):
        #hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        hidden = Variable(torch.zeros(1, 1, self.hidden_size))


        if USE_CUDA:
            hidden = hidden.cuda()

        return hidden

class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(output_size, hidden_size)
        self.gru         = nn.GRU(hidden_size, hidden_size)
        self.out         = nn.Linear(hidden_size, output_size)
        self.softmax     = nn.LogSoftmax(dim=1)                                                                         #TODO: Set to output size?

    def forward(self, input, hidden):
        output         = self.embedding(input).view(1, 1, -1)
        output         = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output         = self.softmax(self.out(output[0]))

        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        if USE_CUDA:
            return result.cuda()
        else:
            return result

class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, n_layers=1):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.dropout_p    = dropout_p
        self.max_length   = max_length
        self.n_layers     = n_layers
        self.attn         = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout      = nn.Dropout(self.dropout_p)

        # Layers
        self.embedding    = nn.Embedding(self.output_size, self.hidden_size)
        # TODO: LSTM
        self.gru          = nn.GRU(self.hidden_size, self.hidden_size)
        #self.gru          = nn.GRU(self.hidden_size, self.hidden_size, n_layers, dropout=dropout_p)

        self.out          = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded       = self.embedding(input).view(1, 1, -1)
        embedded       = self.dropout(embedded)
        attn_weights   = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied   = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output         = torch.cat((embedded[0], attn_applied[0]), 1)
        output         = self.attn_combine(output).unsqueeze(0)
        output         = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output         = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        if USE_CUDA:
            return result.cuda()
        else:
            return result


# Optional: BahdanauAttnDecoderRNN

# ______________________________________________________________________________________________________________________
# Helper / utils
import time
import math

def as_minutes(s):
    m  = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s   = now - since
    es  = s / (percent)
    rs  = es - s

    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
# matplotlib inline

def save_plot(points, percentages, path):
    #plt.figure()
    fig, ax1 = plt.subplots()
    color = 'tab:blue'

    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    if points != None:
        ax1.yaxis.set_major_locator(loc)

        ax1.set_ylabel('loss', color=color)
        ax1.plot(points, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        start, end = ax1.get_ylim()
        ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))

    if percentages!= None:
        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.set_ylabel('correctly predicted', color=color)
        ax2.plot(percentages, color=color)
        ax2.yaxis.set_major_locator(loc)
        ax2.tick_params(axis='y', labelcolor=color)

        start, end = ax2.get_ylim()
        ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))
    # plt.plot(points)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(path)

def model_comparison_save_plot(percentages, path, x_begin, x_end, x_step):
    x_values = np.arange(x_begin, x_end, x_step)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    #ax2 = ax1.twinx()
    ax1.set_ylabel('correctly predicted', color=color)
    plt.xlabel('model iteration version')
    plt.plot(x_values, percentages, color=color)
    plt.xticks(x_values)
    plt.ylim(0, 1)
    #ax1.yaxis.set_major_locator(loc)
    ax1.tick_params(axis='y', labelcolor=color)

#    start, end = ax1.get_ylim()
#    ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))
    # plt.plot(points)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(path)