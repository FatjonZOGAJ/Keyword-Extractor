import torch
import torch.nn as nn
from torch import optim

from torch.autograd import Variable
from settings_configuration import use_cuda, MAX_LENGTH, teacher_forcing_ratio, gradient_clipping, gradient_clipping_value
from utils import *# SOS_token, EOS_token,
import random


# one training cycle
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss            = 0

    # length of input (sentences/tweet) and target (keyword)
    input_length    = input_variable.size()[0]
    target_length   = target_variable.size()[0]

    # put words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei]            = encoder_output[0][0]

    # prepare input & output
    decoder_input  = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input  = decoder_input.cuda() if use_cuda else decoder_input
    # last hidden state from encoder is used for start of decoder
    decoder_hidden = encoder_hidden

    decoded_words      = []


    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            loss          += criterion(decoder_output, target_variable[di])
            # next target is next input
            decoder_input  = target_variable[di]
    else: # use own prediction as next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            # most likely word + index
            topv, topi     = decoder_output.data.topk(1)
            ni             = topi[0][0]
            decoder_input  = Variable(torch.LongTensor([[ni]]))
            decoder_input  = decoder_input.cuda() if use_cuda else decoder_input
            loss          += criterion(decoder_output, target_variable[di])#.item()     #TODO: https://discuss.pytorch.org/t/cuda-error-out-of-memory/28123/2

            # ni contains the index of the found word
            if ni == EOS_token:
                # TODO: EOS nicht returnen wegen Länge
                # decoded_words.append('<EOS>')
                break
            else:
                # decoded_words.append(output_lang.index2word[ni])                                                           TODO: CHANGED
                decoded_words.append(output_lang.index2word[ni.item()])

            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    # Gradient Clipping
    if gradient_clipping:
        torch.nn.utils.clip_grad_norm(encoder.parameters(), gradient_clipping_value)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), gradient_clipping_value)

    encoder_optimizer.step()
    decoder_optimizer.step()

    #return loss.data[0] / target_length                                                                                TODO: ORIGINAL CHANGED
    return loss.data / target_length, decoded_words
