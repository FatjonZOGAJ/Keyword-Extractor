import torch
import torch.nn as nn
from torch import optim

from torch.autograd import Variable
from settings_configuration import USE_CUDA, MAX_LENGTH, teacher_forcing_ratio, gradient_clipping, \
    gradient_clipping_value, optimizer_conf, EVALUATE_ON_TRAINING_WHILE_TRAINING, LEARNING_RATE
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
    encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei]            = encoder_output[0][0]

    # prepare input & output
    decoder_input  = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input  = decoder_input.cuda() if USE_CUDA else decoder_input
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
            decoder_input  = decoder_input.cuda() if USE_CUDA else decoder_input
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



def train_interations(pairs, input_lang, output_lang, encoder, decoder, n_epochs, n_save_every=1000, n_plot_tick_every=200, learning_rate=LEARNING_RATE, path =''):
    start            = time.time()
    save_loss_total = 0
    plot_tick_loss_total = 0

    plot_predicted_keywords_total = 0
    save_predicted_keywords_total = 0
    plot_correctly_predicted_keywords_total = 0
    save_correctly_predicted_keywords_total = 0

    plot_losses = []
    plot_correct_predictions_percentages = []

    if optimizer_conf.lower() == 'sgd':
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    elif optimizer_conf.lower() == 'adam':
        encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
        decoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)

# if we train for a very long time this allocates too much memory unnecessarily
    # we evaluate if the predicted word is the right one
#    test_pairs = [random.choice(pairs) for i in range(n_epochs)]
    #training_pairs    = [variables_from_pair(input_lang, output_lang, random.choice(pairs)) for i in range(n_epochs)]
#    training_pairs    = [variables_from_pair(input_lang, output_lang, test_pairs[i]) for i in range(n_epochs)]

    criterion         = nn.NLLLoss()

    for iter in range(1, n_epochs + 1):
        pair = random.choice(pairs)
        training_pair = variables_from_pair(input_lang, output_lang, pair)
#        training_pair   = training_pairs[iter - 1]
        input_variable  = training_pair[0]
        target_variable = training_pair[1]
        decoded_words = []

        loss, decoded_words = train(
            input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang
        )

        # length is zero if teacher_forcing was used
        if EVALUATE_ON_TRAINING_WHILE_TRAINING & len(decoded_words) > 0:
            actual_test_pair_keywords = pair[1].split(' ')
#            actual_test_pair_keywords = test_pairs[iter-1][1].split(' ')
            plot_predicted_keywords_total += len(actual_test_pair_keywords)
            plot_correctly_predicted_keywords_total += len(list(set(actual_test_pair_keywords).intersection(decoded_words)))        # amount of words they have in common

        save_loss_total += loss
        plot_tick_loss_total += loss

        if iter % n_plot_tick_every == 0:
            plot_tick_loss_avg = plot_tick_loss_total / n_plot_tick_every
            plot_tick_loss_total = 0
            plot_losses.append(plot_tick_loss_avg)

            plot_correct_predictions_avg =  plot_correctly_predicted_keywords_total / plot_predicted_keywords_total
            plot_correct_predictions_percentages.append(plot_correct_predictions_avg)

            # for every save
            save_correctly_predicted_keywords_total += plot_correctly_predicted_keywords_total
            save_predicted_keywords_total += plot_predicted_keywords_total
            plot_predicted_keywords_total = 0
            plot_correctly_predicted_keywords_total = 0


        if iter % n_save_every == 0:
            save_loss_avg   = save_loss_total / n_save_every
            save_loss_total = 0

            save_correct_predictions_percentage_avg = save_correctly_predicted_keywords_total /save_predicted_keywords_total


            print('%s (%d %d%%) Loss = %.4f Train eval score = %.4f (%d/%d)' % (
                time_since(start, iter / n_epochs), iter, iter / n_epochs * 100, save_loss_avg, save_correct_predictions_percentage_avg, save_correctly_predicted_keywords_total, save_predicted_keywords_total
            ))

            save_predicted_keywords_total = 0
            save_correctly_predicted_keywords_total = 0

            save_path =  path + '/' + str(iter)

            save_plot(plot_losses, plot_correct_predictions_percentages, save_path + '.png')

            #save model after every print (useful for overnight training)
            torch.save(encoder, save_path + 'encoder.pt')
            torch.save(decoder, save_path + 'decoder.pt')

