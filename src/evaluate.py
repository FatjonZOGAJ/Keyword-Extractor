from datetime import datetime

import torch
import torch.nn as nn
from pip._vendor.distlib.compat import raw_input
from torch import optim
import time
import random

from torch.autograd import Variable
from settings_configuration import *
from utils import *

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    input_variable  = variable_from_sentence(input_lang, sentence)
    input_length    = input_variable.size()[0]
    encoder_hidden  = encoder.init_hidden()
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    #encoder_outputs = encoder_ouputs.cuda() if use_cuda else encoder_ouputs                                            TODO: ORIGINAL VERSION HAD AN ERROR


    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei]            = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input      = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input      = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden     = encoder_hidden
    decoded_words      = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        decoder_attentions[di] = decoder_attention.data
        topv, topi             = decoder_output.data.topk(1)                                                            #TODO: within top k 10 ?
        ni                     = topi[0][0]

        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            #decoded_words.append(output_lang.index2word[ni])                                                           TODO: CHANGED
            decoded_words.append(output_lang.index2word[ni.item()])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def evaluation_iterations(encoder, decoder, input_lang, output_lang, pairs, train_input_lang_size, n_epochs, n_save_every=1000, n_plot_tick_every=200, path =''):
    print('\nSTARTING EVALUATION ITERATION____________________________________________________________________________' )

    start = time.time()
    max_prints = 3

    plot_predicted_keywords_total = 0
    plot_correctly_predicted_keywords_total = 0
    save_predicted_keywords_total = 0
    save_correctly_predicted_keywords_total = 0
    epoch_predicted_keywords_total = 0
    epoch_correctly_predicted_keywords_total = 0

    plot_correct_predictions_percentages = []


    for iter in range(1, n_epochs +1):
        pair = random.choice(pairs)
        filtered_sentence = filter_sentence_containing_only_train_index(pair, input_lang, train_input_lang_size)

        decoded_words, trash = evaluate(encoder, decoder, filtered_sentence, input_lang, output_lang)

        # length is zero if teacher_forcing was used
        if len(decoded_words) > 0:
            actual_test_pair_keywords = pair[1].split(' ')
            plot_predicted_keywords_total += len(actual_test_pair_keywords)
            plot_correctly_predicted_keywords_total += len(list(set(actual_test_pair_keywords).intersection(decoded_words)))    # amount of words they have in common

            if max_prints > 0 and random.random() > 0.1:
                print('\t Sentence:   ', pair[0])
                print('\t Keywords:   ', pair[1])
                print('\t Prediction: ', ' '.join(decoded_words[:-1]), '\n')
            max_prints += -1


        if iter % n_plot_tick_every == 0:
            plot_correct_predictions_avg = plot_correctly_predicted_keywords_total / plot_predicted_keywords_total
            plot_correct_predictions_percentages.append(plot_correct_predictions_avg)

            # for every save
            save_correctly_predicted_keywords_total += plot_correctly_predicted_keywords_total
            save_predicted_keywords_total += plot_predicted_keywords_total


            print('%s (%d %d%%) Test eval score = %.4f (%d/%d)' % (
                time_since(start, iter / n_epochs), iter, iter / n_epochs * 100,
                plot_correct_predictions_avg, plot_correctly_predicted_keywords_total,
                plot_predicted_keywords_total
            ))

            plot_predicted_keywords_total = 0
            plot_correctly_predicted_keywords_total = 0

        if iter % n_save_every == 0:
            max_prints = 3
            save_correct_predictions_percentage_avg = save_correctly_predicted_keywords_total /save_predicted_keywords_total

            epoch_predicted_keywords_total += save_predicted_keywords_total
            epoch_correctly_predicted_keywords_total += save_correctly_predicted_keywords_total
            save_predicted_keywords_total = 0
            save_correctly_predicted_keywords_total = 0

            save_path =  path + '/' + MODEL_ITERATIONS_VERSION + str(iter)
            # save_plot(None, plot_correct_predictions_percentages, save_path + '.png')

    return epoch_predicted_keywords_total, epoch_correctly_predicted_keywords_total

def evaluation_iterations_multiple_models(input_lang, output_lang, pairs, train_input_lang_size, n_epochs, n_save_every=1000, n_plot_tick_every=200, path =''):
    start_time = datetime.today().strftime('%Y-%m-%d-%H%M')

    plot_correct_predictions_percentages = []

    for model_version in range(MODEL_ITERATIONS_START, MODEL_ITERATIONS_END, MODEL_ITERATIONS_STEP):
        model_path = root_path + '/models/' + date_folder + '/' + str(model_version)
        print('MODEL_VERSION: ', model_path)
        encoder = torch.load(model_path + 'encoder.pt')
        decoder = torch.load(model_path + 'decoder.pt')

        predicted_keywords_total, correctly_predicted_keywords_total = evaluation_iterations(encoder, decoder, input_lang, output_lang, pairs, train_input_lang_size,
                                                                                             n_epochs, n_save_every, n_plot_tick_every,
                                                                                             path='/models/' + date_folder)
        plot_correct_predictions_percentages.append(correctly_predicted_keywords_total / predicted_keywords_total)

    model_comparison_save_plot(plot_correct_predictions_percentages, path + '/MODELCOMPARISON' + start_time + '.png', MODEL_ITERATIONS_START, MODEL_ITERATIONS_END, MODEL_ITERATIONS_STEP)

def start_random_evaluation_print(pairs, test_pairs, encoder, decoder, input_lang, output_lang, train_input_lang_size):
    print('\nSTARTING RANDOM EVALUATION____________________________________________________________________________' )
    i = 0
    while i < RANDOM_EVALUATION_AMOUNT:
        print(str(i) + ' ___________________')

        if EVALUATE_ON_TESTING:
            evaluate_randomly(test_pairs, encoder, decoder, input_lang, output_lang, train_input_lang_size)
        else:
            evaluate_randomly(pairs,      encoder, decoder, input_lang, output_lang, train_input_lang_size)

        i = i + 1


def evaluate_randomly(pairs, encoder, decoder, input_lang, output_lang, train_input_lang_size):

    pair = random.choice(pairs)
    filtered_sentence = pair[0]

    if EVALUATE_ON_TESTING:                                                                                             # if evaluating on test there may be words that have an index that exceeds the trained embedding size
        filtered_sentence = filter_sentence_containing_only_train_index(pair, input_lang, train_input_lang_size)

    prediction = output_evaluation(filtered_sentence, encoder, decoder, input_lang, output_lang)
    print('Actual Keywords    = ', pair[1])


def output_evaluation(input_sentence, encoder, decoder, input_lang, output_lang):
    output_words, attentions = evaluate(
        encoder, decoder, input_sentence, input_lang, output_lang
    )

    print("Input              = ", input_sentence)
    print("Predicted Keywords = ", ' '.join(output_words))
    return output_words



def evaluate_console_input(encoder, decoder, input_lang, output_lang):
    print('\nSTARTING CONSOLE INPUT EVALUATION____________________________________________________________________________' )

    while(True):
        try:
            inp = raw_input("Please enter input: ")
            inp = normalize_string(inp)
            output_evaluation(inp, encoder, decoder, input_lang, output_lang)
        except KeyError as e:
            print ('I got a KeyError - reason "%s"' % str(e))
