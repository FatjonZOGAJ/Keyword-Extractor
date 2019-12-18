from pip._vendor.distlib.compat import raw_input

from settings_configuration import * # use_cuda, TRAIN, pre_text, post_text, hidden_size, encoder_load_path, decoder_load_path, random_evaluation_amount, learning_rate
from utils import *       # TODO OWN import

from torch import optim
from train import *
from evaluate import *

from datetime import datetime
import random



#def main():
print("CUDA : ", use_cuda)
print("TRAIN: ", TRAIN)

# TODO pickling
input_lang, output_lang, pairs = prepare_data(pre_text, post_text, False)

# indizes of only the training words
train_input_lang_size = input_lang.n_words
# TODO train model with train + test EMBEDDING (without test data)
if EVALUATE_ON_TESTING:
    global test_pairs
    trash1, trash2, test_pairs = prepare_data(pre_test_text, post_test_text, False)

    # TODO fix bug so that they are only added if they exist in existing dictionary
    for pair in test_pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])


encoder1 = None
attn_decoder1 = None


print('Amount of words', input_lang.n_words)


def train_interations(encoder, decoder, n_epochs, n_save_every=1000, n_plot_tick_every=200, learning_rate=learning_rate, path =''):
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
            torch.save(encoder1,      save_path + 'encoder.pt')
            torch.save(attn_decoder1, save_path + 'decoder.pt')


def start_training():
    print("Started Training_______________________________________")
    global encoder1, attn_decoder1, input_lang, output_lang, pairs
    start_time = datetime.today().strftime('%Y-%m-%d-%H%M')
    os.makedirs('./models/' + start_time)

    if NEW_MODEL:
        encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
        attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
    else:
        encoder1 = torch.load(encoder_load_path)
        attn_decoder1 = torch.load(decoder_load_path)

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    train_interations(encoder1, attn_decoder1, N_EPOCHS, n_save_every=N_SAVE_EVERY, n_plot_tick_every=N_PLOT_EVERY,
                      learning_rate=learning_rate, path= 'models/' + start_time)

    torch.save(encoder1, 'models/' + start_time + '/encoder.pt')
    torch.save(attn_decoder1, 'models/decoder.pt')

def start_evaluating():
    global encoder1, attn_decoder1, input_lang, output_lang, pairs

    encoder1 = torch.load(encoder_load_path)
    attn_decoder1 = torch.load(decoder_load_path)

    if DEMO_MODE or TEST_EVALUATION_MODE == 1:      # randomly evaluate some sentences
        start_random_evaluation_print(pairs, test_pairs, encoder1, attn_decoder1, input_lang, output_lang, train_input_lang_size)
    if DEMO_MODE or TEST_EVALUATION_MODE == 2:    # compare the loaded model on test dataset
        predicted_keywords_total, correctly_predicted_keywords_total = evaluation_iterations(encoder1, attn_decoder1, input_lang, output_lang, test_pairs, train_input_lang_size,
                                                                                             N_TEST_EVALUATION_EPOCHS, N_TEST_EVALUATION_SAVE_EVERY, N_TEST_EVALUATION_PLOT_EVERY, path='/models/' + date_folder)
        print('Overall Test eval score = %.4f (%d/%d)' % (
        correctly_predicted_keywords_total / predicted_keywords_total, correctly_predicted_keywords_total,
        predicted_keywords_total))
    if DEMO_MODE or TEST_EVALUATION_MODE == 3:    # evaluating input
        evaluate_console_input(encoder1, attn_decoder1, input_lang, output_lang)
    elif TEST_EVALUATION_MODE == 4:                 # compare different models on test dataset
        evaluation_iterations_multiple_models(input_lang, output_lang, test_pairs, train_input_lang_size,
                                              N_TEST_EVALUATION_EPOCHS, N_TEST_EVALUATION_SAVE_EVERY, N_TEST_EVALUATION_PLOT_EVERY, path ='models/' + date_folder)


if TRAIN:
    start_training()
else:
    start_evaluating()
