from flask import Flask, jsonify, request
from flask_cors import CORS

from settings_configuration import *
from utils import *

from train import *
from evaluate import *

from datetime import datetime
import random

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

PREDICTIONS = [
    {
        'input': 'foo bar test MAIN',
        'keywords': 'foo',
        'prediction': 'bar'
    }
]

print("CUDA : ", USE_CUDA)
print("TRAIN: ", TRAIN)

# input_lang = tweet Dictionary, output_lang = Keyword Dictionary
input_lang, output_lang, pairs = prepare_data(pre_text, post_text, False)
test_pairs = []

# indizes of only the training words
train_input_lang_size = input_lang.n_words

# use words/pairs from validation dataset
if EVALUATE_ON_TESTING:
    trash1, trash2, test_pairs = prepare_data(pre_test_text, post_test_text, False)

    for pair in test_pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

encoder1 = None
attn_decoder1 = None


print('Amount of words', input_lang.n_words)

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

    if USE_CUDA:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    train_interations(pairs, test_pairs, train_input_lang_size, input_lang, output_lang, encoder1, attn_decoder1, N_EPOCHS, n_save_every=N_SAVE_EVERY, n_plot_tick_every=N_PLOT_EVERY,
                      learning_rate=LEARNING_RATE, path='models/' + start_time)

    torch.save(encoder1, 'models/' + start_time + '/encoder.pt')
    torch.save(attn_decoder1, 'models/decoder.pt')

def start_evaluating():
    global encoder1, attn_decoder1, input_lang, output_lang, pairs, PREDICTIONS

    encoder1 = torch.load(encoder_load_path)
    attn_decoder1 = torch.load(decoder_load_path)

    if SERVER_MODE:
        PREDICTIONS = start_random_evaluation_print(pairs, test_pairs, encoder1, attn_decoder1, input_lang, output_lang,
                                      train_input_lang_size, PREDICTIONS)
    else:
        if DEMO_MODE or TEST_EVALUATION_MODE == 1:      # randomly evaluate some sentences
            start_random_evaluation_print(pairs, test_pairs, encoder1, attn_decoder1, input_lang, output_lang, train_input_lang_size)
        if DEMO_MODE or TEST_EVALUATION_MODE == 2:      # compare the loaded model on test dataset
            predicted_keywords_total, correctly_predicted_keywords_total = evaluation_iterations(encoder1, attn_decoder1, input_lang, output_lang, test_pairs, train_input_lang_size,
                                                                                                 N_TEST_EVALUATION_EPOCHS, N_TEST_EVALUATION_SAVE_EVERY, N_TEST_EVALUATION_PLOT_EVERY, path='/models/' + date_folder)
            print('Overall Test eval score = %.4f (%d/%d)' % (
            correctly_predicted_keywords_total / predicted_keywords_total, correctly_predicted_keywords_total,
            predicted_keywords_total))
        if DEMO_MODE or TEST_EVALUATION_MODE == 3:      # evaluating input
            evaluate_console_input(encoder1, attn_decoder1, input_lang, output_lang, train_input_lang_size)
        elif TEST_EVALUATION_MODE == 4:                 # compare different models on test dataset
            evaluation_iterations_multiple_models(input_lang, output_lang, test_pairs, train_input_lang_size,
                                                  N_TEST_EVALUATION_EPOCHS, N_TEST_EVALUATION_SAVE_EVERY, N_TEST_EVALUATION_PLOT_EVERY, path ='models/' + date_folder)



@app.route('/predictions', methods=['GET', 'POST', 'DELETE'])
def all_predictions():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        post_data = request.get_json()
        prediction = evaluate_input(post_data.get('input'), encoder1, attn_decoder1, input_lang, output_lang, train_input_lang_size)
        prediction.remove("<EOS>")
        PREDICTIONS.insert(0, {
            'input': post_data.get('input'),
            'keywords': post_data.get('keywords'),
            'prediction': ' '.join(prediction)
        })
        response_object['message'] = 'Prediction added!'
    elif request.method == 'DELETE':
        PREDICTIONS.clear()
        start_evaluating()
        response_object['message'] = 'Predictions reloaded!'
    else:
        response_object['predictions'] = PREDICTIONS
    return jsonify(response_object)


if __name__ == '__main__':
    if TRAIN:
        start_training()
    else:
        start_evaluating()


if SERVER_MODE:
    app.run()
