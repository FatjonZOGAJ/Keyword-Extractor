import torch
import os

os.chdir('C:/Users/Fatjon/Documents/AppliedDeepLearning')                                                               # change root
root_path = os.getcwd()

USE_CUDA = True #False
if USE_CUDA:
    USE_CUDA = torch.cuda.is_available()

# training or testing
TRAIN    = False
# Used for training: creates a new model if true, else it takes the ones based on load_encoder & load_decoder
NEW_MODEL = False                                                                                                       # set this to true if you want to try training and have not downloaded the model from OneDrive

# evaluates on training dataset during training
EVALUATE_ON_TRAINING_WHILE_TRAINING = True

# evaluates on      test/validation dataset during training (if true validation score will be plotted, else the evaluation of the training data)
# defines if we evaluate on testing dataset during testing
EVALUATE_ON_TESTING = True

# 1.	Randomly evaluate some sentences
# 2.	Compare the loaded model on test dataset
# 3.	Evaluate User Input
# 4.	Compare different models on test dataset (needs to let the model train for multiple epochs / iterations and then set the respective MODEL_ITERATIONS_... variables)
TEST_EVALUATION_MODE = 1

# runs a demo for TEST_EVALUATION_MODE 1-3
DEMO_MODE = True


# ______________________________________________________
sentence_keyword_data = 'trnTweet-keyword'                                                                              # train data
sentence_keyword_data_test = 'testTweet-keyword'                                                                        #  test data

# which model to load
MODEL_ITERATIONS_VERSION = '195000'                                                                                     # longest trained: 195000 (best correct prediction percentage: 180000)
date_folder = '2019-12-18-0349'

# used for MODEL_COMPARISONS_EVALUATION = 4
MODEL_ITERATIONS_START = 185000
MODEL_ITERATIONS_END = 195000
MODEL_ITERATIONS_STEP =  5000

# Load parameters__________________________________
pre_text, post_text = sentence_keyword_data.split('-')                                                                  # training data loading
pre_test_text, post_test_text = sentence_keyword_data_test.split('-')                                                   #  testing data loading

encoder_load_path = root_path + '/models/' + date_folder + '/' + MODEL_ITERATIONS_VERSION + 'encoder.pt'
decoder_load_path = root_path + '/models/' + date_folder + '/' + MODEL_ITERATIONS_VERSION + 'decoder.pt'

# General parameters________________________________

RANDOM_EVALUATION_AMOUNT = 20

SOS_token = 0   # Start of Sentence has index 0
EOS_token = 1   # End   of Sentence has index 1

# training analysis + visualization
N_EPOCHS =    1000#0  #200000
N_SAVE_EVERY = 100#0    #5000
N_PLOT_EVERY = 100#0    #5000

#
N_TEST_EVALUATION_EPOCHS = 300
N_TEST_EVALUATION_SAVE_EVERY = 100
N_TEST_EVALUATION_PLOT_EVERY = 100



# Network parameters__________________________________
LEARNING_RATE = 0.01
# determines how often the actual values is passed to the Decoder instead of the predicted one
teacher_forcing_ratio = 0.5

optimizer_conf =  'sgd' #'adam'  # sgd works better with 195000


# amount of hidden states
hidden_size = 256
MAX_LENGTH = 512


gradient_clipping = False
gradient_clipping_value = 5