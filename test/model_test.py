from utils import *


encoder_test = EncoderRNN(10, 10,  1)
decoder_test = AttnDecoderRNN(10, 10, 0.1, 1)
print(encoder_test)
print(decoder_test)