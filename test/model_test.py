from utils import *


encoder_test = EncoderRNN(10, 2,  1)
decoder_test = AttnDecoderRNN(2, 2, 0.1, 4)
print(encoder_test)
print(decoder_test)