import unittest
from utils import *
from settings_configuration import root_path
class TestClass(unittest.TestCase):

    def test_data_format(self):
        lines = open('%s/%s-%s.txt' % (root_path + '/data', 'testTweet', 'keyword'), encoding='utf-8').read().strip().split(
            '\n')

        for line in lines:
            sentence_keywords = line.split('\t')

            self.assertFalse(len(sentence_keywords) != 2)

            sentence = sentence_keywords[0].split(' ')
            keywords = sentence_keywords[1].split(' ')

            self.assertFalse(len(sentence) < 1 or len(keywords) < 1)

        self.assertTrue(1==1)

    def test_language_stop_words(self):
        lang = Lang('test')
        self.assertEqual('SOS', lang.index2word[0])
        self.assertEqual('EOS', lang.index2word[1])

    def test_language_size(self):
        lang = Lang('test')
        lang.add_sentence('This sentence has nine words plus <SOS> and <EOS>.')
        self.assertEqual(11, lang.n_words)

    def test_language_indizes(self):
        lang = Lang('test')
        lang.add_sentence('Our fifth word will be TEST')
        self.assertEqual(1 + 6, lang.word2index['TEST'])

    def test_language_word(self):
        lang = Lang('test')
        lang.add_sentence('TWO will be on index two')
        self.assertEqual(lang.index2word[2], 'TWO')

    def test_train_validation_no_indizes(self):
        lang = Lang('test')
        lang.add_sentence('We now have seven words')
        previous_size = lang.n_words
        pair = ['This is a sentence with a keyword', 'keyword']

        emptySentence = filter_sentence_containing_only_train_index(pair, lang, previous_size)
        self.assertEqual(0, len(emptySentence))

    def test_train_validation_one_index(self):
        lang = Lang('test')
        lang.add_sentence('We now have seven words')
        previous_size = lang.n_words
        pair = ['words was at the end.', 'keyword']

        oneWordSentence = filter_sentence_containing_only_train_index(pair, lang, previous_size)
        self.assertEqual(1, len(oneWordSentence.split(' ')))

    def test_train_validation_all_indizes(self):
        sentence = 'We now have seven words'
        lang = Lang('test')
        lang.add_sentence(sentence)
        previous_size = lang.n_words
        pair = [sentence, 'keyword']

        allWordSentence = filter_sentence_containing_only_train_index(pair, lang, previous_size)
        self.assertEqual(pair[0], allWordSentence)
        self.assertEqual(len(sentence.split(' ')), len(allWordSentence.split(' ')))

if __name__ == '__main__':
    unittest.main()