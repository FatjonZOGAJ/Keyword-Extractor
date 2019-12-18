from settings_configuration import  root_path
import unittest

class DataTest(unittest.TestCase):

    def test_data_format(self):
        lines = open('%s/%s-%s.txt' % (root_path + '/data', 'tweet', 'keyword'), encoding='utf-8').read().strip().split(
            '\n')

        for line in lines:
            sentence_keywords = line.split('\t')

            self.assertFalse(len(sentence_keywords) != 2)

            sentence = sentence_keywords[0].split(' ')
            keywords = sentence_keywords[1].split(' ')

            self.assertFalse(len(sentence) < 1 or len(keywords) < 1)

        #self.assertTrue(1==1)


if __name__ == '__main__':
    unittest.main()