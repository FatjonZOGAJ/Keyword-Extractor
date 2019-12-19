# Applied Deep Learning Exercise 1
The goal of exercise 1 is to create a detailed plan of the next steps for the lecture *Applied Deep Learning* with time estimation for each of them. It contains a short overview in form of a table followed by a written description.

| 11708468|Fatjon ZOGAJ|
|:--|:--|
| subject| *Keyword Extraction* |
| topic | Natural Language Processing |
| project type | Bring your own method |

The following list  contains papers, blogs and code which will be used for  the future exercises. 

|References ||
|--:|:--|
|  [1] <br/> <br/> [2]  <br/> <br/> [3]  <br/> <br/> [4]  <br/> <br/> [5]  <br/> <br/> [6] <br/> <br/>| Keyphrase Extraction Using Deep Recurrent Neural Networks on Twitter <br/> [Link paper](http://jkx.fudan.edu.cn/~qzhang/paper/keyphrase.emnlp2016.pdf), [Link GitHub](https://github.com/fudannlp16/KeyPhrase-Extraction) <br/>  Bidirectional LSTM Recurrent Neural Network for Keyphrase Extraction <br/> [Link paper](http://ailab.uniud.it/wp-content/uploads/2018/05/Bidirectional-Lstm-Keyphrase_IRCDL2018.pdf), [Link GitHub](https://github.com/basaldella/deepkeyphraseextraction) <br/> Deep Keyphrase Generation<br/> [Link paper](https://arxiv.org/pdf/1704.06879.pdf), [Link GitHub](https://github.com/memray/seq2seq-keyphrase), [Link Github PyTorch](https://github.com/memray/seq2seq-keyphrase-pytorch) <br/> Tweet Classification with CNN, RNN<br/> [Link GitHub](https://github.com/sinadabiri/Tweet-Classification-Deep-Learning-Traffic) <br/> Semi-supervised Convolutional Neural Networks forText Categorization via Region Embedding<br/> [Link paper](https://arxiv.org/pdf/1504.01255.pdf), [Link GitHub C++](https://github.com/riejohnson/ConText) <br/> Hierarchical Attention Networks for Document Classification <br/> [Link paper](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf), [Link GitHub](https://github.com/arunarn2/HierarchicalAttentionNetworks), [Link other GitHub classification](https://github.com/richliao/textClassifier) ||
| Other <br/> <br/> <br/> <br/> <br/>| [Extracting Keywords From Short Text](https://towardsdatascience.com/extracting-keywords-from-short-text-fce39157166b) <br/> [CNN and what not for Text Classification](https://towardsdatascience.com/nlp-learning-series-part-3-attention-cnn-and-what-not-for-text-classification-4313930ed566) <br/> [Multitext Classification](https://github.com/adsieg/Multi_Text_Classification) <br/> [Text Classification Using LSTM and visualize Word Embeddings](https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b) <br/> [GloVe vs word2vec Word Embedding](https://cai.tools.sap/blog/glove-and-fasttext-two-popular-word-vector-models-in-nlp/)|

## Description
After spending approximately 25-30 hours reading lots of different kinds of papers, blogs and code trying to choose a topic that is  

 1. a little more sophisticated (e.g. not Connect Four, Flappy Bird, Candy Crush)  
 2. a current hot topic in the research world

I have decided on taking on the problem of “Keyword Extraction” with the project type “Bring your own method”. Ideally, I would turn this into “Beat the stars” but considering the scope of this lecture (and my obvious limited possibilities compared to teams working on this problem together) achieving this task may be a little farfetched. 
Nevertheless, I have chosen this subfield of Natural Language Processing as it contains various topics such as word embedding, text analysis/semantics, translation, summarization, etc. which are of big interest to me.  
My choice of project type allows me to get familiar with not only the most common used frameworks and the to me currently “unknown” language Python, but also with the specific Neural Networks in particular. This way, I want to consolidate the theoretical knowledge acquired in the lecture also in a practical way.

### Approach
The plan is to compare different kinds of Neural networks by trying out different parameters and network architectures. Reflecting the state-of-the-art it would be interesting to see how they stack up against each other based on a few public datasets. As a Business Informatics student the dataset of Tweets [1] may be of particular interest as it could lend insight into how one could use publicly available data for marketing/busines strategy purposes. Further datasets will be taken from the papers mentioned in the references to ensure comparability. These may include abstract papers (INSPEC contains 2000 documents with controlled and uncontrolled manual keyphrases)[2][3], scientific articles (SEMEVAL contains 144 documents)[2][3], scientific publications (KP20k with 20k articles in computer science)[3], reviews [5] or others. The twitter dataset was gathered by the linked study and contains around 41 million unprocessed tweets. Important to note is that tweets were removed that contained multiple hashtags, hashtags at the end or were part of a conversation and not a standalone tweet which reduced the amount of tweets to 110k. This way only tweets which included a single hashtag in a semantically natural sentences were picked for the study. Manual random evaluation has shown that in 90% of the cases the found hashtags fit well as keyphrases. These datasets have been preprocessed in some way or another (remove non-latin words, special characters, etc) and may need further processing.

I will start off the future exercise by looking into the different kinds of datasets and prepare the data to be fed into the Neural Networks. After mapping it with word embeddings (see reference table Other) I will continue by analyzing different Neural Networks (Deep RNN [1], Bidirectional LSTM RNN [2], Encoder-Decoder RNN [3], Classification CNN [4]) and try to optimize them by writing my own one. Based on how fast I will progress I may choose to forego some completely. A report and a final presentation will follow the implementation.  As these tasks will take a considerate amount of time, the application to present the results will be a rather simple API. 

The time estimates for the corresponding tasks have been summarized in the following table. As can be seen the current spent time already does not comply with the time plan. To ensure an adequate scope, parts of the project may be changed based on future progress.

### Work Breakdown Structure
|Task| Time estimate | Actual time|
|:--|--:| --:|
| Find topic and create plan|10h| **25h** |
| Dataset collection + preparation| 5h |
| Network design + implementation| 20h |
| Training + finetuning| 15h |
| Building application | 5h |
| Report + presentation| *14h |
| Lecture | *16h| 16h |
| | 3 ECTS + **10h** |

\* taken from the lecture TISS website 

#### References
[1] Wang, Yang et al. (2016). Keyphrase Extraction Using Deep Recurrent Neural Networks on Twitter.  836-845. 10.18653/v1/D16-1080.

[2] Basaldella, Marco et al. (2018). Bidirectional LSTM Recurrent Neural Network for Keyphrase Extraction. _IRCDL_.

[3] Meng, Rui et al. (2017). Deep Keyphrase Generation. 582-592. 10.18653/v1/P17-1054. 

[4] Dabiri, Sina (2018). Tweet-Classification-Deep-Learning-Traffic. Github repository: https://github.com/sinadabiri/Tweet-Classification-Deep-Learning-Traffic 

[5] Johnson, Rie & Zhang, Tong. (2015). Semi-supervised Convolutional Neural Networks for Text Categorization via Region Embedding. Advances in neural information processing systems. 28. 919-927. 

[6] Yang, Zichao et al. (2016). Hierarchical Attention Networks for Document Classification. _HLT-NAACL_.



<br/>
<br/>
<br/>
<br/>

#### Notes
The following paragraph consists of some additional notes I have taken during my research and is not directly part of *Exercise 1*. Thus it is unformatted and partly in German.

Ideas:
-	Rechteckige Platform, die sich bewegt und versucht eine Kugel zu balancieren, indem sie sich um ihre 2 Achsen dreht (dabei kommt von random Orten eine Kraft, die auf sie wirkt und die Kugel anstoßt -> lernen, wie sie sich drehen muss, um dagegen zu wirken)
	-	Falls genug Zeit bleibt könnte man auch die Kraft selber darauf wirken lassen)
-	Algorithmus schreiben, der basierend auf Wörtern/einem Satz einen Satz zurückgibt, der sich reimt und ungefähr mit dem Thema von dem Inputsatz zu tun hat (e.g. keine Liebestexte bei einem Rapsong) 
-	Flappy bird? Candy Crush, Connect Four 4 Gewinnt
-	AI to classify a picture/drawing into a certain genre (baroque, modern, …) 
-	“Translate” songs to pictures/pictures to songs (based on keywords)
Encoder-Decoder
-	Use Instagram pictures to find good hashtags for it (picture scraping is illegal?)
-	Optical Character Recognition extension for other languages (Albanian characters)
-	Language Modeling (determining probability of sequence of words)
-	Write AI that checks whether some words are keywords or not (use own data based off old PDFs I have marked) (based on being part of certain word embedding ?)
	-	https://pdfs.semanticscholar.org/9302/03b85f789972107df4bc08da77632e477b84.pdf NEURAL NETWORK BASEDKEYWORD EXTRACTION USING WORD FREQUENCY, POSITION, USAGE AND FORMAT FEATURES
	- https://monkeylearn.com/keyword-extraction/
“Most systems that use some kind of linguistic information outperform those that don’t do so” (siehe paper)
global context and local context information -> feature maps? Based on words
SVM https://link.springer.com/content/pdf/10.1007%2F11775300.pdf
classification:
good keyword: general notion of doc, several important proportions, meaning,
bad: doesn’t describe, hard to understand, 
indifferent between good and bad 
	- text analysis ? (https://monkeylearn.com/text-analysis/
	- “last words of a sentence can affect first words of translation” -> Sequence to sequence encoding/decoding nutzen auf Absätze (um mehr Sinn rauszuholen)
aber: RNN mit langen Sätzen: vanishing/exploding gradients -> dauert lange (gute Parameterinitializaiton, Gradient Clipping, faster optimizers helfen), jedoch sogar bei moderately long sequences (100 inputs) dauert es noch zu lange.
Lösung: unroll RNN only over limited number of time steps during training (truncated backpropagation) z.B.: TensorFlow: truncate input sequence 
Nachteil: long-term Muster können nicht gelernt werden (dies kann umgangen werden indem alte und neue Daten genutzt werden, stock prices: Monatliche Daten letzter 5 Monate, wöchentliche Daten letzter 5 Wochen und tägliche Daten letzter 5 Tage (dadurch werden aber Details ignoriert, die auch wichtig sein könnten, z.B.: Wahl die langfristig beeinflusst)
lange RNNs -> Memory von frühen Input geht mit der Zeit verloren
-> verschiedene Arten von Zellen mit Langzeit Memory (so nützlich, dass andere Zellen fast obsolet sind) 
-> LSTM ZELLEN
-> Gated Recurrent Unit ist einfacher und funktioniert ähnlich gut wie LSTM
beide gut für Natural Language Processing (Übersetzung, Zusammenfassung, Sinnanalyse, …)

Natural Language Processing:
word hashing: one hot word vector mapped to limited letter trigram
Embedding: ganzes Vokabular auf z.B.: 150 Dimensionen reduzieren (one hot encoding zu aufwenidg)
RNN lernt die embeddings auf Grund des Trainings selbst zuzuweisen (ähnliche Wörter clustern sich zueinander)
Satz „I drink milk“ einspielen:
-	Preprocess and divide into list of known words 
o	Replace unknown words by [UNK]
o	Numerical values [NUM], url -> [URL], Großbuchstaben -> [ABR], …
-	50.000er Wörterbuch: „I drink milk“ = z.B.: [72, 3335, 338]
-> word embeddings lernen oder vortrainierte benutzen (dieses embedding kann beim Arbeiten angepasst werden über backpropagation (bessere performance) oder gleich bleiben trainable = False (geht schneller)
-	Translation: Encode „I drink milk” in reverse “milk drink I” because “I” is first thing decoder needs to translate
integers are looked up with embedding lookup -> embeddings are input for encoder and decoder 
-> each step: decoder outputs a score for each word in output vocabulary (French) -> Softmax turns this into probability (“I”: “Je” = 20%, “Tu” = 1%, …) -> word with highest probability is output
Sampling technique let’s us reduce complexity for Softmax
Different lengths: padding or grouping similar lengths together 

Notes:
NLP: translation, summarization, image captioning, …; overlap between these 4 
-	language modeling: which words follow (can be viewed as “what do words mean”)
-	morphology: how are words formed (prefix, suffix, tense)
-	parsing: which words affect others
-	semantics: what words mean as a collective (individual meaning + relation) 
	- comparison: matching two sentences with similar meaning
	- sentence modeling: model meaning of sentence in vector
-	https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
in order to perform fine-tuning, all layers should start with properly trained weights: for instance you should not slap a randomly initialized fully-connected network on top of a pre-trained convolutional base. This is because the large gradient updates triggered by the randomly initialized weights would wreck the learned weights in the convolutional base. In our case this is why we first train the top-level classifier, and only then start fine-tuning convolutional weights alongside it.
-	we choose to only fine-tune the last convolutional block rather than the entire network in order to prevent overfitting, since the entire network would have a very large entropic capacity and thus a strong tendency to overfit. The features learned by low-level convolutional blocks are more general, less abstract than those found higher-up, so it is sensible to keep the first few blocks fixed (more general features) and only fine-tune the last one (more specialized features).
-	fine-tuning should be done with a very slow learning rate, and typically with the SGD optimizer rather than an adaptative learning rate optimizer such as RMSProp. This is to make sure that the magnitude of the updates stays very small, so as not to wreck the previously learned features.
-	CNN: not enough data? Use data from related domain to train and then use in-domain data to fine tune it

