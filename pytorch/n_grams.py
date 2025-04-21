import warnings
from tqdm import tqdm
import time
from collections import OrderedDict
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
nltk.download('punkt')
nltk.download('punkt_tab')
warnings.simplefilter('ignore')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#preprocess
song= """We are no strangers to love
You know the rules and so do I
A full commitments what Im thinking of
You wouldnt get this from any other guy
I just wanna tell you how Im feeling
Gotta make you understand
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Weve known each other for so long
Your hearts been aching but youre too shy to say it
Inside we both know whats been going on
We know the game and were gonna play it
And if you ask me how Im feeling
Dont tell me youre too blind to see
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Weve known each other for so long
Your hearts been aching but youre too shy to say it
Inside we both know whats been going on
We know the game and were gonna play it
I just wanna tell you how Im feeling
Gotta make you understand
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you"""
def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", '', s)
    s = re.sub(r"\d", '', s)
    return s
def preprocess(words):
    tokens = word_tokenize(words)
    tokens = [preprocess_string(w) for w in tokens]
    return [w.lower() for w in tokens if len(w) != 0 or not(w in string.punctuation)]
tokens = preprocess(song)

#unigram model
fdist = nltk.FreqDist(tokens)
C=sum(fdist.values())
print(fdist['strangers']/C)
vocabulary=set(tokens)

#bigram
bigrams = nltk.bigrams(tokens)
my_bigrams = list(nltk.bigrams(tokens))
freq_bigrams = nltk.FreqDist(nltk.bigrams(tokens))
#predicting
word="strangers"
vocab_probabilities={}
for next_word in vocabulary:
    vocab_probabilities[next_word] = freq_bigrams[(word,next_word)] / fdist[word]
vocab_probabilities = sorted(vocab_probabilities.items(), key=lambda x:x[1], reverse = True)
print(vocab_probabilities[0:4])
def make_predictions(my_words, freq_grams, normalize=1, vocabulary=vocabulary):
    vocab_probabilities = {}
    context_size = len(list(freq_grams.keys())[0])
    my_tokens = preprocess(my_words)[0:context_size - 1]
    for next_word in vocabulary:
        temp = my_tokens.copy()
        temp.append(next_word)
        if normalize!=0:
            vocab_probabilities[next_word] = freq_grams[tuple(temp)] / normalize
        else:
            vocab_probabilities[next_word] = freq_grams[tuple(temp)] 
    vocab_probabilities = sorted(vocab_probabilities.items(), key=lambda x: x[1], reverse=True)
    return vocab_probabilities
my_words="are"
vocab_probabilities=make_predictions(my_words,freq_bigrams,normalize=fdist['i'])
#creating song
my_song="i"
for i in range(100):
    my_word=make_predictions(my_word,freq_bigrams)[0][0]
    my_song+=" "+my_word

#trigram
freq_trigrams  = nltk.FreqDist(nltk.trigrams(tokens))
make_predictions("so do",freq_trigrams,normlize=freq_bigrams[('do','i')] )[0:10]
my_song=""
w1=tokens[0]
for w2 in tokens[0:100]:
    gram=w1+' '+w2
    my_word=make_predictions(gram,freq_trigrams )[0][0]
    my_song+=" "+my_word
    w1=w2
print(my_song)