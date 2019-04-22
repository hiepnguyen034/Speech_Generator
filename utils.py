import tensorflow as tf
import numpy as np
import io
import re
from tensorflow.python.framework import ops
import pickle 


max_len = 20
step = 3
num_units = 128
learning_rate = 0.001
batch_size = 64
epoch = 64
temperature = 1.0
len_unique_chars=6645

def read_data(file_name):
    '''
     open and read text file
    '''
    with open(file_name, 'r') as f:
        text=f.read().lower()
    text = re.split(r'(\s+)', text)
    return text

def featurize(text):
    '''
     featurize the text to train and target dataset
    '''
    unique_chars = sorted(set(text))
    len_unique_chars = len(unique_chars)

    input_chars = []
    output_char = []

    for i in range(0, len(text) - max_len, step):
        input_chars.append(text[i:i+max_len])
        output_char.append(text[i+max_len])
    
        
    train_data = np.zeros((len(input_chars), max_len, len_unique_chars))
    target_data = np.zeros((len(input_chars), len_unique_chars))

    for i , each in enumerate(input_chars):
        for j, char in enumerate(each):
            train_data[i, j, unique_chars.index(char)] = 1
        target_data[i, unique_chars.index(output_char[i])] = 1
    return train_data, target_data, unique_chars, len_unique_chars

def sample(predicted):
    '''
     helper function to sample an index from a probability array
    '''
    exp_predicted = np.exp(predicted/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities

def text_to_1hot(sentence):
    with open ('unique_chars.pkl', 'rb') as g:
        unique_chars=pickle.load(g)
    sentence=sentence.lower()
    sentence=re.split(r'(\s+)', sentence)
    input_sentence=[]
    for i in range(0, len(sentence) - max_len, step):
        input_sentence.append(sentence[i:i+max_len])
    test_sentence = np.zeros((len(sentence), max_len, len_unique_chars))
    for i , each in enumerate(input_sentence):
        for j, char in enumerate(each):
            test_sentence[i, j, unique_chars.index(char)] = 1
    return input_sentence,test_sentence