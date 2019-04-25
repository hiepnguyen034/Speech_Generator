import tensorflow as tf
import numpy as np
import io
import re
from tensorflow.python.framework import ops
import pickle
from utils import * 

max_len = 20
step = 3
num_units = 128
learning_rate = 0.001
batch_size = 64
epoch = 64
temperature = 1.0
len_unique_chars=6645
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def take_input():
    while True:
        num_words=input('Number of words in the speech: ')
        try:
            num_words=int(num_words)
            break
        except ValueError as e:
            print(e, 'We need an integer here') 
    while True:
        sentence=input('Please insert the beginning of the speech (first 10 words will be used): ')
        if len(sentence)<20:
            print ('We need more than 10 words. Please try again')
        else:
            break

    return num_words,sentence


def restore_model(sentence,num_words):
    input_sentence,test=text_to_1hot(sentence)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('saved_model/RNN_model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('saved_model/'))
        print('Model restored from ./saved_model/')
        graph = tf.get_default_graph()
        prediction=graph.get_tensor_by_name('prediction:0')
        x=graph.get_tensor_by_name('input:0')
        #result=sess.run([pred], feed_dict=feed_dict)
        seed = test[:1]
        with open ('unique_chars.pkl', 'rb') as g:
            unique_chars=pickle.load(g)
        seed_chars = ''
        #seed_chars=''
        for each in seed[0]:
                seed_chars += unique_chars[np.where(each == max(each))[0][0]]
        print ("Seed:", seed_chars)
        
        #predict next 1000 characters
        for i in range(num_words*2):
            if i > 0:
                remove_fist_char = seed[:,1:,:]
                seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, len_unique_chars]), axis=1)
            predicted = sess.run([prediction], feed_dict = {x:seed})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            probabilities = sample(predicted)
            predicted_chars = unique_chars[np.argmax(probabilities)]
            seed_chars += predicted_chars
        print ('Result:', seed_chars)

if __name__ == '__main__':
    num_words,sentence=take_input()
    restore_model(sentence,num_words)