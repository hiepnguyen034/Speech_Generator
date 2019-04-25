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

def rnn(x, weight, bias, len_unique_chars):
    '''
     define rnn cell and prediction
    '''
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, len_unique_chars])
    x = tf.split(x, max_len, 0)
    
    cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0,name='basicLSTM')
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.add(tf.matmul(outputs[-1], weight), bias,name='prediction')
    return prediction


def run(train_data, target_data, unique_chars, len_unique_chars):
    '''
     main run function
    '''
    ops.reset_default_graph()
    x = tf.placeholder("float", [None, max_len, len_unique_chars],name='input')
    y = tf.placeholder("float", [None, len_unique_chars],name='target')
    weight = tf.get_variable('w1',shape=(num_units, len_unique_chars),initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('b',shape=(len_unique_chars),initializer=tf.contrib.layers.xavier_initializer())

    prediction = rnn(x, weight, bias, len_unique_chars)
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)

        num_batches = int(len(train_data)/batch_size)

        for i in range(epoch):
            print ("----------- Epoch {}/{} -----------".format(i+1, epoch))
            count = 0
            for _ in range(num_batches):
                train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
                count += batch_size
                sess.run([optimizer] ,feed_dict={x:train_batch, y:target_batch})

            
            seed = train_batch[:1:]

            
            seed_chars = ''
            for each in seed[0]:
                    seed_chars += unique_chars[np.where(each == max(each))[0][0]]
            print ("Seed:", seed_chars)

            
            for i in range(40):
                if i > 0:
                    remove_fist_char = seed[:,1:,:]
                    seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, len_unique_chars]), axis=1)
                predicted = sess.run([prediction], feed_dict = {x:seed})
                predicted = np.asarray(predicted[0]).astype('float64')[0]
                probabilities = sample(predicted)
                predicted_chars = unique_chars[np.argmax(probabilities)]
                seed_chars += predicted_chars
            print ('Result:', seed_chars)
        save_path = saver.save(sess, "./saved_model/RNN_model.ckpt")
        print("Model is saved to ", save_path)
        with open ('unique_chars.pkl', 'wb') as g:
            pickle.dump(unique_chars,g)



if __name__ == '__main__':
    text=read_data('speeches.txt')
    train_data, target_data, unique_chars, len_unique_chars = featurize(text)
    run(train_data, target_data, unique_chars, len_unique_chars)