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

class Cudnn_LSTM_with_bias(lstm_ops.LSTMBlockCell):
  """
  Redefine CudnnCompatibleLSTMCell to add forget_bias = 1.0
  """

  def __init__(self, num_units, reuse=None):
    super().__init__(
        num_units, forget_bias=1.0, cell_clip=None, use_peephole=False,
        reuse=reuse, name="cudnn_compatible_lstm_cell")
    self._names.update({"scope": "cudnn_compatible_lstm_cell"})
    
def lstm_cell(keep_prob):
    '''
    Define one single lstm cell
    args:
    keep_prob: tensor scalar
    '''
    if tf.test.is_gpu_available():
        lstm = Cudnn_LSTM_with_bias(num_units)
    else:
        lstm = tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1.0)
    lstm=tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return lstm

def rnn(x, weight, bias,keep_prob):
    '''
     define stacked cells and prediction
     x: data with shape [batch_size,max_len,len_unique_char]
    ''' 
    cells = tf.contrib.rnn.MultiRNNCell([lstm_cell(keep_prob) for _ in range(num_layers)])
    outputs, states = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)
    prediction = tf.add(tf.matmul(states[-1].h, weight), bias,name='prediction')
    return prediction


def run(train_data, target_data, unique_chars, len_unique_chars):
    '''
     main run function
     train_data: data with shape [length_of_data,max_len,len_unique_char]
     target_data: data with shape [length_of_data,len_unique_char]
    '''
    ops.reset_default_graph()
    keep_prob=tf.placeholder_with_default(1.0,shape=())
    x = tf.placeholder("float", [None, max_len, len_unique_chars],name='input')
    y = tf.placeholder("float", [None, len_unique_chars],name='target')
    weight = tf.get_variable('w1',shape=(num_units, len_unique_chars),initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('b',shape=(len_unique_chars),initializer=tf.contrib.layers.xavier_initializer())

    prediction = rnn(x, weight, bias,keep_prob)
    softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        costs = []
        num_batches = int(len(train_data)/batch_size)

        for i in range(epoch):
            print ("----------- Epoch {0}/{1} -----------".format(i, epoch))
            count = 0
            minibatch_cost = 0.
            for _ in range(num_batches):
                train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
                count += batch_size
                _,temp_cost=sess.run([optimizer,cost] ,feed_dict={x:train_batch, y:target_batch,keep_prob:0.8})
                minibatch_cost += temp_cost / num_batches
            
            costs.append(minibatch_cost)
            if (i) % 4==0:
                print ("Cost after epoch %i: %f" % (i+1, minibatch_cost))
            
            #get on of training set as seed
            
            
            seed = train_batch[:1:]

            #to print the seed 40 characters
            seed_chars = ''
            for each in seed[0]:
                    seed_chars += unique_chars[np.where(each == max(each))[0][0]]
            print ("Seed:", seed_chars)

            #predict next 1000 characters
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
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per two)')
        
        print("Model is saved to ", save_path)
        
        with open ('unique_chars.pkl', 'wb') as g:
            pickle.dump(unique_chars,g)
        sess.close()



if __name__ == '__main__':
    text=read_data('speeches.txt')
    train_data, target_data, unique_chars, len_unique_chars = featurize(text)
    run(train_data, target_data, unique_chars, len_unique_chars)