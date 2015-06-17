__author__ = 'iankuoli'

import numpy
from rnn import MetaRNN
import clean_text
import pickle
import theano

class RNNmodel:

    def __init__(self, word2vec_path, label_path, model_path, n_in, n_hidden):

        self.n_in = n_in
        self.n_hidden = n_hidden

        #
        # Initialize the label vectorID
        #
        self.word2label = {}
        self.label2word = {}
        with open(label_path, 'r', encoding='UTF-8') as file:
            for line in file:
                lines = line.strip('\n').split('\t')
                label = lines[0]
                word = lines[1]
                self.word2label[word] = label
                self.label2word[label] = word
        self.n_out = len(self.label2word)

        #
        # Initialize the word2vec model (model feature dim has 200)
        #
        self.model = {}
        with open(word2vec_path) as fin:
            for line in fin:
                items = line.replace('\r', '').replace('\n', '').split(' ')
                if len(items) < 10:
                    continue
                word = items[0]
                vect = numpy.array([float(i) for i in items[1:]]) # if len(i) > 1])
                if vect.shape[0] != 200:
                    print(vect)

                self.model[word] = vect

        #
        # Initialize the RNN Model
        #
        self.RNN = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=self.n_out,
                activation='tanh', output_type='softmax',
                use_symbolic_softmax=True)

        save = pickle.load(open(model_path, 'rb'), encoding='latin1')
        cg_last_x, best, lambda_, first_iteration, init_p = save

        for i, j in zip(self.RNN.rnn.params, init_p):
            i.set_value(j)

        inputs = [self.RNN.x, self.RNN.y]
        cost = self.RNN.rnn.loss(self.RNN.y)
        self.f_pred = theano.function(inputs, cost, on_unused_input='ignore')

    #
    # Predict the loss of the input x which is a list of strings
    #
    # Input: a list of strings
    # Output: a list of loss
    def predict(self, x):
        predicts = []

        for test_str in x:
            test_str = "<s> " + test_str + " </s>"

            a = clean_text.clean_text(test_str)
            print(a)
            a = a.split(' ')

            x_seq = numpy.zeros((len(a), self.n_in), dtype='float64')
            y_seq = numpy.zeros((len(a),), dtype='int32')

            for i in range(len(a)):
                word = a[i]
                if word in self.word2label:
                    y_seq[i] = self.word2label[word]
                else:
                    y_seq[i] = self.word2label["%%%"]

                if word in self.model:
                    x_seq[i, :] = self.model[word]
                else:
                    x_seq[i, :] = self.model["xxxxx"]

            predicts.append(self.f_pred(*[x_seq, y_seq]))

        return predicts