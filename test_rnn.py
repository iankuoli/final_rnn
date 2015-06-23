__author__ = 'iankuoli'

import numpy
from rnn import MetaRNN
import matplotlib.pyplot as plt
import logging
import clean_text
import pickle
import theano

model_path = "save_param2_hidden100_best"

plt.ion()
logging.basicConfig(level=logging.INFO)

# model feature dim has 200
model = {}
fin = open('vectors.6B.200d.txt')
for line in fin:
    items = line.replace('\r', '').replace('\n', '').split(' ')
    if len(items) < 10:
        continue
    word = items[0]
    vect = numpy.array([float(i) for i in items[1:]]) # if len(i) > 1])
    if vect.shape[0] != 200:
        print(vect)

    model[word] = vect

word_vec_len = 200

word2label = {}
label2word = {}

x_seq_list = []
y_seq_list = []

labelCount = 0

# Convet "<s>" and "</s>" to "."
word2label["."] = labelCount
label2word[labelCount] = "."
labelCount += 1

# Other words are set "%%%"
word2label["%%%"] = labelCount
label2word[labelCount] = "%%%"
labelCount += 1

# Map word to labelID
with open('MLDS_Final/sentence/train_clean.set', 'r', encoding='UTF-8') as file:
    for line in file:

        a = clean_text.clean_text(line)
        a = a.split(' ')

        for i in range(len(a)):
            word = a[i]
            if word not in word2label:
                word2label[word] = labelCount
                label2word[labelCount] = word
                labelCount += 1

n_hidden = 100
n_in = word_vec_len
n_out = len(label2word)
RNN = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                activation='tanh', output_type='softmax',
                use_symbolic_softmax=True)

save = pickle.load(open(model_path, 'rb'), encoding='latin1')
cg_last_x, best, lambda_, first_iteration, init_p = save

for i, j in zip(RNN.rnn.params, init_p):
    i.set_value(j)

inputs = [RNN.x, RNN.y]
#costs = [model.rnn.loss(model.y), model.rnn.errors(model.y)]
cost = RNN.rnn.loss(RNN.y)
f_pred = theano.function(inputs, cost, on_unused_input='ignore')

test_str = "i am fire thin q"
test_str = "<s> " + test_str + " </s>"

a = clean_text.clean_text(test_str)
a = a.split(' ')

x_seq = numpy.zeros((len(a), word_vec_len), dtype='float64')
y_seq = numpy.zeros((len(a),), dtype='int32')

for i in range(len(a)):
    word = a[i]
    if word in word2label:
        y_seq[i] = word2label[word]
    else:
        y_seq[i] = word2label["%%%"]

    if word in model:
        x_seq[i, :] = model[word]
    else:
        x_seq[i, :] = model["xxxxx"]

predict = f_pred(*[x_seq, y_seq])
print(predict)






