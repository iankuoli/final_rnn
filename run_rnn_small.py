import numpy as np
from rnn import MetaRNN
from hf import SequenceDataset, hf_optimizer
import matplotlib.pyplot as plt
import logging
import time
import clean_text

plt.ion()
logging.basicConfig(level=logging.INFO)

print("Read word2vec ...")
# model feature dim has 200
word2vec = {}
with open('vectors.6B.200d.txt') as fin:
    for line in fin:
        items = line.replace('\r', '').replace('\n', '').split(' ')
        if len(items) < 10:
            continue
        word = items[0]
        vect = np.array([float(i) for i in items[1:]]) # if len(i) > 1])
        if vect.shape[0] != 200:
            print(vect)

        word2vec[word] = vect
'''
model = {}
model["xxxxx"] = [1, 1, 1]
'''
word_vec_len = 200

word2label = {}
label2word = {}

x_seq_list = []
y_seq_list = []

print("Read label2word ...")
with open("label2word.txt", 'r', encoding='utf-8') as file:
    for line in file:
        lines = line.strip('\n').split('\t')
        label = lines[0]
        word = lines[1]
        word2label[word] = label
        label2word[label] = word

print("Read training data ...")
with open('train_small.txt', 'r', encoding='UTF-8') as file:
#with open('training.txt', 'r', encoding='UTF-8') as file:
    for line in file:

        a = clean_text.clean_text(line)
        a = a.split(' ')

        if len(a) < 5:
            continue

        x_seq = np.zeros((len(a), word_vec_len), dtype='float64')
        y_seq = np.zeros((len(a),), dtype='int32')

        for i in range(len(a)):
            word = a[i]

            if word in word2label:
                y_seq[i] = word2label[word]
            else:
                y_seq[i] = word2label["%%%"]

            if word in word2vec:
                x_seq[i, :] = word2vec[word]
            else:
                x_seq[i, :] = word2vec["xxxxx"]

        x_seq_list.append(x_seq)
        y_seq_list.append(y_seq)

t0 = time.time()

n_hidden = 1000
n_in = word_vec_len
n_steps = 10
n_seq = len(x_seq_list)
n_classes = len(word2label)
n_out = n_classes  # restricted to single softmax per time step

n_updates = 500
n_epochs = 300

'''
np.random.seed(0)
# simple lag test
seq = np.random.randn(n_seq, n_steps, n_in)
targets = np.zeros((n_seq, n_steps), dtype=np.int)

thresh = 0.5
# if lag 1 (dim 3) is greater than lag 2 (dim 0) + thresh
# class 1
# if lag 1 (dim 3) is less than lag 2 (dim 0) - thresh
# class 2
# if lag 2(dim0) - thresh <= lag 1 (dim 3) <= lag2(dim0) + thresh
# class 0
targets[:, 2:][seq[:, 1:-1, 3] > seq[:, :-2, 0] + thresh] = 1
targets[:, 2:][seq[:, 1:-1, 3] < seq[:, :-2, 0] - thresh] = 2
#targets[:, 2:, 0] = np.cast[np.int](seq[:, 1:-1, 3] > seq[:, :-2, 0])

'''
print("Convert to SequenceDataset ...")

gradient_dataset = SequenceDataset([x_seq_list, y_seq_list], batch_size=None, number_batches=100)
cg_dataset = SequenceDataset([x_seq_list, y_seq_list], batch_size=None, number_batches=50)

model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out, n_epochs=n_epochs,
                activation='tanh', output_type='softmax',
                use_symbolic_softmax=True, L2_reg=0.0001)

# optimizes negative log likelihood
# but also reports zero-one error
opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                   s=model.rnn.y_pred, n_in=n_in,
                   costs=[model.rnn.loss(model.y),
                          model.rnn.errors(model.y)], h=model.rnn.h)

print("Training ...")
opt.train(gradient_dataset, cg_dataset, initial_lambda=1.0, num_updates=n_updates, save_progress='save_param')

seqs = range(10)

plt.close('all')
for seq_num in seqs:
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(x_seq_list[seq_num])
    ax1.set_title('input')
    ax2 = plt.subplot(212)

    # blue line will represent true classes
    true_targets = plt.step(range(n_steps), y_seq_list[seq_num], marker='o')

    # show probabilities (in b/w) output by model
    guess = model.predict_proba(x_seq_list[seq_num])
    guessed_probs = plt.imshow(guess.T, interpolation='nearest',
                               cmap='gray')
    ax2.set_title('blue: true class, grayscale: probs assigned by model')

print("Elapsed time: %f" % (time.time() - t0))