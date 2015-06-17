__author__ = 'iankuoli'

from rnn_model import RNNmodel

model_path = "save_param_hidden80_328_043"
word2vec_path = "vectors.6B.200d.txt"
label_path = "label2word.txt"

n_in = 200
n_hidden = 80

rnn = RNNmodel(word2vec_path, label_path, model_path, n_in, n_hidden)

strlist = []
strlist.append("I'm gonna make him an offer he can't refuse")
strlist.append("I'm gonna make he man o for he can refugee")
strlist.append("I'm go lamb him an of her he can't refuse")
strlist.append("I'm go lake he man of her he ant refugee")

ret = rnn.predict(strlist)

print(ret)