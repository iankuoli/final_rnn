__author__ = 'iankuoli'

from rnn_model import RNNmodel

model_path = "save_param2_hidden100_best"
word2vec_path = "vectors.6B.200d.txt"
label_path = "label2word.txt"

n_in = 200
n_hidden = 80

rnn = RNNmodel(word2vec_path, label_path, model_path, n_in, n_hidden)

strlist = []
strlist.append("I am a student")
strlist.append("I an a student")
strlist.append("I am a stood aunt")
strlist.append("I am stood")

ret = rnn.predict(strlist)

print(ret)