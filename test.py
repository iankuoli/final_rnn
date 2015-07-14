__author__ = 'iankuoli'

import os
from rnn_model import RNNmodel

model_path = "save_param2_hidden100_best"
word2vec_path = "vectors.6B.200d.txt"
label_path = "label2word_small.txt"

n_in = 200
n_hidden = 80

rnn = RNNmodel(word2vec_path, label_path, model_path, n_in, n_hidden)

'''
#
# HW3
#
with open("testing_data.txt", encoding='utf-8') as f:
    str_list = [sentence.strip("\n") for sentence in f]

strindx_list = list(map(lambda s: s.find(" "), str_list))
ret_list = list(map(lambda i: str_list[i][strindx_list[i]+1:], range(len(str_list))))
ret = rnn.predict(ret_list)

with open("HW3_sub.csv", 'w') as f_w:
    for i in range(int(len(ret)/5)):
        ans_cand = list(map(lambda j: ret[i*5+j].tolist(), range(5)))
        ans = ans_cand.index(max(ans_cand))
        print(str(i+1) + "  " + str(ans_cand))
        if ans == 0:
            ans_id = "a"
        elif ans == 1:
            ans_id = "b"
        elif ans == 2:
            ans_id = "c"
        elif ans == 3:
            ans_id = "d"
        else:
            ans_id = "e"
        f_w.write(str(i+1) + "," + ans_id + "\n")
'''

#
# Final
#
rootDir = "layer8-structure-svm-phone60"
for lists in os.listdir(rootDir):
    if lists[-8:] != "sentence":
        continue
    with open(rootDir + "/" + lists, 'r', encoding='utf-8') as f_sentence:
        strlist = [sentence.strip("\n") for sentence in f_sentence]
        ret = rnn.predict(strlist)

    with open("_ret_" + rootDir + "/_ret_" + lists, 'w') as f_ret:
        for indx in range(len(strlist)):
            a = strlist[indx]
            f_ret.write(strlist[indx] + "\t" + str(ret[indx].tolist()) + "\n")


'''
strlist = []
strlist.append("I am a")
strlist.append("I am a student")
strlist.append("I am a array")
strlist.append("I am an array")
strlist.append("I am a car")
strlist.append("I am a independence")
strlist.append("I am a stood aunt")
strlist.append("I am stood")
strlist.append("I un a stood dent")
strlist.append("I a mass to dent")
strlist.append("I am a stand aunt")
strlist.append("I am stand up")

ret = rnn.predict(strlist)

print(ret)
'''
