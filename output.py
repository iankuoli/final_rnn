__author__ = 'iankuoli'

import os

id_list = set()

with open("layer8_hw2_best.csv", "r") as f_id:
    for line in f_id:
        lines = line.strip("\n").split(",")
        id_list.add(lines[0])

print(len(id_list))

answer = dict()
ann = dict()

word2id = dict()
id2word = dict()

with open("timit.chmap", "r") as f_mapp:
    for line in f_mapp:
        lines = line.strip("\n").split("\t")
        word = lines[0]
        idd = lines[1]
        word2id[word] = idd
        id2word[idd] = word

rootDir = "_ret_layer8-structure-svm-phone60"
for lists in os.listdir(rootDir):
    if lists[-8:] != "sentence":
        continue
    phoneID = lists[5:-9]
    best_sentence = ""
    best_score = 1000
    with open(rootDir + "/" + lists, 'r', encoding='utf-8') as f_sentence:
        for line in f_sentence:
            lines = line.strip("\n").split("\t")
            sentence = lines[0]
            score = float(lines[1])
            if score < best_score:
                best_sentence = sentence

        best_sentence_list = best_sentence.split(" ")
        ans_sentence = ""
        for i in range(len(best_sentence_list)):
            if best_sentence_list[i] == "":
                continue
            ans_sentence += word2id[best_sentence_list[i]]
        if phoneID not in answer:
            answer[phoneID] = ans_sentence
            ann[phoneID] = best_score
        else:
            if ann[phoneID] > best_score:
                answer[phoneID] = ans_sentence
                ann[phoneID] = best_score

with open("output.csv", "w") as f_output:
    for i in answer.keys():
        if i in id_list:
            f_output.write(str(i) + "," + answer[i] + "\n")