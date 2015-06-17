__author__ = 'iankuoli'

import clean_text

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

with open('label2word.txt', 'w', encoding='UTF-8') as file:
    for i in label2word.keys():
        strout = str(i) + '\t' + label2word[i] + '\n'
        file.write(strout)