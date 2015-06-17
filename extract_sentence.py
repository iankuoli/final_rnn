__author__ = 'iankuoli'

sent_clean = []
with open('MLDS_Final/sentence/train.set', 'r', encoding='UTF-8') as file:
    for line in file:
        a = line.strip('\n')
        indx = a.find(',') + 1
        sent_clean.append(a[indx:])

with open('MLDS_Final/sentence/train_clean.set', 'w', encoding='UTF-8') as file:
    for line in sent_clean:
        file.write('<s> ' + line + ' </s>\n')