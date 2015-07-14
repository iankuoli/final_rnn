__author__ = 'iankuoli'
import clean_text

setLabel = set()

# Convet "<s>" and "</s>" to "."
setLabel.add(".")

# Other words are set "%%%"
setLabel.add("%%%")

dictWord = dict()

with open('training_2.txt', 'r', encoding='UTF-8') as file:
    for line in file:

        a = clean_text.clean_text(line)
        a = a.split(' ')

        for i in range(len(a)):
            word = a[i]

            if word not in dictWord:
                dictWord[word] = 1
            else:
                dictWord[word] += 1

for key in dictWord.keys():
    if dictWord[key] > 20:
        setLabel.add(key)

# Map word to labelID
with open('MLDS_Final/sentence/train_clean.set', 'r', encoding='UTF-8') as file:
    for line in file:

        a = clean_text.clean_text(line)
        a = a.split(' ')

        for i in range(len(a)):
            word = a[i]
            setLabel.add(word)


with open('testing_data.txt', 'r', encoding='UTF-8') as file:
    for line in file:

        a = clean_text.clean_text(line)
        a = a.split(' ')

        for i in range(len(a)):
            word = a[i]
            setLabel.add(word)

with open('label2word.txt', 'w', encoding='UTF-8') as file:

    labelCount = 0

    for word in setLabel:
        strout = str(labelCount) + '\t' + word + '\n'
        file.write(strout)
        labelCount += 1

print(labelCount)