__author__ = 'iankuoli'

def clean_text(a):

    a = a.strip('\n').lower()
    a = a.replace("-ups", "ups")
    a = a.replace(".", " .")
    a = a.replace("?", " ?")
    a = a.replace("!", " !")
    a = a.replace(":", " :")
    a = a.replace(",", " ,")
    a = a.replace("'s", " 's")
    a = a.replace("'ve", " 've")
    a = a.replace("n't", " n't")
    a = a.replace("'d", " 'd")
    a = a.replace("'re", " 're")
    a = a.replace("'ll", " 'll")
    a = a.replace("won't", "wont")
    #a = a.replace("'", " '")
    a = a.replace("-", "")
    a = a.replace("[", "")
    a = a.replace("]", "")
    a = a.replace("<s>", ".")
    a = a.replace("</s>", ".")

    return a

