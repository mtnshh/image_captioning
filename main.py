import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
import spacy

train = pd.read_csv('./dataset-flick8k/caption-token.csv', delimiter = '\t', header = None)
train.info()

# print(train)

dictonary = set()
corpus = []

for i in range(0, len(train)):
    # print(train[1][i])
    text = train[1][i]
    # print(text)
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    # nlp = spacy.load('en', disable=['parser', 'ner'])
    # doc = nlp(text)
    # text = " ".join([token.lemma_ for token in doc])
    text = text.split()
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if not word in set(stopwords.words('english'))] 
    # text = []
    final = []
    for word in text:
        if word != "-PRON-":
            final.append(word)
    text = ' '.join(final)
    # print(text)
    # for word in text:
    #     dictonary.update(word)
    corpus.append(text)
    if i%100 ==  0:
        print(i)

word_count = {}

for text in corpus:
    for w in  text.split(' '):
        word_count[w] = word_count.get(w, 0) + 1
    
vocab = [w for w in word_count if word_count[w] >= 15]

print(len(vocab))

cnt = 0
file = open(dir, 'r')
while 1:
    s = file.readline()
    cnt = cnt + 1
    if s == "": 
        break
    image = "./dataset-flick8k/Images/" + s
    image = image[:len(image)-1]
    s = s[:len(s)-1]
    img = cv2.imread(image)
    save_to = "./train/" + s
    # print(save_to)
    # print(img)
    cv2.imwrite(save_to, img)
    if cnt%100:
        print(cnt)
