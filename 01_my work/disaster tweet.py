import numpy as np
import pandas as pd
import re
import string
import gensim #word2vec에 사용

from nltk.corpus import stopwords #remove_stopwords 에 사용
from nltk.stem import WordNetLemmatizer #lemmatization 에 사용
from nltk.stem import PorterStemmer #stemming 에 사용
from nltk import word_tokenize #tokenization 에 사용 주로 쓰는게 이것과 nltk 의 WordPunctTokenizer


df_train = pd.read_csv("C:/Users/jiu kim/Desktop/computer science/git/first-kaggle/00_source/train.csv")
df_test = pd.read_csv("C:/Users/jiu kim/Desktop/computer science/git/first-kaggle/00_source/test.csv")

print('train set shape = {}'.format(df_train.shape))
print('test set shape = {}'.format(df_test.shape))

# url 제거
def remove_url(text):
    url = re.compile(r'https?://www.')
    text = url.sub(r'', text)
    return text

# html 제거
def remove_html(text):
    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)
    return text

# punctuation 제거
def remove_punctuation(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

'''def remove_punctuation(text):
    table = str.maketrans(string.punctuation, str([' ' for _ in range(len(string.punctuation))]))
    text = text.translate(table)
    return text'''
'''
문자 바꾸기
>>> table = str.maketrans('aeiou', '12345') a는 1 e는 2 ... 에 대응하는 테이블이 만들어짐
>>> 'apple'.translate(table) 테이블에 맞게 문자변경 (기존 apple이 바뀌는게 아님! immutable 이니께)
'1ppl2'
string.punctuation 과 길이가 같은 값을 넣어줘야하는데.. 리스트로 넣어줘야하는 argument면 [' ' for _ in range(len(string.punctuation))] 식으로 넣으면 될지도 
위의 코드 억지로 str() 했는데도 안됨ㅜㅜ 일단 다른방식으로 쓰자
'''
'''
>>> string.punctuation
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
'''
# stopwords 제거
stopwords = set(stopwords.words()) # 앗 이거 함수 안으로 넣으면 안되네?
#nltk의 stopwords는 wordcloud의 stopwords하고 다른가봐 후자는 그냥 단어 리스트를 줬는데 얘는 도구인듯. 이렇게 다시 한번 설정해줘야 한다.
def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

# lemmatization 작업
def lemmatization(text):
    n = WordNetLemmatizer()
    text = ' '.join([n.lemmatize(word) for word in text.split()])
    return text

# stemming 작업
p = PorterStemmer()
def stemming_text(text):
    text = ' '.join([p.stem(word) for word in text.split()]) #stem 뒤에는 리스트가 아니라 string이 들어가야 한다. 즉 리스트화와 stem의 순서를 바꿔줘야함
    return text


def cleaning_text(text):
    #위의 helper function 전부 합친거. 더 예쁘고 간단하게 쓰고싶은데..
    text = remove_url(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = text.lower()
    text = remove_stopwords(text)
    text = lemmatization(text)
    text = stemming_text(text)
    return text

df_total = pd.concat([df_train, df_test])
df_total['text'] = df_total['text'].apply(lambda x: cleaning_text(x))
print(df_total['text'].head())
# text cleaning 완료

# tokenization
# 토큰화 거치면 어떤 데이터타입으로 내보내야 하지? word2vec 이나 glove가 뭘 input 으로 받는지 확인
def tokenization(text):
    '''
    라이브러리 안쓰고 그냥 이렇게도 하길래 나도 해봄 뭐야 제대로 안되는데?
    tokens = re.split('/W+', text)
    return tokens'''
    tokens = word_tokenize(text)
    return tokens

tokens = df_total['text'].apply(lambda x: tokenization(x))
print(tokens.head(10))

# word embedding
model = gensim.models.Word2Vec(size=150, window=5, workers=4, sg=0, min_count=5)
model.build_vocab(tokens)
model.train(sentences=tokens, total_examples=len(tokens), epochs=model.epochs)
word_vectors = model.wv
print(len(word_vectors.vocab))

'''word_vectors = model.wv
count = 0
for word in word_vectors.vocab:
    if count<10:
        print(word)
        count += 1
    else:
        break'''
'''vector = model.wv["earthquak"]  # model.mv가 단어 : 벡터로 이루어진 딕셔너리인듯
print(len(vector))
print(vector)'''

# 이제 모델 만들차례...
# bert를 쓰긴했는데 모델 공부하다 보니까 앞에 기본적인 fcnn이나 머신러닝을 다뤄보지도 않고 이걸 깊게 파고드는 건 시기상조인 것 같아서 일단 돌아가기만 하도록 했다. 스터디는 해야하니까..
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

model = build_model(bert_layer, max_len=160)
print(model.summary())



















