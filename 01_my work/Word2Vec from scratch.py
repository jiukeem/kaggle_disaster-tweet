# text 는 cleaning 처리된 텍스트(한개의 sentence가 element인 리스트 형태), tokenization과 one-hot encoding은 패키지를 쓴다
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import numpy as np

def tokenization(sentence):
    t = Tokenizer()
    voca = t.fit_on_texts(sentence)

    return voca

def onehot_encoding(sentence):
    encoded=t.texts_to_sequences(sentence)[0]
    one_hot = to_categorical(encoded)

    return one_hot

def preprocess_text(text):
    for sentence in text:
        sentence = tokenization(sentence)
        one_hot = onehot_encoding(sentence)
    return one_hot
# 여기까지 전처리인데 케라스를 안써봐서 아마 제대로 안돌아갈듯? 나중에 보충


def generate_word2vec_exmaple_set(one_hot, window_size):
    X = []
    Y = []
    n = window_size
    for example in one_hot:
        for i in range(len(example) - 2*n):
            X.append([example[i: i+2], example[i+3: i+5]])
            Y.append(example[i+2])

    return X, Y
# X의 형태는 [[], [], [], [], ...  ,[]] 이고 각 element 안에는
# [one-hot], [one-hot], [ont-hot], ...  , [one-hot] <- 2n개 만큼

def embedding_matrix_random_initialization(X, embedding_size):
    m = embedding_size
    v = X.shape(1)
    W1 = np.random.randn((v, m))
    W2 = np.random.randn((m, v))
    b2 = np.random.randn((1, v))

    return W1, W2, b2

def word2vec_forward_propagation(X, Y, W1, W2, b2):
    loss = 0
    count = 0
    for example in X:
        p_total = 0
        for vector in example:
            p_total += np.matmul(vector, W1)
        p = p_total/ len(example)
        z = np.matmul(p, W2) + b2
        a = softmax(z) #소프트맥스 어디 라이브러리에서 가져와야되지?
        loss += - (np.sum(np.multiply(a, Y[count])))
        count += 1
    cost = loss / len(X)

    return z, a, cost

def word2vec_back_propagation(z, a, cost):

    return dW1, dW2, db2

def parameter_update(W1, W2, b2, dW1, dW2, db2):
    W1 -= dW1
    W2 -= dW2
    b2 -= db2

    return W1, W2, b2

def fit_model(X, Y, embedding_size, iteration_num = 500):
    W1, W2, b2 = embedding_matrix_random_initialization(X, embedding_size)
    for _ in range(iteration_num):
        z, a, cost = word2vec_forward_propagation(X, Y, W1, W2, b2)
        dW1, dW2, db2 = word2vec_back_propagation(z, a, cost)
        W1, W2, b2 = parameter_update(W1, W2, b2, dW1, dW2, db2)

    # accuracy check
    return W1, accuracy



