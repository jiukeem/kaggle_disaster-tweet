import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import string

df_train = pd.read_csv("C:/Users/jiu kim/Desktop/computer science/git/first-kaggle/00_source/train.csv")
df_test = pd.read_csv("C:/Users/jiu kim/Desktop/computer science/git/first-kaggle/00_source/test.csv")

print('train set shape = {}'.format(df_train.shape))
print('test set shape = {}'.format(df_test.shape))

# url 제거
def url_remove(text):
    url = re.compile(r'https?://www.')
    return url.sub(r'', total)

# html 제거
def html_remove(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', total)

# punctuation 제거
def punctuation_remove(text):
    table = str.maketrans(string.punctuation, '')
    return text.translate(table)
'''
문자 바꾸기
>>> table = str.maketrans('aeiou', '12345') a는 1 e는 2 ... 에 대응하는 테이블이 만들어짐
>>> 'apple'.translate(table) 테이블에 맞게 문자변경 (기존 apple이 바뀌는게 아님! immutable 이니께)
'1ppl2'
'''
'''
>>> string.punctuation
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
'''










