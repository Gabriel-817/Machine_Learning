# step4_learning.py
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle
from time import time
import pandas as pd
from step3_word_tokenizer import tokenizer, tokenizer_porter, tokenizer_stopwordsr

def step4_learning() :
    # csv 파일에서 데이터를 읽어온다.
    df = pd.read_csv('./9.Sentiment/data/refined_movies_review.csv')
    # 테스트, 학습데이터로 나눈다.
    X_train = df.loc[:35000 - 1, 'review'].values
    y_train = df.loc[:35000 - 1, 'sentiment'].values

    X_test = df.loc[35000:50000, 'review'].values
    y_test = df.loc[35000:50000, 'sentiment'].values

    # X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3, random_state=0)
    # print(y_test)
    # print(len(X_train), len(X_test))

    # 단어장을 만들어주는 객체 생성
    tfidf = TfidfVectorizer(lowercase=False, tokenizer=tokenizer_porter)
    # tfidf = TfidfVectorizer(lowercase=False, tokenizer=tokenizer_stopwordsr)
    # 데이터를 학습하기 위한 객체
    logistic = LogisticRegression(C=10.0, penalty='l2', random_state=0)
    # 파이프 라인 설정
    pipeline = Pipeline([('vect', tfidf), ('clf', logistic)])

    # 학습한다.
    stime = time()
    print('학습 시작')
    pipeline.fit(X_train, y_train)
    print('학습 종료')
    print('총 학습시간 : %d' % (time() - stime))

    # 테스트
    y_pred = pipeline.predict(X_test)
    print("정확도 : %.3f" % accuracy_score(y_test, y_pred))

    # 성능 확인
    y_true = y_test
    y_hat = y_pred
    print("R2 score : ", r2_score(y_true, y_hat))
    print("mean_absolute_error : ", mean_absolute_error(y_true, y_hat))
    print("mean_squared_error : ", mean_squared_error(y_true, y_hat))

    # 학습이 완료된 객체를 저장한다.
    with open('./9.Sentiment/data/pipe.dat', 'wb') as fp :
        pickle.dump(pipeline, fp)

    print('저장완료')
