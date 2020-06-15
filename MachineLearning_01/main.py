import numpy as np
import pandas as pd
from perceptron import Perceptron
from time import time
import pickle 

def step1_get_data():
    # iris.data 파일에서 데이터를 읽어온다.
    df = pd.read_csv('./iris.data', header=None)
    # print(df)
    # 꽃잎 데이터를 추출한다.
    X = df.iloc[0:100, [2,3]].values
    # print(X)
    # 꽃 종류 데이터를 추출한다.
    y = df.iloc[0:100, 4].values
    y = np.where(y=='Iris-setosa', 1, -1)
    # print(y)
    return X,y

def step2_learning():
    ppn = Perceptron(eta=0.1)
    data = step1_get_data()
    X = data[0]
    y = data[1]
    # 학습한다.
    ppn.fit(X, y)
    print(ppn.errors_)
    print(ppn.w_)
    # 학습된 객체를지정한다.
    # 학습이 완료된 객체를 파일로 저장한다.
    with open('./3.IrisPerceptron/perceptron.dat', 'wb') as fp:
        pickle.dump(ppn, fp)
    print("학습 완료")

def step3_using():
    # 파일로 부터 객체를 복원한다.
    with open('./3.IrisPerceptron/perceptron.dat', 'rb') as fp:
        ppn = pickle.load(fp)
    
    while True:
        a1 = input("너비를 입력하세요 : ")
        a2 = input("길이를 입력하세요 : ")

        X = np.array([float(a1), float(a2)])

        result = ppn.predict(X)
        if result == 1:
            print('결과 : Iris-setosa')
        else:
            print('결과 : Iris-versicolor')

# 퍼셉트론_1
    # def step1_learning() :
    #     # 학습과 테스트를 위해 사용할 데이터
    #     X = np.array([[0,0], [0,1],[1,0],[1,1]])
    #     y = np.array([-1, -1,-1, 1])
    #     # 퍼셉트론 객체 생성
    #     ppn = Perceptron(eta = 0.1)
    #     # 학습
    #     stime = time()
    #     ppn.fit(X,y)
    #     etime = time()
    #     print("학습에 걸린 시간 :",(etime - stime))
    #     print("학습 중 오차가 난 개수 :", ppn.errors_)

    #     # 학습이 완료된 객체를 파일로 저장
    #     with open('/Users/kimyihwan/20191216/kaggle/example/2.Perceptron/perceptron.dat','wb') as fp :
    #         pickle.dump(ppn,fp)
        
    #     print("머신러닝 학습 완료")

    # def step2_using() :
    #     # 파일로부터 객체 복원
    #     with open('/Users/kimyihwan/20191216/kaggle/example/2.Perceptron/perceptron.dat','rb') as fp :
    #         ppn = pickle.load(fp)
        
    #     while True :
    #         a1 = input("첫 번째 2진수를 입력해주세요 :")
    #         a2 = input("두 번째 2진수를 입력해주세요 :")

    #         X = np.array([int(a1), int(a2)])
    #         # 계산된 결과 가져옴
    #         result = ppn.predict(X)
    #         if result == 1:
    #             print("결과 : 1")
    #         else :
    #             print("결과 : 0")

if __name__ == "__main__":
# 퍼셉트론_1
    # step1_learning()
    # step2_using()
# Iris
    step1_get_data()
    step2_learning()
    step3_using()