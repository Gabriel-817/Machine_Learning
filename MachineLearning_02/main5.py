
from sklearn import datasets # 학습용 데이터

from sklearn.model_selection import train_test_split # 데이터를 학습용과 테스트 용으로 나눌 수 있는 함수

from sklearn.preprocessing import StandardScaler # 데이터 표준화

from sklearn.linear_model import Perceptron # perceptron 머신러닝을 위한 클래스

from sklearn.linear_model import LogisticRegression # 로지스틱 회귀를 위한 클래스

from sklearn.svm import SVC # SVM을 위한 클래스

from sklearn.tree import DecisionTreeClassifier # 의사결정나무를 위한 클래스

# 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # 그리드서치 클래스
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# 교차검증 : from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score # 정확도 계산을 위한 함수

import pickle # 파일 저장
import numpy as np

# from mylib.plotdregion import *

names = None

def step1_get_data() :
    # 아이리스 데이터 추출
    iris = datasets.load_iris()
    # print(iris)
    # 꽃 정보 데이터 추출
    X = iris.data[:150, [2,3]] # 꽃잎 정보
    y = iris.target[:150]      # 꽃 종류
    names = iris.target_names[:2] # 꽃 이름
    # print(X[0])
    # print(y[0])
    # print(names[0])
    return X, y

def step2_learnig() :
    X, y = step1_get_data()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 0) # 학습 데이터와 텍스트 데이터로 나눔
    sc = StandardScaler() # 표준화 작업 : 데이터들을 표준 정규분포로 변환하여 적은 학습횟수와 높은 학습 정확도를 갖기 위해 하는 작업
    # 데이터 표준화
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    # 학습
    #ml = Perceptron(eta0=0.01, max_iter=40, random_state=0) # 몇 번 학습할 건지, 랜덤으로 하지 말고
    #ml = LogisticRegression(C=1000.0, random_state=0)  
    # kernel : 알고리즘 종류, linear, poly, rbf, sigmoid / C : 분류의 기준이 되는 경계면을 조절
    #ml = SVC(kernel='linear', C=1.0, random_state=0) 
    # criterion : 불순도 측정 방식, entropy, gini / max_depth : 노드 깊이의 최대값
    #ml = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    #n_estimators : 포레스트 내의 나무 개수
    # n_jobs : 병렬처리에서 사용할 코어의 개수
    ml = RandomForestClassifier(criterion = 'entropy',n_estimators = 10, max_depth =3, n_jobs =2, random_state=0)
    ml.fit(X_train_std, y_train)
    # parameters = {
    #       'n_estimators'      :[320,330,340],
    # }

    # 학습 정확도 확인
    X_test_std = sc.transform(X_test)
    y_pred = ml.predict(X_test_std)
    print("학습 정확도:", accuracy_score(y_test, y_pred))

    # 성능 확인
    y_true = y_test
    y_hat = ml.predict(X_test_std)
    print("R2 score : ", r2_score(y_true, y_hat))
    print("mean_absolute_error : ",
    mean_absolute_error(y_true, y_hat))
    print("mean_squared_error : ",
    mean_squared_error(y_true, y_hat))

    # 학습이 완료된 객체 저장
    with open('./8.Scikit-RandomForest/ml.dat', 'wb') as fp:
        pickle.dump(sc, fp)
        pickle.dump(ml, fp)

    print("학습 완료")
    
    # 시각화


def step3_using() :
    # 학습이 완료된 객체 복원
    with open('./8.Scikit-RandomForest/ml.dat', 'rb') as fp:
        sc = pickle.load(fp)
        ml = pickle.load(fp)

    X = [
        [1.4, 0.2], [1.3, 0.2],[1.5, 0.2],
        [4.5, 1.5], [4.1, 1.0],[4.5, 1.5],
        [5.2, 2.0], [5.4, 2.3],[5.1, 1.8]
    ]
    X_std = sc.transform(X)
    y_pred = ml.predict(X_std)

    for value in y_pred :
        if value == 0 :
            print('Iris-setosa')
        elif value == 1 :
            print('Iris-versicolor')
        elif value == 2:
            print('Iris-virginica')
 
if __name__ == "__main__":
    step1_get_data()
    step2_learnig()
    step3_using()