import numpy as np
import tensorflow as tf
import pandas as pd


data = pd.read_csv('gpascore.csv')

#데이터 전처리
# print(data.isnull().sum())
data = data.dropna() #data.fillna(100) 빈칸을 100으로 채워줌
# print(data['gpa'].max())

#list로 담아줌

ydata = data['admit'].values

xdata = []

for i, rows in data.iterrows():
   xdata.append([rows['gre'], rows['gpa'], rows['rank']])



#1. 딥러닝 model 디자인하기
model = tf.keras.models.Sequential([
    # hidden layer  
    # Dense()안에 숫자 : hidden layer node 개수
    # 보통 2의 제곱수로 어림잡음
    tf.keras.layers.Dense(64, activation='tanh'),    #레이어에 activation function 넣기
    tf.keras.layers.Dense(128, activation='tanh'),
    #마지막 레이어에는 항상 예측결과를 뱉어야함, 0~1 사이의 확률은 sigmoid
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),                         
])

#2.model compile하기
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

#3.model 학습(fit)시키기 
# x데이터: 정답 예측에 필요한 인풋 [[380, 3.21, 3]  [660, 3.67, 3]]
# y데이터 : 정답 [정답1, 정답2, 정답3 ...]
# x데이터, y데이터에는 numpy array or tf tensor가 들어가야함
# epochs -> 몇번 연습을 시킬지 더 많이 시킬수록 정확성이 높아진다다
model.fit(np.array(xdata), np.array(ydata), epochs=2000)

#4.학습시킨 모델로 예측해보기
pred = model.predict([ [700, 3.66, 2], [800, 4.0, 1] ])
print(pred)

