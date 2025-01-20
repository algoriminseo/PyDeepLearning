import tensorflow as tf

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

a = tf.Variable(0.1)
b = tf.Variable(0.1)



# mean squared error 한번에 loss값 계산해줌
def 손실함수(a, b):
    예측_y = train_x * a + b 
    return tf.keras.losses.mse(train_y, 예측_y)


#경사하강법 도와주는 친구 
opt = tf.keras.optimizers.Adam(learning_rate= 0.01)

#경사하강법 2900번 해준다
for i in range(2900):
    #경사하강법 1번해준다
    opt.minimize(lambda: 손실함수(a, b), var_list=[a,b])
    # print(a.numpy(), b.numpy())
for i in range(len(train_x)):
    print(train_x[i] * a.numpy() + b.numpy())

#만드는 순서 1. 모델 만들기
#2. optimizer, 손실함수 정하기
#3. 학습하기(경사하강으로 변수값 업데이트하기)
