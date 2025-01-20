import tensorflow as tf

height = [170, 180, 175, 160]
shoesize = [260, 270, 265, 255]

a = tf.Variable(0.1)
b = tf.Variable(0.2)




# shoesize = height * a + b
# 실제값 - 예측값
def 손실함수():
    return tf.square(260 - (height * a + b))



#경사하강법 도와주는 친구
opt = tf.keras.optimizers.Adam(learning_rate= 0.1)

#경사하강법 300번 해준다
for i in range(300):
    #경사하강법 1번해준다
    opt.minimize(손실함수, var_list=[a,b])
    print(a.numpy(), b.numpy())

