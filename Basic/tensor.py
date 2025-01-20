import tensorflow as tf

tensor = tf.constant([[3,5,4,5]])
tensor2 = tf.constant([[6,7,8,9]])
tensor2rev = tf.transpose(tensor2)
tensor3 = tf.constant( [ [1,2], 
                        [3,4]])

#3개의 0으로 구성된 행렬을 2개 만들고 그거를 또 2개만듬
tensor4 = tf.zeros([2, 2, 3])  

w = tf.Variable(1.0)
w.assign(2)
print(w.numpy())


node1 = tf.matmul(tensor2, tensor2rev)
print(node1)