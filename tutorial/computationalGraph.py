'''
Created on Jun 27, 2017
 
@author: wangxing
'''
import numpy as np
np.random.seed(0)
 
N, D = 3, 4
# randn for standard normal
x = np.random.randn(N, D)   # size of 3x4
y = np.random.randn(N, D)
z = np.random.randn(N, D)
# print 'x=', x, '\ny=', y, '\nz=', z
 
a = x * y 
b = a + z 
c = np.sum(b)
print c 
grad_c = 1.0
grad_b = grad_c * np.ones((N, D))
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_x = grad_a * y 
grad_y = grad_a * x 
print 'grad_x=', grad_x, '\ngrad_y=', grad_y, '\ngrad_z=', grad_z
print '#########################################'
#########################################
# PyTorch
#########################################
import torch 
from torch.autograd import Variable 
N, D = 3, 4
x = Variable(torch.randn(N, D), requires_grad = True)
# x = Variable(torch.from_numpy(x), requires_grad = True)
y = Variable(torch.randn(N, D), requires_grad = True)
z = Variable(torch.randn(N, D), requires_grad = True)
    # Forward Pass
a = x * y 
b = a + z 
c = torch.sum(b)
    # Backward 
c.backward() 
# print x.grad.data
# print y.grad.data 
# print z.grad.data 

# x = torch.randn(3)
# x = Variable(x, requires_grad = True)
# y = Variable(torch.Tensor(3), requires_grad=True)
# y = x * 2 
# while y.data.norm() < 1000:
#     y = y * 2
# print y
# 
# gradients = torch.FloatTensor([.1, 1., 1e-4])
# y.backward(gradients)
# print x.grad
#########################################
# TensorFlow 
#########################################
import tensorflow as tf 
np.random.seed(0) 
N, D = 3, 4

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = tf.placeholder(tf.float32)

a = x * y
b = a + z 
c = tf.reduce_sum(b)

grad_x, grad_y, grad_z = tf.gradients(c, [x, y, z])

with tf.Session() as sess:
    values = {
        x : np.random.randn(N, D), 
        y : np.random.randn(N, D), 
        z : np.random.randn(N, D)
    }
    out = sess.run([c, grad_x, grad_y, grad_z], feed_dict = values)
    c_val, grad_x_val, grad_y_val, grad_z_val = out


print c_val
print 'grad_x=', grad_x_val, '\ngrad_y=', grad_y_val, '\ngrad_z=', grad_z_val



