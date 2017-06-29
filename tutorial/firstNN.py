'''
Created on Jun 27, 2017

@author: wangxing
'''
import torch 
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F 


############### Defining the Network ##################
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 convolution 
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation y=Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # max pooling over a 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
print net

params = list(net.parameters())
print len(params)
print params[0].size()

input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
net.zero_grad()
out.backward(torch.randn(1, 10))

############# Loss Function ###################
output = net(input)
target = Variable(torch.arange(1, 11))
criterion = nn.MSELoss()
loss = criterion(output, target)
print loss.creator
print loss.creator.previous_functions[0][0]
print loss.creator.previous_functions[0][0].previous_functions[0][0]

############ Backpropagation #################
net.zero_grad()
print "conv1.bias.grad before backward"
print net.conv1.bias.grad 
loss.backward() 
print "conv1.bias.grad after backward"
print net.conv1.bias.grad

############ Update the Weights ##############
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()


# #################################################
# # TensorFlow 
# #################################################
# import numpy as np
# import tensorflow as tf 
# 
# N, D, H = 64, 1000, 1000
# x = tf.placeholder(tf.float32, shape=(N, D))
# y = tf.placeholder(tf.float32, shape=(N, D))
# # w1 = tf.placeholder(tf.float32, shape=(D, H))
# # w2 = tf.placeholder(tf.float32, shape=(H, D))
# w1 = tf.Variable(tf.random_normal((D, H)))
# w2 = tf.Variable(tf.random_normal((H, D)))
# 
# ## 2-layer RELU
# h = tf.maximum(tf.matmul(x, w1), 0)
# y_pred = tf.matmul(h, w2)
# # L2 loss
# diff = y_pred - y 
# #loss = tf.reduce_mean(tf.reduce_sum(diff**2, axis=1))
# loss = tf.losses.mean_squared_error(y_pred, y)
# 
# 
# # grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
# # learning_rate = 1e-5
# # new_w1 = w1.assign(w1 - learning_rate * grad_w1)
# # new_w2 = w2.assign(w2 - learning_rate * grad_w2)
# # updates = tf.group(new_w1, new_w2)
# 
# optimizer = tf.train.GradientDescentOptimizer(1e-3)
# updates = optimizer.minimize(loss)
# 
# 
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     values = {
#         x : np.random.randn(N, D), 
#         y : np.random.randn(N, D)
# #         w1 : np.random.randn(D, H), 
# #         w2 : np.random.randn(H, D), 
#     }
#     for t in range(50):
#         out = sess.run([loss, updates], feed_dict = values)
# #         loss_val, grad_w1_val, grad_w2_val = out   
# #         values[w1] -= learning_rate * grad_w1_val
# #         values[w2] -= learning_rate * grad_w2_val  
# # print 'loss=', loss_val, '\ngrad_w1=', grad_w1_val, '\ngrad_w2=', grad_w2_val
