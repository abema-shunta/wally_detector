from six.moves import cPickle
import numpy as np
import argparse
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import matplotlib.pyplot as plt
import pdb
import six


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

all_data = unpickle("./data.pkl")
x_all = np.asarray(all_data["data"]).astype(np.float32)
y_all = np.asarray(all_data["label"]).astype(np.int32)

x_train, x_test = np.split(x_all, [18000])
y_train, y_test = np.split(y_all, [18000])

## Build Model
model = FunctionSet( 
  l1 = F.Linear(784, 200),
  l2 = F.Linear(200, 80),
  l3 = F.Linear(80, 20),
  l4 = F.Linear(20, 2)
)

optimizer = optimizers.SGD()
optimizer.setup(model)

def forward(x_data, y_data):
  x = Variable(x_data)
  t = Variable(y_data)
  h1 = F.relu(model.l1(x))
  h2 = F.relu(model.l2(h1))
  h3 = F.relu(model.l3(h2))
  y = model.l4(h3)
  return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

accuracy_data = []

batchsize = 1000
datasize = 18000  
for epoch in range(40):
  print('epoch %d' % epoch)
  indexes = np.random.permutation(datasize)
  for i in range(0, datasize, batchsize):
    x_batch = x_train[indexes[i : i + batchsize]]
    y_batch = y_train[indexes[i : i + batchsize]]
    optimizer.zero_grads()
    loss, accuracy = forward(x_batch, y_batch)
    loss.backward()
    accuracy_data.append(accuracy.data)
    optimizer.update()

sum_loss, sum_accuracy = 0, 0

plt.plot(accuracy_data, 'k--')
plt.show()
plt.savefig("accuracy.png")

for i in range(0, 3000, batchsize):
  x_batch = x_test[i : i + batchsize]
  y_batch = y_test[i : i + batchsize]
  loss, accuracy = forward(x_batch, y_batch)
  sum_loss      += loss.data * batchsize
  sum_accuracy  += accuracy.data * batchsize

mean_loss     = sum_loss / 3000
mean_accuracy = sum_accuracy / 3000

print('mean_loss %.2f' % mean_loss)
print('mean_accuracy %d' % (mean_accuracy * 100))

if (mean_accuracy * 100) > 90:
  with open('trained_model.pkl', 'wb') as output:
    six.moves.cPickle.dump(model, output, -1)
    print "model has saved, it has enough quality as trained model :)"