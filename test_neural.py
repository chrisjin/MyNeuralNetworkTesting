import numpy as np

def softmax(vec):
    return np.exp(vec) / np.sum(np.exp(vec), axis = 0)

def relu(vec, alpha, beta):
    return np.piecewise(vec, [vec <= 0, vec > 0], 
        [lambda x: -alpha * x, lambda x: beta * x])

def drelu(vec, alpha, beta):
    return np.piecewise(vec, [vec <= 0, vec > 0], [-alpha, beta])

train_data = np.loadtxt("mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt("mnist_test.csv", 
                       delimiter=",") 

train_X = train_data[:, 1:].T
train_Y = train_data[:, 0].T
print("Shapes: ", train_data.shape, test_data.shape)
print("XY Shapes: ", train_X.shape, train_Y.shape)
train_Y_vec = np.zeros((10, len(train_Y)))
for i in range(len(train_Y)):
    train_Y_vec[int(train_Y[i]), i] = 1

np.random.seed(3)

input_dim = train_X.shape[0]
layer1 = 17
layer2 = 18
output_dim = 10
w1 = np.random.rand(layer1, input_dim) * 0.001
w2 = np.random.rand(layer2, layer1) * 0.001
w3 = np.random.rand(output_dim, layer2) * 0.001
b1 = np.zeros((layer1, 1))
b2 = np.zeros((layer2, 1))
b3 = np.zeros((output_dim, 1))
relu_alpha = 0.001
relu_beta = 1

### forward
def forward(input):
    z1 = np.dot(w1, input) + b1
    a1 = relu(z1, relu_alpha, relu_beta)
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2, relu_alpha, relu_beta)
    z3 = np.dot(w3, a2) + b3
    output = -np.log(softmax(z3))
    return {
        'input': input,
        'z1': z1,
        'a1': a1,
        'z2': z2,
        'a2': a2,
        'z3': z3,
        'output': output
    }

def loss(forward_results, y):
    output = forward_results['output']
    return np.sum(np.multiply(y, output))/y.shape[1]

# backward(result, train_Y_vec)
def backward(forward_results, y):
    z1 = forward_results['z1']
    z2 = forward_results['z2']
    a1 = forward_results['a1']
    a2 = forward_results['a2']
    z3 = forward_results['z3']
    input = forward_results['input']
    dw3a2 = -y + softmax(z3)
    dw3 = np.dot(dw3a2, a2.T) / y.shape[1]
    db3 = np.sum(dw3a2, axis = 1, keepdims=True) / y.shape[1]
    dz2 = np.multiply(np.dot(w3.T,  dw3a2), drelu(z2, relu_alpha, relu_beta))
    dw2 = np.dot(dz2, a1.T) / y.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), drelu(z1, relu_alpha, relu_beta))
    dw1 = np.dot(dz1, input.T) / y.shape[1]
    db1 = np.sum(dz1, axis = 1, keepdims=True) / y.shape[1]
    return {
        'dw1': dw1,
        'dw2': dw2,
        'dw3': dw3,
        'db1': db1,
        'db2': db2,
        'db3': db3
    }

rate = 0.01
for i in range(1, 100):
    result = forward(train_X)
    the_loss = loss(result, train_Y_vec)
    back = backward(result, train_Y_vec)
    dw1 = back['dw1']
    dw2 = back['dw2']
    dw3 = back['dw3']
    db1 = back['db1']
    db2 = back['db2']
    db3 = back['db3']
    w1 -= rate * dw1
    w2 -= rate * dw2
    w3 -= rate * dw3
    b1 -= rate * db1
    b2 -= rate * db2
    b3 -= rate * db3
    print(the_loss)



test_X = test_data[:, 1:].T
test_Y = test_data[:, 0].T
print('test X,Y', test_X.shape, test_Y.shape)

test_forward = forward(test_X)['output']
print('output shape:', test_forward.shape)

wrong = np.sum(abs(test_Y - np.argmin(test_forward, axis = 0)) > 0)
total = len(test_Y)
print((total - wrong) / total)
