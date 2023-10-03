import numpy as np
from network import network
from activations import softmax
np.random.seed(5)

list_neu = [2 ,5, 5, 2]
trainer = np.reshape([
    [0, 0], [0, 1], [1, 0], [1, 1]
], (4, 2, 1))

result = np.reshape([
    [0, 0], [0, 1], [1, 0], [0, 0]
], (4, 2, 1))
# result = np.reshape([
#     [0], [1], [1], [0]
# ], (4, 1, 1))

neural = network(neurons=list_neu)
neural.layers[-1] = softmax()

ty = neural.train(trainer, result, cases=3000)
# print(ty)
f = lambda x : (np.tanh(x[0]) + np.tanh(x[1])) / 2

test_x = np.zeros(shape=(1000, 2, 1))
test_y = np.zeros(shape=(1000, 1, 1))

# print(test_x)
# print(test_y)

for index, elem in enumerate(test_x) :
    one, two = np.random.rand(), np.random.rand()
    test_x[index][0] = one
    test_x[index][1] = two
    test_y[index] = f([one, two]) 

# neural.train(test_x, test_y, cases=100)
while True :
    tests = str(input('entrez des cas separer par espaces : '))
    tests = tests.split(' ')
    tests = list(map(float, tests))
    tests = np.array(tests).reshape((2, 1)) 
    print('test : ', tests)

    output = neural.predict(input=tests)
    print('the result is : ', output)
    print('the right answer is : ', f(tests))