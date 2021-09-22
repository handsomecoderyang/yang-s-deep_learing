import numpy as np
import matplotlib.pyplot as plt


def error_comute(w, b, points):
    error = np.mat(points)[:, :-1] * np.mat(w).T + b - np.mat(points)[:, -1]
    error_sum = error.T * error
    print('---------------', error.shape)
    # error = sum(np.mat(points[:, :-1]) * np.mat(w).T + b - np.mat(points[:, -1])**2)
    return error_sum
# print(error_comute(2, 1, np.mat([[1, 1], [1, 3]])))

def gradient_desent(w_current, b_current, points, learing_rate):
    w = w_current - 2 * (np.mat(points)[:, :-1] * np.mat(w_current).T + b_current - np.mat(points)[:, -1]).T * np.mat(points)[:, :-1] / 100 * learing_rate
    b = b_current - np.sum(2 * (np.mat(points)[:, :-1] * np.mat(w_current).T + b_current - np.mat(points)[:, -1]).T) / 100 * learing_rate
    return w, b

def main_function(initial_w, initial_b, points, iteration = 100, learing_rate=0.01):
    w = initial_w
    b = initial_b
    for i in range(iteration):
        w, b = gradient_desent(w, b, points, learing_rate)
        loss = error_comute(w, b, points)
        print("it is the {} iteration, the loss is{}".format(i+1, loss))
    return w, b

datamat = np.zeros((100, 2))
num_matrix = np.random.randn(100, 2)
datamat[:, 0] = num_matrix[:, 0]
datamat[:, 1] = 2 * datamat[:, 0] + 0.1 * num_matrix[:, 1]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datamat[:, 0], datamat[:, 1])
plt.show()
w, b = main_function(0, 0, datamat, 10000, learing_rate=0.001)
print(w, b)

