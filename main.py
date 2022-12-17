import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv('data/tips.csv')

feature_array = np.array(data_frame.total_bill)
label_array = np.array(data_frame.tip)

total_cases = feature_array.shape[0]

y_matrix = np.matrix(label_array).T
x_matrix = np.hstack((np.ones((total_cases, 1)), np.matrix(feature_array).T))

y_predicted = np.zeros(total_cases)


# required functions for matrix calculations

# get weight matrix
def get_weight_matrix(current_x, x_matrix, tau):
    weight_matrix = np.mat(np.eye(total_cases))
    for j in range(total_cases):
        distance_vector = current_x - x_matrix[j]
        weight_matrix[j, j] = np.exp(-1 * distance_vector*distance_vector.T / (2.0 * tau ** 2))
    return weight_matrix


# get theta matrix
def get_theta_matrix(current_x, x_matrix, y_matrix, tau):
    weight_matrix = get_weight_matrix(current_x, x_matrix, tau)
    theta_matrix = np.linalg.inv(x_matrix.T * (weight_matrix * x_matrix)) * x_matrix.T * weight_matrix * y_matrix
    return theta_matrix


# get final y predicted
def get_y_for_all_x(current_x, x_matrix, y_matrix, tau):
    theta_matrix = get_theta_matrix(current_x, x_matrix, y_matrix, tau)
    y_predicted_result = current_x * theta_matrix
    return y_predicted_result


# main loop
for i in range(total_cases):
    y_predicted[i] = get_y_for_all_x(x_matrix[i], x_matrix, y_matrix, 1)

# Plotting the results
sorted_indexes = x_matrix[:, 1].argsort(0)
sorted_x = x_matrix[sorted_indexes][:, 0]

figure = plt.figure()
axes = figure.add_subplot(1, 1, 1)
axes.scatter(feature_array, label_array, color='blue')

axes.plot(sorted_x[:, 1], y_predicted[sorted_indexes], color='green', linewidth=4)

plt.xlabel('Total Bill')
plt.ylabel('Tip')

plt.show()
