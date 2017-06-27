#!flask/bin/python
import os
import csv
import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time

dest_file = 'data.csv'
UPLOAD_FOLDER = os.getcwd() + '/datasets'
ALLOWED_EXTENSIONS = {'csv'}
training_set_ratio = 80

models = {'Logistic Regression': linear_model.LogisticRegression(),
          'Decision Tree': tree.DecisionTreeClassifier(),
          'Gaussian Naive Bayes': GaussianNB(),
          'Ada Boost Decision Tree Ensemble': AdaBoostClassifier(),
          'Multi-Layer Perceptron': MLPClassifier(solver='lbfgs',
                                                  alpha=1e-5,
                                                  hidden_layer_sizes=(5, 2),
                                                  random_state=1),
          }


def load_data(filename):
	with open(filename, 'r') as dest_f:
		data_iter = csv.reader(dest_f, delimiter=',')
		raw_data = [data for data in data_iter]
		return raw_data


def machine_learning():
	with open(dest_file, 'r') as dest_f:
		data_iter = csv.reader(dest_f, delimiter=',')
		raw_data = [data for data in data_iter]

	# Data preparation
	[row.pop(0) for row in raw_data]  # Remove unnecessary column
	column_names = raw_data.pop(0)
	raw_labels = [row.pop(0) for row in raw_data]
	labels = np.array(list(
		map(lambda x: 0 if x == 'B' else 1, raw_labels)))  # Map it and turn it to a list. 0 for bening, 1 for malignant
	features = np.array(raw_data, dtype='float')

	iteration = 0
	for model_name, model in models.items():
		results = []
		results_f1 = []
		time_results = []
		for i in range(0, 500):
			x_train, x_test, y_train, y_test = train_test_split(features, labels)
			start_time = time.time()
			# Model training and performance testing
			model.fit(x_train, y_train)
			time_results.append(time.time() - start_time)
			results.append(model.score(x_test, y_test))
			predictions = model.predict(x_test)
			f1score = f1_score(y_test, predictions)
			results_f1.append(f1score)
		max_result, min_result, mean, f1_score_mean, time_mean, time_std, acc_std_dev, f1_std_dev = max(results), min(results), np.mean(results), \
		                                                         np.mean(results_f1), np.mean(time_results), \
                                                                 np.std(time_results), np.std(results), np.std(results_f1)
		print("Model: ", model_name)
		print("max: ", max_result)
		print("min: ", min_result)
		print("mean: ", mean)
		print("f1 score mean: ", f1_score_mean)
		print("Time mean: ", np.mean(time_results))
		print("Time standard deviation: ", np.std(time_results))
		plotting_range = list(range(0, 500))
		# Plot individual accuracy
		plt.figure(1)
		plt.title(model_name + " Accuracy Results")
		plt.plot(plotting_range, results)
		plt.xlabel("Iteration")
		plt.ylabel("Accuracy")
		plt.text(0.5, max_result, "Mean Accuracy: " + str(mean))
		plt.text(0.5, max_result - (max_result * 0.005), "Minimum Result: " + str(min_result))
		plt.text(0.5, max_result - (max_result * 0.01), "Maximum Result: " + str(max_result))
		plt.text(0.5, max_result - (max_result * 0.015), "Standard Deviation: " + str(acc_std_dev))
		plt.savefig(model_name + " Accuracy Results")
		plt.clf()
		# Plot all accuracies
		plt.figure(2)
		plt.axis([0, 500, 0.35, 1.0])
		plt.title("Full Accuracy Results")
		p = plt.plot(plotting_range, results)
		color = p[0].get_color()
		plt.text(0.5, 0.8 - (iteration / 20), model_name, color=color)
		plt.xlabel("Iteration")
		plt.ylabel("Accuracy")
		plt.savefig("Full Accuracy Results")
		# Plot individual F1 Score
		plt.figure(3)
		plt.title(model_name + " F1 Score")
		plt.plot(plotting_range, results_f1)
		plt.xlabel("Iteration")
		plt.ylabel("F1 Score")
		plt.text(0.5, max(results_f1), "F1 Score Mean: " + str(f1_score_mean))
		plt.text(0.5, max(results_f1) - max(results_f1) * 0.005, "F1 Score Standard Deviation: " + str(f1_std_dev))
		plt.savefig(model_name + " F1 Score")
		plt.clf()
		# Plot all F1 Scores
		plt.figure(4)
		plt.axis([0, 500, 0.35, 1.0])
		plt.title("Full F1 Score Results")
		p = plt.plot(plotting_range, results_f1)
		color = p[0].get_color()
		plt.text(0.5, 0.7 - (iteration / 20), model_name, color=color)
		plt.xlabel("Iteration")
		plt.ylabel("F1 Score")
		plt.savefig("Full F1 Score Results")
		# Plot individual time figures
		plt.figure(5)
		plt.title(model_name + " Time (s)")
		plt.plot(plotting_range, time_results)
		plt.xlabel("Iteration")
		plt.ylabel("Time")
		plt.text(0.5, max(time_results), "Time Mean: " + str(time_mean))
		plt.text(0.5, max(time_results) - max(time_results) * 0.02, "Time Standard Deviation: " + str(time_std))
		plt.savefig(model_name + " Time Results")
		plt.clf()
		# Plot all time figures
		plt.figure(6)
		plt.title("Full Time Results")
		p = plt.plot(plotting_range, time_results)
		color = p[0].get_color()
		plt.text(0.5, 0.09 - (iteration / 100), model_name, color=color)
		plt.xlabel("Iteration")
		plt.ylabel("Time (s)")
		plt.savefig("Full Time Results")
		iteration += 1


if __name__ == '__main__':
	machine_learning()
