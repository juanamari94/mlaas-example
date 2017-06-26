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
from sklearn import svm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import random

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
          'Support Vector Machine with RBF Kernel': svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                                                            decision_function_shape=None, degree=3, gamma='auto',
                                                            kernel='rbf', max_iter=-1, probability=False,
                                                            random_state=None, shrinking=True, tol=0.001,
                                                            verbose=False)}


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

	for model_name, model in models.items():
		results = []
		results_f1 = []
		for i in range(0, 10):
			x_train, x_test, y_train, y_test = train_test_split(features, labels)

			# Model training and performance testing
			model.fit(x_train, y_train)
			results.append(model.score(x_test, y_test))
			predictions = model.predict(x_test)
			f1score = f1_score(y_test, predictions)
			results_f1.append(f1score)
		max_result, min_result, mean, f1_score_mean = max(results), min(results), sum(results) / 10, sum(results_f1) / 10
		print("Model: ", model_name)
		print("max: ", max_result)
		print("min: ", min_result)
		print("mean: ", mean)
		print("f1 score mean: ", f1_score_mean)
		plotting_range = list(range(0, 10))
		# Plot individual accuracy
		plt.figure(1)
		plt.title(model_name + " Accuracy Results")
		plt.plot(plotting_range, results)
		plt.xlabel("Iteration")
		plt.ylabel("Accuracy")
		plt.text(0.5, max_result, "Mean Accuracy: " + str(mean))
		plt.text(0.5, max_result - 0.005, "Minimum Result: " + str(min_result))
		plt.text(0.5, max_result - 0.01, "Maximum Result: " + str(max_result))
		plt.savefig(model_name + " Accuracy Results")
		plt.clf()
		# Plot all accuracies
		plt.figure(2)
		plt.title("Full Accuracy Results")
		plt.plot(plotting_range, results)
		plt.xlabel("Iteration")
		plt.ylabel("Accuracy")
		plt.savefig("Full Accuracy Results")
		# Plot individual F1 Score
		plt.figure(3)
		plt.title(model_name + " F1 Score")
		plt.plot(plotting_range, results_f1)
		plt.xlabel("Iteration")
		plt.ylabel("F1 Score")
		plt.text(0.5, max(results_f1) - 0.01, "F1 Score Mean: " + str(f1_score_mean))
		plt.savefig(model_name + " F1 Score")
		plt.clf()
		# Plot all F1 Scores
		plt.figure(4)
		plt.title("Full F1 Score Results")
		plt.plot(plotting_range, results)
		plt.xlabel("Iteration")
		plt.ylabel("F1 Score")
		plt.savefig("Full F1 Score Results")


if __name__ == '__main__':
	machine_learning()
