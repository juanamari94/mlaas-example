import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score

dest_file = 'data.csv'
training_set_ratio = 80


def main():
	with open(dest_file, 'r') as dest_f:
		data_iter = csv.reader(dest_f, delimiter=',')
		raw_data = [data for data in data_iter]

	# Data preparation
	[row.pop(0) for row in raw_data]# Remove unnecessary column
	column_names = raw_data.pop(0)
	raw_labels = [row.pop(0) for row in raw_data]
	labels = list(map(lambda x: 0 if x == 'B' else 1, raw_labels)) # Map it and turn it to a list. 0 for bening, 1 for malignant
	features = np.array(raw_data)

	# Dataset splitting (80% training, 20% test set)
	dataset_length = len(features)
	training_set_length = (training_set_ratio * dataset_length) // 100
	training_set = features[0:training_set_length]
	training_set_labels = labels[0:training_set_length]
	test_set = features[training_set_length:dataset_length]
	test_set_labels = labels[training_set_length:dataset_length]

	# Model training and performance testing
	#logreg = linear_model.LogisticRegression()
	#logreg.fit(training_set, training_set_labels)
	#print(logreg.predict(training_set))
	#print(accuracy_score(test_set, test_set_labels))


if __name__ == '__main__':
	main()