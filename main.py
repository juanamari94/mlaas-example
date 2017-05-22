#!flask/bin/python
from flask import Flask
import csv
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

dest_file = 'data.csv'
training_set_ratio = 80
app = Flask(__name__)


def machine_learning():
	with open(dest_file, 'r') as dest_f:
		data_iter = csv.reader(dest_f, delimiter=',')
		raw_data = [data for data in data_iter]

	# Data preparation
	[row.pop(0) for row in raw_data]# Remove unnecessary column
	column_names = raw_data.pop(0)
	raw_labels = [row.pop(0) for row in raw_data]
	labels = np.array(list(map(lambda x: 0 if x == 'B' else 1, raw_labels))) # Map it and turn it to a list. 0 for bening, 1 for malignant
	features = np.array(raw_data, dtype='float')

	results = []
	for i in range(0, 10000):
		x_train, x_test, y_train, y_test = train_test_split(features, labels)

		# Model training and performance testing
		logreg = linear_model.LogisticRegression()
		logreg.fit(x_train, y_train)
		results.append(logreg.score(x_test, y_test))
	print(results)
	print("max: ", max(results))
	print("min: ", min(results))
	print("mean: ", sum(results) / 10000)


@app.route('/')
def index():
	return "anime"


@app.route('/test')
def machine():
	machine_learning()
	return "xd"


if __name__ == '__main__':
	app.run(debug=True)