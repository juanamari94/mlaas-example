#!flask/bin/python
import os
from flask import Flask, render_template, request, flash, abort
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
from werkzeug.utils import secure_filename
from supervised_binary_classification_model import SupervisedBinaryClassificationModel

dest_file = 'data.csv'
UPLOAD_FOLDER = os.getcwd() + '/datasets'
ALLOWED_EXTENSIONS = {'csv'}
training_set_ratio = 80
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

	for model_name, model in models.items():
		results = []
		results_f1 = []
		for i in range(0, 10000):
			x_train, x_test, y_train, y_test = train_test_split(features, labels)

			# Model training and performance testing
			model.fit(x_train, y_train)
			results.append(model.score(x_test, y_test))
			predictions = model.predict(x_test)
			f1score = f1_score(y_test, predictions)
			results_f1.append(f1score)
		print("Model: ", model_name)
		print("max: ", max(results))
		print("min: ", min(results))
		print("mean: ", sum(results) / 10000)
		print("f1 score mean: ", sum(results_f1) / 10000)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/about')
def about():
	return render_template('about.html')


@app.route("/requirements")
def requirements():
	return render_template('requirements.html')


@app.route('/classify')
def classify_view():
	return render_template("classify_estimate.html")


@app.route('/estimate')
def estimate_view():
	return render_template("classify_estimate.html")


@app.route('/upload', methods=['POST'])
def upload_file():
	if ('train_test_set' or 'predict_set') not in request.files:
		flash('No file part')
		return "failure"
	train_test_set = request.files['train_test_set']
	predict_set = request.files['predict_set']
	if train_test_set.filename == '' or predict_set.filename == '':
		flash('No selected file')
		return "failure"
	if (train_test_set and allowed_file(train_test_set.filename)) \
		and (predict_set and allowed_file(predict_set.filename)):
		train_test_set_filename = secure_filename(train_test_set.filename)
		train_test_set_path = os.path.join(app.config['UPLOAD_FOLDER'], train_test_set_filename)
		train_test_set.save(train_test_set_path)
		predict_set_filename = secure_filename(predict_set.filename)
		predict_set_path = os.path.join(app.config['UPLOAD_FOLDER'], predict_set_filename)
		predict_set.save(predict_set_path)
		referrer = request.referrer
		function = referrer[referrer.rfind('/') + 1:]
		raw_training_data = load_data(train_test_set_path)
		raw_prediction_data = load_data(predict_set_path)
		if function == 'classify':
			classifier = SupervisedBinaryClassificationModel(raw_training_data, raw_prediction_data, \
			                                                 linear_model.LogisticRegression())
			classifier.train()
			print(raw_prediction_data)
			predictions = classifier.predict()
			result = '<table><tr>'
			for column_name in classifier.column_names:
				result += '<th>' + column_name + '</th>'
			result += '</tr>'
			for tup in predictions:
				result += "<tr><td>" + str(classifier.classes[tup[0]]) + "</td>"
				for feature in tup[1]:
					result += "<td>" + str(feature) + "</td>"
				result += "</tr>"
			return result
		elif function == 'estimate':
			print("b")
		else:
			return abort(400)
		return "success"
	return 'failure'


@app.route('/test')
def test():
	machine_learning()
	return "test"


if __name__ == '__main__':
	app.run(debug=True)
