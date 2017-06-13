#!flask/bin/python
import os
from flask import Flask, render_template, request, flash, redirect, url_for
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
          'Support Vector Machine with RBF Kernel': svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                                                            decision_function_shape=None, degree=3, gamma='auto',
                                                            kernel='rbf', max_iter=-1, probability=False,
                                                            random_state=None, shrinking=True, tol=0.001, verbose=False)}


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
	if (train_test_set and allowed_file(train_test_set.filename))\
		and (predict_set and allowed_file(predict_set.filename)):
		train_test_set_filename = secure_filename(train_test_set.filename)
		train_test_set.save(os.path.join(app.config['UPLOAD_FOLDER'], train_test_set_filename))
		predict_set_filename = secure_filename(predict_set.filename)
		predict_set.save(os.path.join(app.config['UPLOAD_FOLDER'], predict_set_filename))
		return "success"
	return 'failure'

if __name__ == '__main__':
	app.run(debug=True)