from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np


class ClassifiedLearningDataSet:

	def __init__(self, training_set, predict_set, model):
		self.model = model
		self.training_set = np.array(training_set, dtype='float')
		self.predict_set = predict_set
		raw_labels = [row.pop(0) for row in training_set]
		labels = np.array(list(map(lambda x: 0 if x == 'B' else 1, raw_labels)))
		self.x_train, self.y_train, self.x_test, self.y_test = train_test_split(training_set, labels)

	def train(self):
		return self.model.fit(self.x_train, self.y_train)

	def predict(self):
		return self.model.predict(self.predict_set)

	def accuracy_metrics(self):
		return self.model.score(self.x_test, self.y_test)

	def calculate_f1_score(self):
		test_predictions = self.model.predict(self.x_test)
		return f1_score(self.y_test, test_predictions)
