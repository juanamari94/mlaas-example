from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import data_parser

class BaseModel:
	def __init__(self, model):
		self.model = model

	def train(self):
		return self.model.fit(self.x_train, self.y_train)

	def accuracy_metrics(self):
		return self.model.score(self.x_test, self.y_test)

	def calculate_f1_score(self):
		test_predictions = self.model.predict(self.x_test)
		return f1_score(self.y_test, test_predictions)

class SupervisedBinaryClassificationModel(BaseModel):

	def __init__(self, raw_training_set, raw_predict_set, model):
		super().__init__(model)
		self.column_names = raw_training_set.pop(0)
		raw_labels = data_parser.parse_labels(raw_training_set)
		self.classes = list(set(raw_labels))
		if len(self.classes) != 2:
			raise Exception("A binary classificator can only have two classes.")
		self.labels = data_parser.parse_classification_labels(raw_labels, self.classes)
		self.features = data_parser.parse_features(raw_training_set)
		self.predict_set = data_parser.parse_features(raw_predict_set)
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.features, self.labels)

	def predict(self):
		predictions = self.model.predict(self.predict_set)
		results = []
		for i in range(0, len(predictions)):
			results.append((int(predictions[i]), self.predict_set[i]))
		return results

class SupervisedEstimationModel(BaseModel):

	def __init__(self, raw_training_set, raw_predict_set, model):
		super().__init__(model)
		self.column_names = raw_training_set.pop(0)
		raw_labels = data_parser.parse_labels(raw_training_set)
		self.labels = data_parser.parse_estimation_labels(raw_labels)
		self.features = data_parser.parse_features(raw_training_set)
		self.predict_set = data_parser.parse_features(raw_predict_set)
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.features, self.labels)

	def predict(self):
		predictions = self.model.predict(self.predict_set)
		results = []
		for i in range(0, len(predictions)):
			results.append((float(predictions[i]), self.predict_set[i]))
		return results