import numpy as np


def parse_features(raw_set):
	return np.array(raw_set, dtype='float')


def parse_labels(raw_set):
	labels = [row.pop(0) for row in raw_set]
	return labels


def parse_classification_labels(raw_labels, classes):
	labels = []
	for label in raw_labels:
		if label == classes[0]:
			labels.append(0)
		else:
			labels.append(1)
	return np.array(labels, dtype='float')


def parse_estimation_labels(raw_labels):
	return np.array(raw_labels, dtype='float')
