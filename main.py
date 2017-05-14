import csv
import numpy as np

dest_file = 'wisconsin_u_breast_cancer_dataset.csv'


def main():
	with open(dest_file, 'r') as dest_f:
		data_iter = csv.reader(dest_f, delimiter=',')
		data = [data for data in data_iter]
	feature_names = data.pop(0)
	raw_dataset = np.matrix(data)
	features = raw_dataset[:, 0]
	


if __name__ == '__main__':
	main()