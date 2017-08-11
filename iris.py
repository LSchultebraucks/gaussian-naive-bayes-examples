#
# iris dataset
# 150 total entries
# features are: sepal length in cm, sepal width in cm,petal length in cm        - petal width in cm\n 
# labels names: setosa, versicolor, virginica
#
# used algorithm: Gaussian Naive Bayes (GaussianNB)
#
# accuracy ~ 100%
#
from time import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def main():
	data_set = load_iris()

	features, labels = split_features_labels(data_set)

	train_features, train_labels, test_features, test_labels = split_train_test(features, labels, 0.18)

	print(len(train_features), " ", len(test_features))

	clf = GaussianNB()

	print("Start training...")
	tStart = time()
	clf.fit(train_features, train_labels)
	print("Training time: ", round(time()-tStart, 3), "s")

	print("Accuracy: ", accuracy_score(clf.predict(test_features), test_labels))



def split_train_test(features, labels, test_size):
	total_test_size = int(len(features) * test_size)
	np.random.seed(2)
	indices = np.random.permutation(len(features))
	train_features = features[indices[:-total_test_size]]
	train_labels = labels[indices[:-total_test_size]]
	test_features  = features[indices[-total_test_size:]]
	test_labels  = labels[indices[-total_test_size:]]
	return train_features, train_labels, test_features, test_labels

def split_features_labels(data_set):
	features = data_set.data
	labels = data_set.target
	return features, labels


if __name__ == "__main__":
	main()