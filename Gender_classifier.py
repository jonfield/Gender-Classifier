from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
import numpy as np

# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_svm = svm.SVC()
clf_NeuralNet = MLPClassifier()
clf_knc = neighbors.KNeighborsClassifier()

# Train the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_NeuralNet.fit(X, Y)
clf_knc.fit(X, Y)

# Testing using the same data
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 1000
###print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 1000
print('Accuracy for SVM: {}'.format(acc_svm))

pred_Net = clf_NeuralNet.predict(X)
acc_Net = accuracy_score(Y, pred_Net) * 1000
print('Accuracy for NeuralNet: {}'.format(acc_Net))

pred_knc = clf_knc.predict(X)
acc_knc = accuracy_score(Y, pred_knc) * 1000
print('Accuracy for KNC: {}'.format(acc_knc))


# print prediction
if acc_svm > acc_Net and acc_svm > acc_knc:
	print("The best gender classifier is SVM : ",acc_svm)
elif acc_Net > acc_knc and acc_Net > acc_svm:
	print("The best gender classifier is NeuralNet : ",acc_Net)
else :
	print("The best gender classifier is KNeighborsClassifier : ",acc_knc)