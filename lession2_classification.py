# lession2_classification.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import graphviz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def prepare_data():
    input_data = pd.read_csv("./data/breast-cancer-wisconsin.csv",
        names=['id', 'clump_thickness', 'cell_size', 'cell_shape',
            'marginal_adhesion', 'single_cell_size', 'bare_nuclei',
            'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class'])
    input_data = input_data.apply(pd.to_numeric, errors='coerce')

    imputer = Imputer(strategy="mean")
    X = imputer.fit_transform(input_data)
    return pd.DataFrame(X, columns=input_data.columns)

def data_split(input_data):
    target = input_data['class'].copy()
    return input_data.drop(['class'], axis=1), target

def logistic_reg(train_data, train_value, test_data):
    logReg = LogisticRegression()
    logReg.fit(train_data, train_value)
    return logReg.predict(test_data)

def write_file(tree_clf):
    dot_data = export_graphviz(tree_clf, out_file=None,
        filled=True, proportion=True, rounded=True)
    graph = graphviz.Source(source=dot_data,
        filename="./decision_tree", format="png")
    graph.render()

def dicision_tree(train_data, train_value, test_data):
    tree_clf = DecisionTreeClassifier(max_depth=5)
    tree_clf.fit(train_data, train_value)
    write_file(tree_clf)
    return tree_clf.predict(test_data)

def measure_predict(y_true, y_predict):
    lb = LabelBinarizer()
    bin_true = pd.Series(lb.fit_transform(y_true)[:,0])
    bin_predict = pd.Series(lb.fit_transform(y_predict)[:,0])

    print("confusion matrix: \n{}\n".format(
        confusion_matrix(bin_true, bin_predict)))
    print("recall_score: {}\n".format(
        recall_score(bin_true, bin_predict)))
    print("precision_score: {}\n".format(
        precision_score(bin_true, bin_predict)))
    print("f1_score : {}\n".format(
        f1_score(bin_true, bin_predict)))

if __name__ == '__main__':
    input_data = prepare_data()

    train_set, test_set = train_test_split(input_data,
        test_size=0.1, random_state=59)
    train_data, train_value = data_split(train_set)
    test_data, test_value = data_split(test_set)

    predict_value1 = logistic_reg(train_data, train_value, test_data)
    #print("== logistic regression ==")
    #measure_predict(test_value, predict_value1)

    predict_value2 = dicision_tree(train_data, train_value, test_data)
    print("== decision tree ==")
    measure_predict(test_value, predict_value2)
