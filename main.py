import numpy as np
from csvUtil import *

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Trains the SVM model with the labeled datasets
# @param labeled_set    the data entries with labeled classes
# @return clf           the trained svm model

def trainAndTestSVM (labeled_set) :
    train_feature_set = np.delete(labeled_set, [0,1],1)       # remove the first two columns in the training set (label and ID (unique))
    train_label_set = labeled_set[:,0]                         # first column is the labels for the training set

    feat_train, feat_test, lab_train, lab_test = train_test_split(train_feature_set, train_label_set, random_state=0)               # split labeled set into training and testing data sets (train:75%, test:25%)

    clf = SVC(kernel='linear', gamma='scale',class_weight='balanced')
    clf.fit(feat_train, lab_train)
    test_label_pred = clf.predict(feat_test)

    conf_matrix = confusion_matrix(lab_test, test_label_pred)
    print(conf_matrix)
    print(classification_report(lab_test, test_label_pred))

    return clf

# Predicts the classes for unlabeled data entries
# @param svm_model      the trained svm model
# @param unlabel_set    the unlabeled data entries
# @return predicted classes for the entries
def labelEntries(svm_model, unlabel_set):
    feature_set = np.delete(unlabel_set, [0,1],1)
    predict_labels = svm_model.predict(feature_set)

    return predict_labels


def main():

    csv = csvUtil('MLChallengeData.csv', 'output.csv')
    csv.readAndStore()
    label_set = csv.getLabelSet()
    unlabel_set = csv.getUnLabelSet()

    trained_svm_model = trainAndTestSVM(label_set)

    labels = labelEntries(trained_svm_model, unlabel_set)

    csv.writeCSV(labels, unlabel_set)

if __name__ == "__main__":
    main()
