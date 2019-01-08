import csv
import numpy as np
import copy

NUM_ENTRIES = 35644
NUM_TRAIN_ENTRIES = 3218
NUM_TEST_ENTRIES = NUM_ENTRIES - NUM_TRAIN_ENTRIES
NUM_FEATURES_AND_LABEL = 20

# Helper function to convert each row in the dataset into an array of floats
# Customized to handle the empty values (label column) and sets to -1 as indication for test entry
# @param row    row read in from csv
# @return temp  an array of type float
def convertToFloatArray (row) :
    temp = np.zeros(len(row))
    ind = 0
    for i in row:
        if i != '':
            temp[ind] = float(i)
        else:
            temp[ind] = -1.0
        ind += 1
    return temp


def readCsvAndStore():

    train_set = np.zeros((NUM_TRAIN_ENTRIES, NUM_FEATURES_AND_LABEL))
    test_set = np.zeros((NUM_TEST_ENTRIES, NUM_FEATURES_AND_LABEL))
    train_index = 0
    test_index = 0

    with open('MLChallengeData.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            # Ignore header line
            if line_count > 0:
                temp = convertToFloatArray(row)
                #training set has labels {0,1}
                if (temp[0] != -1):
                    train_set[train_index] = temp
                    train_index +=  1
                #test set has label {-1}
                else:
                    test_set[test_index] = temp
                    test_index += 1
            line_count += 1
    return [train_set, test_set]

table = readCsvAndStore()
