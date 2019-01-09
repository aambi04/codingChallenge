import csv
import numpy as np
import pandas as pd

NUM_ENTRIES = 35644
NUM_TRAIN_ENTRIES = 3218
NUM_TEST_ENTRIES = NUM_ENTRIES - NUM_TRAIN_ENTRIES
NUM_FEATURES_AND_LABEL = 20


class csvUtil:

    def __init__(self, fileIn, fileOut):
        self.fileIn = fileIn
        self.fileOut = fileOut
        self.label_set = np.zeros((NUM_TRAIN_ENTRIES, NUM_FEATURES_AND_LABEL))
        self.unlabel_set = np.zeros((NUM_TEST_ENTRIES, NUM_FEATURES_AND_LABEL))

    # Helper function to convert each row in the dataset into an array of floats
    # Customized to handle the empty values (label column) and sets to -1 as indication for test entry
    # @param row    row read in from csv
    # @return temp  an array of type float
    def convertToFloatArray (self,row) :
        temp = np.zeros(len(row))
        ind = 0
        for i in row:
            if i != '':
                temp[ind] = float(i)
            else:
                temp[ind] = -1.0
            ind += 1
        return temp

    # Reads the data file and stores into two datasets {Label & Unlabeled}
    def readAndStore(self):

        train_index = 0
        test_index = 0

        with open(self.fileIn) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                # Ignore header line
                if line_count > 0:
                    temp = self.convertToFloatArray(row)
                    #training set has labels {0,1}
                    if (temp[0] != -1):
                        self.label_set[train_index] = temp
                        train_index +=  1
                    #test set has label {-1}
                    else:
                        self.unlabel_set[test_index] = temp
                        test_index += 1
                line_count += 1

    # getter of Labeled dataset
    def getLabelSet(self):
        return self.label_set

    #getter of unlabeled dataset
    def getUnLabelSet(self):
        return self.unlabel_set

    # Writes solution to CSV
    def writeCSV(self, label, set):

        df = pd.DataFrame(list(zip(*[label, set[:,1]])), columns=['Class Label', 'ID'])

        df.to_csv(self.fileOut, index=True)
