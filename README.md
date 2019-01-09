In this coding challenge, the dataset recorded various features of potential candidates for the position as Chief TroubleMaker.

Preprocessing the Data:

    1. With the way the data was organized the labels and unique ID were the first two columns. Therefore during training the svm all columns but the first two were
    treated as the feature set. Consequently the labels were split into their own array organized to match the index of their features in the feature set.

    2. During the training, the labeled data entries were split into a training (75%) and testing set(25%).

    3. Since the labeled data did not have an even representation for each class (about 50% more "0" than "1"). I balanced the class weights in the model such that the
    "1" labeled entries were weighted proportional to its frequency in samples. Hence "SVC(class_weight='balanced')"

    Additional work:
        From the results, we can see there can be further actions taken from more accuracy. One thing I would like to have tried is creating a subset of the data such that there is
        and equal representation of both classes. Another thing I would have liked to run a feature selection method to determine the importance of features. From a high level, I have a
        feeling some of these features had very little importance to the decision making process.

The Model:
    For this project I decided to use an SVM to classify the data. I liked the idea of using an SVM for a two-class classification problem with numerical encoded features. I find that RFC
    are more appropriate with multi-class, and different encoded features (categorical, numerical, binary). An SVM is not recommended to work with datasets of 10^5 of more entries. This dataset
    was well under that so performance was not a worry.

    When first running the model without balancing the class weights, all test entries were classified as "0". This made me question the overlap, non-linear separation between the two classes. Due to that, I went forward with using an Radial Basis Function (RBF) as my kernel method. In general, RBF is commonly used with SVM's and non-linear data due to its adaptiveness to noise. Resulting in softening the margin accordingly.

    Additional Work:

        1. I would like to tuning the class weights to find the best results. A form of back propagation where an error rate based on accuracy of results determines the tuning of the class weights. Other parameters worth trying to tune is the gamma parameter which represents the kernel coefficient. The gamma parameter defaulted to 1/num_features.  

        2. I would have liked to try out the RFC or any other decision tree if I cared more for the rules for classifying the customers. It could have also been valuable to better understand feature importances. Overall, RFC was not as fitting for a two-class classification problem.

Results:

The results of the confusion matrix show that there is potential for more tweaks in tuning hyper-parameters and/or preprocessing the data. One take away from these results is that with less exmaples of "1" class we can see it effecting the precision in classifying the unlabeled data. There

Confusion Matrix:

0: [190 336]
1: [55  224]
    0    1


Report:
                precision   recall  f1-score    support

0.0             0.78        0.36    0.49        526
1.0             0.40        0.80    0.53        279

micro avg       0.51        0.51    0.51        805
macro avg       0.59        0.58    0.51        805
weighted avg    0.65        0.51    0.51        805
