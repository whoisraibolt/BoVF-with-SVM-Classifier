#!/usr/bin/env python
# coding: utf-8

# Imports
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from timer import Timer
import globals

# Call function train
def train(gama,
          descriptor_name,
          model_name):

    with Timer() as timer:
        # SVM Model
        SVM = svm.SVC(gama)
        
        SVM.fit(globals.train_feature_vec[0],
                globals.train_feature_vec[1])
        
        print('%s %s\n' % (descriptor_name, model_name), file = globals.file)

        print('Training Set Evaluation\n', file = globals.file)
        
        print('Train score: %.2f\n' % (SVM.score(globals.train_feature_vec[0],
                                                 globals.train_feature_vec[1])),
                                                 file = globals.file)

        return SVM

    print('Time:', timer, '\n', file = globals.file)

# Call function test
def test(model,
         descriptor_name,
         model_name):

    with Timer() as timer:
        # SVM Model
        print('%s %s\n' % (descriptor_name, model_name), file = globals.file)

        print('Testing Set Evaluation\n', file = globals.file)

        SVM_predict = model.predict(globals.test_feature_vec[0])

        print('Test score: %.2f\n' % (accuracy_score(globals.test_feature_vec[1],
                                                     SVM_predict)),
                                                     file = globals.file)

        return SVM_predict

    print('Time:', timer, '\n', file = globals.file)

# Call function classificationReport
def classificationReport(model,
                         predict,
                         descriptor_name,
                         model_name):
    #  SVM Model 
    print('%s %s\n' % (descriptor_name, model_name), file = globals.file)

    # Classification report for Classifier
    print("Classification report for Classifier: \n\n%s: \n\n %s" % (model,
                                                                     classification_report(globals.test_feature_vec[1],
                                                                                           predict)),
                                                                                           file = globals.file)