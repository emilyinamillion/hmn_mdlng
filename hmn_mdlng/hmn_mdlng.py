"""
    package.module
    ~~~~~~~~~~~~~

    Module for human analysis of supervised learning natural language processing model results from sklearn.
    A variety of models and dimensionality reduction algorithms are available. 
    I wrote this module as a way to more clearly understand Precision and Recall in my results, to think critically
    about why my algorithms would be misclassifying as they were. I did not find that the sklearn module does a granular
    job of making this clear (precision and recall are available, but unlabeled as far as classes go, and not generalizable
    for a reusable module as you need to select the number of classes.)
    
    
    
    Defaults:
        Dimensionality Reduction - None
        Holdout data (validation set) - True
        Logistic Regression
        Count Vectorizer

    :author: emilyinamillion
    :license: MIT License
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression, Perceptron
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB

import re
import numpy as np


class NLPModeler(object):
    
    def __init__(self, data, X_column_label, y_column_label, vect = CountVectorizer(), mod = LogisticRegression(), sel = None, holdout = True):
        self.data = data
        self.vect = vect
        self.mod = mod
        self.sel = sel
        self.holdout = holdout
        if self.holdout == True:
            self.data_splitter()
            
        self.X, self.X_label = self.data[X], X
        self.y, self.y_label = self.data[y], y 
        self.features = vect.fit_transform(self.X) 
        self.features_reduced = self.dim_reducer()
        self.X_train_transformed, self.X_test_transformed, \
            self.y_train, self.y_test = train_test_split(self.features_reduced, self.y)
            
        self.test_df = pd.DataFrame(self.y_test)
        self.class_value_dict = self.make_value_counts(self.y_test)
        self.modeled = self.model_trainer()
        if self.holdout == True:
            self.validator()
        
    def data_splitter(self):
        self.validation = self.data.sample(frac = 0.1)
        self.data = self.data.loc[self.data.index.difference(self.validation.index)]
        
    def dim_reducer(self):
        if self.sel is not None:
            return self.sel.fit_transform(self.features)
        else:
            print("No dimensionality reducer allocated - splitting full dataframe")
            return self.features 
    
    def predictor(self, to_predict):
        return self.clf.predict(to_predict)
    
    def validator(self):
        print("\n\nValidation Dataset shape: ", self.validation.shape)
        self.class_value_dict = self.make_value_counts(self.validation[self.y_label])
        self.validation["predicted_class"] = pd.Series(self.modeled.predict(self.validation[self.X_label]), \
                                                    index = self.validation.index)
        self.metrics_(self.validation[self.y_label], self.validation.predicted_class)
        self.accuracy_breakdown(self.validation)
    
    ## split out testing to another function - 6/28/17
    def model_trainer(self):
        self.get_feature_stats()
        
        print("Training Model, does your data have what it takes? ⌐(ಠ۾ಠ)¬")
        self.clf = self.mod.fit(self.X_train_transformed, self.y_train)
        
        print("Model trained successfully, making predictions on test set ٩(- ̮̮̃-̃)۶ ......\n")
        self.y_test_prediction = self.predictor(self.X_test_transformed)
        self.test_df["predicted_class"] = pd.Series(self.y_test_prediction, \
                                                    index = self.test_df.index)
        
        self.metrics_(self.y_test, self.y_test_prediction)
        self.accuracy_breakdown(self.test_df)
        
        pipeline = Pipeline([
                ('vect', self.vect),
                ('sel', self.sel),
                ('clf', self.clf),
            ])
        return pipeline
    
    def get_feature_stats(self):
        
        print("Reduced dimensions (full dataframe) shape: ", self.features.shape)
        print("Train dataframe shape: ", self.X_train_transformed.shape)
        print("Test dataframe shape: ", self.X_test_transformed.shape)
        print("...........................\n")
        print("Train value distribution: \n", self.y_train.value_counts(), "\n")
        print("Test value dist: \n", self.y_test.value_counts(), "\n")
        print("############################################\n")
    
    def make_value_counts(self, to_count):
        value_cnts = to_count.value_counts().reset_index()
        value_cnts.columns = ["class_label", "value"]
        return pd.Series(value_cnts.value.values,index= value_cnts.class_label).to_dict()
    
    def accuracy_breakdown(self, data_to_analyze):
        
        def percent_total_accuracy(row):
            if row["true_class"] == row["predicted_class"]:
                return row["value"] / self.class_value_dict.get(row["true_class"])
            return -(row["value"] / self.class_value_dict.get(row["true_class"]))

        scores_df = data_to_analyze.groupby([self.y_label, "predicted_class"]).size().reset_index()
        scores_df.columns = ["true_class", "predicted_class", "value"]

        scores_df["%"] = scores_df.apply(percent_total_accuracy, axis=1)

        true_scores = scores_df[scores_df["%"] > 0]
        print("\nTrue Assignments\n", true_scores.sort_values("%", ascending = False).head(10), "\n")

        false_scores = scores_df[scores_df["%"] < 0 ]
        false_scores["%"] = false_scores["%"].apply(lambda x: -(x))
        print("False Assignments\n", false_scores.sort_values("%", ascending = False).head(10))
        
    def metrics_(self, predicted_values, actual_values):
        print("------- Performance Metric Summary --------")
        print("Confusion Matrix: \n", confusion_matrix(predicted_values, actual_values))
        print("Total Accuracy Score: ", accuracy_score(predicted_values, actual_values))     
    
    def sound(self):
        import os
        os.system("printf '\a'")
        os.system("printf '\a'")
        os.system("printf '\a'")
    
    def main(self):
        self.sound()
        return self.modeled
