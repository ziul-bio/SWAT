#!/usr/bin/env python3 -u

# To run this script:
# python scripts/run_cls_all_layers_LogR.py -t target -M esm2_150M -o results/classification/pisces_esm2_15B_All_Layers_target.csv

################ imports #####################
import os
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
#from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


import warnings
from sklearn.exceptions import UndefinedMetricWarning

import sys
sys.path.append('scripts/')
from utils import load_meta_data, features_scaler, load_embed_merged


# Suppress only UndefinedMetricWarnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



###################### Define Functions #######################

def run_Log_regression(features, target, target_name, num_classes):
    # Initialize lists for storing results
    folds = []
    accuracies_train, recalls_train, precisions_train, f1_scores_train = [], [], [], []
    accuracies_test, recalls_test, precisions_test, f1_scores_test = [], [], [], []
    
    # Initialize a dictionary where each key will correspond to a class list
    ## for index stating from 0
    coefs = {f'class_{i}_non_zero_coef': [] for i in range(0,num_classes)}
    class_accuracies = {f'class_{i}_accuracy': [] for i in range(0,num_classes)}


    # Define the KFold cross-validator
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Loop over the KFold splits
    for kfold, (train_index, test_index) in enumerate(kf.split(features)):
        # Split the data into training and testing sets
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Define and train the regression model
        model = LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear')
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = pd.DataFrame(model.predict(X_train))
        y_pred_test = pd.DataFrame(model.predict(X_test))

        # get the number of non-zero coefficients
        coefficients = model.coef_

        for i in model.classes_.astype(int):
            coefs[f'class_{i}_non_zero_coef'].append(np.sum(coefficients[i] != 0))

        # Evaluate the model on overall metrics
        accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
        recall_train = metrics.recall_score(y_train, y_pred_train, average='macro')
        precision_train = metrics.precision_score(y_train, y_pred_train, average='macro')
        f1_score_train = metrics.f1_score(y_train, y_pred_train, average='macro')

        accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
        recall_test = metrics.recall_score(y_test, y_pred_test, average='macro')
        precision_test = metrics.precision_score(y_test, y_pred_test, average='macro')
        f1_score_test = metrics.f1_score(y_test, y_pred_test, average='macro')

        # Evaluate the model on class-specific accuracy
        cm = confusion_matrix(y_test, y_pred_test)
        # normalize the confusion matrix
        # here each value in the cm will be divided by the sum of the values in the same row
        # now each value in the cm will represent the percentage of the true class that was predicted as the class in the column
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        class_acc = cm.diagonal()  # the diagonal elements are the accuracies of each class
        
        # for index stating from 0
        for i in range(0, len(class_acc)):
            class_accuracies[f'class_{i}_accuracy'].append(class_acc[i])


        # Append results
        accuracies_train.append(accuracy_train)
        recalls_train.append(recall_train)
        precisions_train.append(precision_train)
        f1_scores_train.append(f1_score_train)

        accuracies_test.append(accuracy_test)
        recalls_test.append(recall_test)
        precisions_test.append(precision_test)
        f1_scores_test.append(f1_score_test)


        folds.append(kfold + 1)
    # Return the collected results
    return accuracies_train, recalls_train, precisions_train, f1_scores_train, accuracies_test, recalls_test, precisions_test, f1_scores_test, folds, coefs, class_accuracies


def save_results(layer, accuracies_train, recalls_train, precisions_train, f1_scores_train, accuracies_test, recalls_test, precisions_test, f1_scores_test, folds, coefs, class_accuracies):
    # Create dictionary for results
    res_dict = {
        "Layer": [layer] * 10,
        "Model": ['LogR'] * 10,
        "Fold": folds,
        "Accuracy_train": accuracies_train,
        "Recall_train": recalls_train,
        "Precision_train": precisions_train,
        "F1_score_train": f1_scores_train,
        "Accuracy_test": accuracies_test,
        "Recall_test": recalls_test,
        "Precision_test": precisions_test,
        "F1_score_test": f1_scores_test
    }
    # update the dictionary with the number of non-zero coefficients for each class
    res_dict.update(coefs)
    res_dict.update(class_accuracies)

    # Convert results to DataFrame
    results = pd.DataFrame(res_dict).reset_index(drop=True)
    return results



def run_Log_regression_all_layers(target_name, model, num_classes):
    '''Run the regression for all layers, for a given target'''
    if model == 'esm2_15B':
        rep_layer = 48
    elif model == 'esm2_650M':
        rep_layer = 33
    elif model == 'esm2_150M' or model == 'esm2_150M_tuned':
        rep_layer = 30
    
  
    results_df = pd.DataFrame()
    meta_data =  load_meta_data(target_name)

    for layer in range(0, rep_layer+1):
        
        embeds = load_embed_merged(target_name, model, layer)
        data = meta_data.merge(embeds, how='inner', left_on='ID', right_on='ID')
        
        # Define target and features
        target = data[target_name]
        
        # define features
        features = data.iloc[:, meta_data.shape[1]:]
        features = features_scaler(features)
        
        # run regression
        accuracies_train, recalls_train, precisions_train, f1_scores_train, accuracies_test, recalls_test, precisions_test, f1_scores_test, folds, coefs, class_accuracies= run_Log_regression(features, target, target_name, num_classes)
        
        # print results and save them in a DataFrame
        print(f'Results regression for {target_name} on layer {layer}, Mean Accuracy: {np.mean(accuracies_test):.3f}')
        
        res = save_results(layer, accuracies_train, recalls_train, precisions_train, f1_scores_train, accuracies_test, recalls_test, precisions_test, f1_scores_test, folds, coefs, class_accuracies)
        results_df = pd.concat([results_df, res]).reset_index(drop=True)

    return results_df


############################# Run Predictions #############################

def main():
    parser = argparse.ArgumentParser(description="Run regression for different target datasets and layers")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file")
    parser.add_argument("-t", "--target", type=str, help="Target name in the metadata")
    parser.add_argument("-M", "--model", type=str, help="Model name (esm2_15B, esm2_650M, esm2_150M)")
    parser.add_argument("-nc", "--numberClasses", type=int, help="Number of classes in the target dataset")
    args = parser.parse_args()
    
    # Define the target name and output file
    target_name = args.target
    output_dir = args.output
    model = args.model
    num_classes = args.numberClasses
    
    # Load the necessary data and run the regression
    print(f'Running regression for {target_name}!')
    results_df = run_Log_regression_all_layers(target_name, model, num_classes)
    results_df.to_csv(output_dir)
    
    print(f'Process Finished for {target_name}')

   
if __name__ == "__main__":
    main()