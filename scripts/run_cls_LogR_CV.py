#!/usr/bin/env python3 -u

# To run this script:
# python scripts/run_cls_LogR_CV.py -i embeddings/embeddings_uniprot_EC/uniprot_EC_esm2_150M_compressed -m data/metadata_uniprot_EC/metadata_uniprot_EC_level1.csv -o results/test.csv -nc 5

################ imports #####################
import os
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


import warnings
from sklearn.exceptions import UndefinedMetricWarning


# Suppress only UndefinedMetricWarnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



###################### Define Functions #######################

def features_scaler(features):
    '''Scale the features by min-max scaler, to ensure that the features selected by Lasso are not biased by the scale of the features'''
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    return pd.DataFrame(scaled_features)


def run_Log_regression_CV(features, target, num_classes):
    # Initialize lists for storing results
    folds = []
    accuracies_train, recalls_train, precisions_train, f1_scores_train = [], [], [], []
    accuracies_test, recalls_test, precisions_test, f1_scores_test = [], [], [], []

    
    coefs = {f'class_{i}_non_zero_coef': [] for i in range(0,num_classes)}
    class_accuracies = {f'class_{i}_accuracy': [] for i in range(0,num_classes)}
    
    # Define the KFold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Loop over the KFold splits
    for kfold, (train_index, test_index) in enumerate(kf.split(features)):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model = LogisticRegressionCV(solver='saga', penalty='l1', cv=5, max_iter=1000, tol=1e-2)
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


        print(f"Results:  Kfold {kfold+1}, ACC_train: {accuracy_train:.3f}, ACC_test: {accuracy_test:.3f}")
        folds.append(kfold + 1)
    # Return the collected results
    return accuracies_train, recalls_train, precisions_train, f1_scores_train, accuracies_test, recalls_test, precisions_test, f1_scores_test, folds, coefs, class_accuracies


def save_results(accuracies_train, recalls_train, precisions_train, f1_scores_train, accuracies_test, recalls_test, precisions_test, f1_scores_test, folds, coefs, class_accuracies):
    # Create dictionary for results
    res_dict = {
        "Model": ['LogR'] * 5,
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



def run_LogR_CV_on_compressed_files(path_compressed_embeds, path_meta_data, num_classes):

    results = pd.DataFrame()
    meta_data = pd.read_csv(path_meta_data)

    for file in os.listdir(path_compressed_embeds)[:2]:
        if file.endswith('.pkl'):
            method = file.split('_')[-1].split('.')[0]
            file_path = os.path.join(path_compressed_embeds, file)
            embed = pd.read_pickle(file_path)
            embed_df = pd.DataFrame.from_dict(embed, orient='index').reset_index()
            embed_df.rename(columns={'index': 'ID'}, inplace=True)

            data = meta_data.merge(embed_df, how='inner', left_on='ID', right_on='ID')
            target = data['target']
            features = data.iloc[:, meta_data.shape[1]:]
            features = features_scaler(features)
        
            # run regression
            print(f'Running regression for {method}')   
            accuracies_train, recalls_train, precisions_train, f1_scores_train, accuracies_test, recalls_test, precisions_test, f1_scores_test, folds, coefs, class_accuracies= run_Log_regression_CV(features, target, num_classes)

            # print results and save them in a DataFrame
            print(f'Mean Accuracy train: {np.mean(accuracies_train):.3f}, Mean Accuracy Test: {np.mean(accuracies_test):.3f}')

            res = save_results(accuracies_train, recalls_train, precisions_train, f1_scores_train, accuracies_test, recalls_test, precisions_test, f1_scores_test, folds, coefs, class_accuracies)
            results = pd.concat([results, res]).reset_index(drop=True)

    return results


############################# Run Predictions #############################

def main():
    parser = argparse.ArgumentParser(description="Run Log regression")
    parser.add_argument("-i", "--input", type=str, help="Path to the output file")
    parser.add_argument("-m", "--metadata", type=str, help="Target name in the metadata")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file")
    parser.add_argument("-nc", "--numberClasses", type=int, help="Number of classes in the target dataset")
    args = parser.parse_args()
    
    # Define the target name and output file
    path_compressed_embed = args.input
    path_meta_data = args.metadata
    num_classes = args.numberClasses
    output_path = args.output

    
    # Load the necessary data and run the regression
    results_df = run_LogR_CV_on_compressed_files(path_compressed_embed, path_meta_data, num_classes)
    results_df.to_csv(output_path)
    
    print(f'Process Finished!')

   
if __name__ == "__main__":
    main()