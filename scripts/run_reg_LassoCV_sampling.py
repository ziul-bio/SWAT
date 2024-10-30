#!/usr/bin/env python3 -u

###################### To run this script ##################################
#python scripts/run_reg_Lasso.py -i embeddings/sumo1_esm2_150M_compressed/ -m data/sumo1_human_metadata_v02.csv -o results/sumo1_esm2_150M_layer_30_compressed.csv


################ imports #####################
import os
import scipy
import torch
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

# to ignore the convergence warnings and Rho computeation warnings. 
# Use with caution. Only use when you are sure that the model is working fine.
import warnings
warnings.filterwarnings('ignore') 
from sklearn.exceptions import ConvergenceWarning


###################### Define Functions #######################

def features_scaler(features):
    '''Scale the features by min-max scaler, to ensure that the features selected by Lasso are not biased by the scale of the features'''
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    return pd.DataFrame(scaled_features)



def run_regression(features, target):
    '''this version computes y_pred for train and test sets'''
    # Initialize lists for storing results
    folds, num_nonzero_coefs = [], []
    r2s_train, maes_train, rmses_train = [], [], []
    r2s_test, maes_test, rmses_test = [], [], []
    rhos_train, rhos_test = [], []

    # Define the KFold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Loop over the KFold splits
    for kfold, (train_index, test_index) in enumerate(kf.split(features)):
        # Split the data into training and testing sets
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Define and train the regression model
        with warnings.catch_warnings():
            #warnings.simplefilter("ignore", category=ConvergenceWarning)
            model = LassoCV(cv=None, random_state=42, max_iter=1000, tol=1e-2, n_jobs=-1)
            model.fit(X_train, y_train)

            # get the number of non-zero coefficients
            coeficients = model.coef_
            num_nonzero_coef = np.sum(coeficients != 0)

            # Make predictions
            y_pred_train = pd.DataFrame(model.predict(X_train))
            y_pred_test = pd.DataFrame(model.predict(X_test))

            # Evaluate the model
            r2_train = metrics.r2_score(y_train, y_pred_train)
            mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
            mse_train = metrics.mean_squared_error(y_train, y_pred_train)
            rmse_train = np.sqrt(mse_train)
            rho_train, p_value_train = spearmanr(y_train, y_pred_train)

            r2_test = metrics.r2_score(y_test, y_pred_test)
            mae_test = metrics.mean_absolute_error(y_test, y_pred_test)
            mse_test = metrics.mean_squared_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)
            rho_test, p_value_test = spearmanr(y_test, y_pred_test)

            # Append results
            r2s_train.append(r2_train)
            maes_train.append(mae_train)
            rmses_train.append(rmse_train)

            r2s_test.append(r2_test)
            maes_test.append(mae_test)
            rmses_test.append(rmse_test)

            rhos_train.append(rho_train)
            rhos_test.append(rho_test)

            folds.append(kfold + 1)
            num_nonzero_coefs.append(num_nonzero_coef)

        # Return the collected results
        print(f"Results:  fold {kfold}, r2_train: {r2_train:.3f}, r2_test: {r2_test:.3f}, Num coefs: {num_nonzero_coef}")
    print(f"Results:  r2_train: {np.mean(r2s_train):.2f}, r2_test: {np.mean(r2s_test):.2f}, Num coefs: {np.mean(num_nonzero_coefs):.2f}")
    return r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds, num_nonzero_coefs


def save_results(r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds, num_nonzero_coefs):
    # Create dictionary for results
    res_dict = {
        "Model": ['Lasso'] * 5,
        "Fold": folds,
        "R2_score_train": r2s_train,
        "MAE_score_train": maes_train,
        "RMSE_score_train": rmses_train,
        "R2_score_test": r2s_test,
        "MAE_score_test": maes_test,
        "RMSE_score_test": rmses_test,
        "rho_score_train": rhos_train,
        "rho_score_test": rhos_test,
        "nun_zero_coefs": num_nonzero_coefs
    }

    # Convert results to DataFrame
    results = pd.DataFrame(res_dict).reset_index(drop=True)
    return results



def run_regression_with_sampling(input_file_path, path_meta_data):
    '''Run regression on compressed embeddings'''
    results = pd.DataFrame()
    print('Reading the embeddings...')
    embed_df = pd.read_pickle(input_file_path)
    
    sample_sizes = [32, 100, 320, 1000, 3200, 10000, 32000, 100000, 320000, 1000000]
    for idx, ss in enumerate(sample_sizes): 
        meta_data = pd.read_csv(path_meta_data)
        ss = min(ss, len(meta_data))
        if sample_sizes[idx] <= len(meta_data) or sample_sizes[idx-1] < len(meta_data) and sample_sizes[idx+1] > len(meta_data):
            print('\nResults for sample size:', ss)
            meta_data = meta_data.sample(n=ss, random_state=42)
    
            data = meta_data.merge(embed_df, how='inner', left_on='ID', right_on='ID')
            target = data['target']
            features = data.iloc[:, meta_data.shape[1]:]
            features = features_scaler(features)
            # run regression
            r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds, num_nonzero_coefs = run_regression(features, target)
            res = save_results(r2s_train, maes_train, rmses_train, r2s_test, maes_test, rmses_test, rhos_train, rhos_test, folds, num_nonzero_coefs)
            res['Sample_size'] = ss
            results = pd.concat([results, res], axis=0)

    return results


############################# Run Predictions #############################

def main():
    parser = argparse.ArgumentParser(description="Run regression for different target datasets and layers")
    parser.add_argument("-i", "--input", type=str, help="Path to the input file")
    parser.add_argument("-m", "--metadata", type=str, help="Target name in the metadata")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file")
    args = parser.parse_args()
    
    # Define the target name and output file
    input_file_path = args.input
    path_meta_data = args.metadata
    output = args.output
   

    results = run_regression_with_sampling(input_file_path, path_meta_data)
    results.to_csv(output)
    print(f'Process Finished!')

   
if __name__ == "__main__":
    main()