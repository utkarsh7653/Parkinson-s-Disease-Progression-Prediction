## Parkinson's Disease Prediction Project

## Overview:

This project focuses on predicting the Movement Disorder Society-Unified Parkinson's Disease Rating Scale (MDS-UPDRS) scores to measure Parkinson's disease progression using protein and peptide data. The primary task is regression to predict continuous UPDRS scores (1-4), with a secondary task of classification to predict binned UPDRS categories (Mild, Moderate, Severe). The dataset is sourced from the Kaggle competition: AMP®-Parkinson’s Disease Progression Prediction.

## Project Structure:

The project includes a Jupyter Notebook (Neural Network Code (1).ipynb) containing the implementation of various machine learning models and a PDF report (Parkinson’s Disease Prediction (MSE 546 Final Project) (5).pdf) summarizing the methodology, results, and insights.

## Files:

Neural Network Code (1).ipynb: Jupyter Notebook with code for data preprocessing, model training (Linear Regression, Random Forest, KNN, Neural Network, and Classification), and performance visualization.
Parkinson’s Disease Prediction (MSE 546 Final Project) (5).pdf: Comprehensive report detailing the project overview, data, methodology, evaluation metrics, results, and key takeaways.

## Dataset:
Source: Kaggle dataset from the AMP®-Parkinson’s Disease Progression Prediction competition.
Description: Merged dataset from clinical, supplemental clinical, peptide, and protein data, containing 4,838 entries and 10 columns.



## Features:
Categorical: upd23b_clinical_state_on_medication (encoded as numerical values).
Continuous: PeptideAbundance, NPX, visit_month.



## Preprocessing:
Duplicates removed.
Missing values filled with 0 or handled appropriately.
Categorical variables encoded numerically.
Features standardized for certain models (e.g., KNN, Neural Network).


## Models and Methodology:

The project implements and evaluates multiple machine learning models to predict UPDRS scores (1-4) and categorize them into severity levels:

Regression Models

Linear Regression (Baseline):
Basic multi-output regression.
Metrics: MAE: 5.16, MSE: 61.46, RMSE: 7.84, R²: 0.056.



Feature-Engineered Linear Regression:
Added interaction terms, polynomial features, and time-based features.
Metrics: MAE: 5.0, MSE: 58.1, RMSE: 6.8, R²: 0.1 (5-fold CV).



Random Forest Regressor:
Features: visit_month, PeptideAbundance, upd23b_clinical_state_on_medication, NPX.


Metrics: MAE: 4.7, MSE: 53.54, RMSE: 6.2, R²: 0.13.
K-Nearest Neighbors (KNN) Regressor:
Optimized with grid search (best k=20).



Metrics: MAE: 4.73, MSE: 52.78, RMSE: 7.27, R²: 0.13, SMAPE: 89.06%.



Neural Network:
Feed-forward network with batch normalization (128 -> 64 -> 32 neurons, ReLU, 20% dropout).
Metrics: MAE: 4.64, MSE: 52.05, RMSE: 7.21, R²: 0.166.

Classification Model:

Random Forest Classifier:
Predicts binned UPDRS scores (Mild, Moderate, Severe).
Evaluated using precision, recall, and F1-score.



Performance varies across UPDRS scores, with challenges in imbalanced classes.

## Evaluation Metrics

Regression:

Mean Absolute Error (MAE): Measures average prediction error.
Mean Squared Error (MSE) / Root Mean Squared Error (RMSE): Penalizes larger errors.
R²: Indicates variance explained by the model.
Symmetric Mean Absolute Percentage Error (SMAPE): Normalizes errors relative to actual values.

Classification:
Precision, Recall, F1-score: Evaluates performance across severity levels.

## Results:
Best Model: Neural Network with Batch Normalization (highest R²: 0.166, MAE: 4.64).



## Key Improvements:

Reduced MAE from 5.16 (baseline) to 4.64 (Neural Network).
Reduced RMSE from 7.84 (baseline) to 6.2 (Random Forest).
Improved R² from 0.056 (baseline) to 0.166 (Neural Network).
Complementary Classifier: Random Forest Classifier provides quick severity categorization for clinical use.


## Key Takeaways:

The Neural Network outperforms other models with the highest R² (0.166) and lowest MAE (4.64).
The upd23b_clinical_state_on_medication feature is critical for predicting Parkinson’s progression.
Enhanced models significantly reduce errors compared to the baseline.

## Key Feature:
upd23b_clinical_state_on_medication is the most important predictor across models.

Challenges include high SMAPE (98.3% for Neural Network) and class imbalance in classification, suggesting future improvements like SMOTE for handling imbalanced data.


## Future Work:
Address high SMAPE through advanced feature engineering or alternative loss functions.
Apply techniques like SMOTE to handle class imbalance in classification.
Explore additional models (e.g., Gradient Boosting, LSTM) for improved performance.

## References:
Kaggle: AMP®-Parkinson’s Disease Progression Prediction
Murphy, Kevin P. Probabilistic Machine Learning: An Introduction. The MIT Press, 2022.




