import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold


def Match_features_with_BCR(feature: pd.DataFrame, BCR: pd.DataFrame, 
                 BCR_column_name="BCR_Binary Outcome", 
                 sample_column_name = "sample_name"):

    """
    This function:
        - Clean sample name to AFM/EAMxxx format
        - Add columns "Days to event", "BCR_binary" to feature df
    Input:
        - BCR_column_name (str): "5yBCR" or "BCR_Binary Outcome" 
        - sample_column_name (str): "sample_name" or "other_patient_id"

    """
    
    # Clean and standardize identifiers
    BCR["Record ID"] = BCR["Record ID"].str.replace("-", "", regex=False)
    feature[sample_column_name] = feature[sample_column_name].str.upper()
    feature[sample_column_name] = feature[sample_column_name].str.extract(r'((?:AFM|EAM)\d{3})')

    # Map BCR outcome and event time
    feature["BCR_binary"] = feature[sample_column_name].map(BCR.set_index("Record ID")[BCR_column_name])
    feature["Days to event"] = feature[sample_column_name].map(BCR.set_index("Record ID")["Days to event"])
    
    # Filter out unknown or missing values
    feature_BCR = feature[feature["BCR_binary"] != "Unknown"].copy()
    feature_BCR = feature_BCR.dropna(subset=["BCR_binary"])
    
    return feature_BCR


def PlotROC_with_AUC(roc_curves, model_name, color='blue'):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for fpr, tpr in roc_curves:
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # Ensure the TPR starts at 0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the TPR ends at 1
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([auc(fpr, tpr) for fpr, tpr in roc_curves])
    print(f'{model_name} (Mean AUC = {mean_auc:.2f} ± {std_auc:.2f})')

    # Plot the mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color=color, lw=2, 
             label=f'{model_name} (Mean AUC = {mean_auc:.2f} ± {std_auc:.2f})')
    plt.grid(False)  # Ensure no grid is displayed
