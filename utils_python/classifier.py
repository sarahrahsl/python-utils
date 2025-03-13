import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold


def PlotROC_with_AUC(roc_curves, model_name):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for fpr, tpr in roc_curves:
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # Ensure the TPR starts at 0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the TPR ends at 1
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([auc(fpr, tpr) for fpr, tpr in roc_curves])

    # Plot the mean ROC curve
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f'{model_name} (Mean AUC = {mean_auc:.2f} ± {std_auc:.2f})')


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
