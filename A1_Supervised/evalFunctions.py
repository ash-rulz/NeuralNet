import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    
    match = 0
    for pred in range(len(LPred)):
        if LPred[pred] == LTrue[pred]:
            match += 1
    acc = match/len(LPred)
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    
    num_class = np.unique(LTrue)
    num_class = np.sort(num_class)
    zipped = np.asarray(list((zip(LTrue, LPred))))
    cM = np.zeros((len(num_class),len(num_class)))
    row = 0

    for i in num_class:
        fltr_arr = zipped[np.in1d(zipped[:, 0], i)]
        pred = list(fltr_arr[:,1])
        col = 0
        for j in num_class:
            uni_count = pred.count(j)
            cM[row,col] = uni_count
            col += 1
        row += 1
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    acc_cM = sum(np.diag(cM))/np.sum(cM)
    # ============================================
    
    return acc_cM
