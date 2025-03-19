import pandas as pd
import numpy  as np
import math

def error(X, Y):
    out = 0
    for row in range(Y.shape[0]):
        for column in range(Y.shape[1]):
            if Y[row][column] <= 10:
                out += (X[row][column] - Y[row][column]) * (X[row][column] - Y[row][column])
    return math.sqrt(out / (Y.shape[0] * Y.shape[1]))

def unset_error(A, B, Y):
    out = 0
    entry_count = 0
    for row in range(Y.shape[0]):
        for column in range(Y.shape[1]):
            if Y[row][column] > 10:
                out += (A[row][column] - B[row][column]) * (A[row][column] - B[row][column])
                entry_count += 1
    return math.sqrt(out / entry_count)

def ignore_empty_lsrl (X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray: 
    # Because of the empty values, we have to reconstruct the matrices cell by cell, by turning a blindeye to the values 99. (or above 10)
    
    output = np.ndarray((X.shape[1],Y.shape[1]))

    for column in range(Y.shape[1]):
        valid_indexes = []
        # Solving one column of y / output at a time
        Y_v = Y[:, column]

        for row in range(len(Y_v)):
            if Y_v[row] <= 10:
                valid_indexes.append(row)
        X_cleaned = X[valid_indexes, :]
        Y_v_cleaned = Y_v[valid_indexes].reshape(-1,1)

        output[:, column] = lsrl(X_cleaned, Y_v_cleaned, lam)[:, 0]

    return output

def lsrl (X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray: 
    # We want some cooefficient matrix V such that X * V = Y, but the dimensions are too wrong to solve 
    # so we do Xt * X * V = Xt Y
    # V = (Xt*X)-1 Xt Y
    # and for some reason, it is better to do Xt X + 0.0001 I to avoid extreme values and not get boned by zero matricies 
    
    X_h = X.shape[0] 
    X_w = X.shape[1] 

    Y_h = Y.shape[0] 
    Y_w = Y.shape[1] 

    assert X_h == Y_h
    assert 0 <= lam <= 1

    # @ is the numpy op for matrx multiply, dumb
    XtX = X.T @ X + lam * np.eye(X_w)
     
    # print(XtX)
    XtY = X.T @ Y 

    out = np.linalg.solve(XtX, XtY)

    return out


if __name__ == '__main__':
    print('Running')
    xls_filename = 'jester-data-1.xls'
    xls_file = pd.ExcelFile(xls_filename)

    # ASSUMING ONE SHEET 
    xls_dataframe = xls_file.parse(xls_file.sheet_names[0])
    # FIRST COLUMN IS THE COUNT OF REVIEWS
    del xls_dataframe[xls_dataframe.columns[0]]

    UxJ = xls_dataframe.to_numpy()


    user_count = UxJ.shape[0]
    joke_count = UxJ.shape[1]

    print(user_count)
    print(joke_count)
    # User X Features * Features X Joke = User X Joke
    # UxF             * FxJ             = UxJ

    iterations = 10
    # feature_count = 70 

    for feature_count in [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
        UxF = np.random.rand(user_count, feature_count)
        FxJ = np.random.rand(feature_count, joke_count)
        # for l in [0.0001, 0.001, 0.01, 0.1, 1]:
        for l in [0.01]:
            for i in range(iterations):
                FxJ = ignore_empty_lsrl(UxF, UxJ, l)
                UxF = ignore_empty_lsrl(FxJ.T, UxJ.T, l).T
                # print(error(UxF @ FxJ, UxJ))
        
        pred1 = UxF @ FxJ


        UxF = np.random.rand(user_count, feature_count)
        FxJ = np.random.rand(feature_count, joke_count)
        # for l in [0.0001, 0.001, 0.01, 0.1, 1]:
        for l in [0.01]:
            for i in range(iterations):
                FxJ = ignore_empty_lsrl(UxF, UxJ, l)
                UxF = ignore_empty_lsrl(FxJ.T, UxJ.T, l).T
                # print(error(UxF @ FxJ, UxJ))
        
        pred2 = UxF @ FxJ
        print(f"Feature Count {feature_count}")
        print(f"Pred1 error {error(pred1, UxJ)}")
        print(f"Pred2 error {error(pred2, UxJ)}")
        print(f"Unset error {unset_error(pred1, pred2, UxJ)}")
        print("")
