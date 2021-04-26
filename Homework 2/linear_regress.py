import numpy as np

def linear_regress(y, X):
    # finding the regression coefficients
    X_trp = np.transpose(X)
    b = np.linalg.inv(X_trp.dot(X)).dot(X_trp).dot(y)
    # finding the error term
    e = y - X.dot(b)
    e_trp = np.transpose(e)
    # finding sigma squared and variance
    sigma_sqr = (e_trp.dot(e)) / (X.shape[0] - X.shape[1] - 1)
    var_b = np.diag(np.multiply(sigma_sqr, np.linalg.inv(X_trp@X)))
    se = np.sqrt(var_b)
    # finding 95% Confidence Interval
    z_score = 1.96
    upper = b+se*z_score
    lower = b-se*z_score
    ci95 = [lower, upper]
    results = {"Regression Coefficients": b,
              "Standard Error (SE)": se,
              "95% Confidence Interval": ci95}
    print(results)
    return results

# NOTES:
# For additional resources on Linear Regression with Matrix Algebra:
# http://kokminglee.125mb.com/math/linearreg3.html
# https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf
# For additional resources on NumPy matrix operations:
# https://numpy.org/doc/stable/reference/generated/numpy.dot.html
# https://www.programiz.com/python-programming/matrix