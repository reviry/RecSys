import math

def evaluate(R_test, R_pred):
    """
    for all (i, j) pairs
    tp is the number of R_test[i][j] >= 4 and R_pred[i][j] >= 4
    fp is the number of R_test[i][j] < 4 and R_pred[i][j] >= 4
    tn is the number of R_test[i][j] < 4 and R_pred[i][j] < 4
    fn is the number of R_test[i][j] >= 4 and R_pred[i][j] < 4
    """
    tp = fp = tn = fn = 0
    count = 0
    squared_error = 0
    for i in xrange(R_test.shape[0]):
        for j in xrange(R_test.shape[1]):
            if R_test[i][j] != 0:
                count += 1
                squared_error += pow(R_test[i][j] - R_pred[i][j], 2)
                if R_test[i][j] >= 4:
                    if R_pred[i][j] >= 4:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if R_pred[i][j] >= 4:
                        fp += 1
                    else:
                        tn += 1
    p = precision(tp, fp)
    r = recall(tp, fn)
    f = fvalue(p, r)
    e = rmse(squared_error, count)
    return p, r, f, e

def precision(tp, fp):
    if tp + fp == 0:
        return 0
    else:
        return tp / float(tp + fp)

def recall(tp, fn):
    if tp + fn == 0:
        return 0
    else:
        return tp / float(tp + fn)

def fvalue(p, r):
    if p + r == 0:
        return 0
    else:
        return 2 * p * r / float(p + r)

def rmse(se, count):
    return math.sqrt(se / float(count))
