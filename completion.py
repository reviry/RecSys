import numpy

def culculate_average(R):
    average_ui = []
    average_dj = []
    for i in xrange(R.shape[0]):
        count = 0
        sum_ui = 0
        for j in xrange(R.shape[1]):
            if R[i][j] != 0:
                count += 1
                sum_ui += R[i][j]
        average_ui.append(sum_ui / float(count))
    for j in xrange(R.shape[1]):
        count = 0
        sum_dj = 0
        for i in xrange(R.shape[0]):
            if R[i][j] != 0:
                count += 1
                sum_dj += R[i][j]
        average_dj.append(sum_dj / float(count))
    return average_ui, average_dj

def compliment(R, beta=0.5):
    average_ui, average_dj = culculate_average(R)
    for i in xrange(R.shape[0]):
        for j in xrange(R.shape[1]):
            if R[i][j] == 0:
                R[i][j] = beta * average_ui[i] + (1 - beta) * average_dj[j]
    return R
