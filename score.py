#!/usr/bin/env python
#
# file: DPATH Scoring
#
#------------------------------------------------------------------------------

# import system modules
#
import os
import sys
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

# compute the confusion matrix
#
def compute_conf(r, h):

    # check the length
    #
    if len(r) != len(h):
        print("**> Error: lists are not the same length (%d)(%d)" %
              (len(r), len(h)))

    # map the hypotheses
    #
    cnf = np.zeros((9,9), dtype = int)
    for (v1, v2) in zip(r,h):
        if (v1 == 7) or (v1 == 4) or (v1 == 1):
            pass
        else:
            cnf[v1][v2] += int(1)
    return cnf

# function: main
#
def main(argv):

    # load the data into a list
    #
    ref = []
    flag = int(0)
    for l in (open(sys.argv[1], 'r')).readlines():
        if flag > int(0):
            t = l.rstrip('\n').split(",")
            ref.append(int(t[0]))
        else:
            flag += int(1)

    hyp = []
    flag = int(0)
    for l in (open(sys.argv[2], 'r')).readlines():
        if flag > int(0):
            t = l.rstrip('\n').split(",")
            hyp.append(int(t[0]))
        else:
            flag += int(1)

    # compute the confusion matrix
    #
    cnf = compute_conf(ref, hyp)

    # compute the average error rate for norm (0)
    #
    avg_norm = float(1.0) - (float(cnf[0][0]) / float(cnf[0].sum()))
    avg_nneo = float(1.0) - (float(cnf[2][2]) / float(cnf[2].sum()))
    avg_infl = float(1.0) - (float(cnf[3][3]) / float(cnf[3].sum()))
    avg_dcis = float(1.0) - (float(cnf[5][5]) / float(cnf[5].sum()))
    avg_indc = float(1.0) - (float(cnf[6][6]) / float(cnf[6].sum()))
    avg_bckg = float(1.0) - (float(cnf[8][8]) / float(cnf[8].sum()))

    # compute the average of these
    #
    avg_err = (avg_norm + avg_nneo + avg_infl + avg_dcis + avg_indc) / float(5)

    # display a summary
    #
    print("Legend:")
    print("0 = norm <**")
    print("1 = artf")
    print("2 = nneo <**")
    print("3 = infl <**")
    print("4 = susp")
    print("5 = dcis <**")
    print("6 = indc <**")
    print("7 = null")
    print("8 = bckg <**")
    print("")
    print("Confusion Matrix:")
    print(cnf)
    print("")
    print("Scoring Summary:")
    print(" (0) norm = %10.4f%%" % (avg_norm * 100.0))
    print(" (2) nneo = %10.4f%%" % (avg_nneo * 100.0))
    print(" (3) infl = %10.4f%%" % (avg_infl * 100.0))
    print(" (5) dcis = %10.4f%%" % (avg_dcis * 100.0))
    print(" (6) indc = %10.4f%%" % (avg_indc * 100.0))
    print(" (8) bckg = %10.4f%%" % (avg_bckg * 100.0))
    print("")
    print(" avg lbls = %10.4f%%" % (avg_err * 100.0))
    print(" avg bckg = %10.4f%%" % (avg_bckg * 100.0))
    print("    score = %10.4f%% <**" % \
          ((0.90 * avg_err + 0.10 * avg_bckg) * 100.0))

    # After making predictions
    predictions_df = pd.DataFrame(hyp, columns=['label'])
    predictions_df.to_csv('my_predictions.csv', index=False)

    # exit gracefully
    #
    return True

# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[0:])

#
# end of file