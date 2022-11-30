# Basic
import numpy as np
from sklearn.model_selection import ShuffleSplit, train_test_split, KFold


class miscFunctions():
    def RemoveNaN(self, X_inp, Y_inp, goodIds=None):
        """ METHODS, INPUTS, & OUTPUTS"""
        ################################################################################################################
        # METHODS:
        # Two methods:
        #   1) if 'goodIds' == None, then the code looks through all the inputs and outputs
        #      and deletes any rows that contain NaN values. Full rows are deleted to preserve
        #      an equal column length between the properties and features.
        #   2) If 'goodIds' != None, then the code goes through the inputs and outputs and
        #      removes any columns whose respective point in the logical 'goodIds' array is
        #      zero, then keep the ones that have one.
        # INPUTS:
        # - X_inp: an (n x m) array holding the features to be used in the regression.
        # - Y_inp: an (n x 1) vector holding the outputs for the regression.
        # - goodIds: an (n x 1) logical vector (only 1's and 0') holding the places that the
        #            features or outputs have a NaN value
        # OUTPUTS:
        # - X: Final feature array without NaN values
        # - Y: Final property array without NaN values
        ################################################################################################################

        szX = np.shape(X_inp)  # the shape (n x m) of the input features
        numPts = szX[0]  # number of data points in the Y_inp array (# of columns)
        numFeatures = szX[1]  # number of features (# of rows)

        X_inp.shape = (numPts, numFeatures)
        Y_inp.shape = (numPts, 1)

        # If no 'goodIds's is input, then one is created

        try:
            if goodIds == None:
                goodIds = []
                for i in range(numPts):
                    if (np.isfinite(Y_inp[i]) and np.isfinite(X_inp[i, :]).all()):
                        # Checks to make sure all rows (inout and output) and not NaNs
                        goodIds.append(1)
                    else:
                        goodIds.append(0)
                goodIds.shape = (numPts, 1)
        except:
            goodIds = goodIds
        # Goes through and delete columns where goodIds[i] == 0
        numGoodPts = np.sum(goodIds)
        Y = np.zeros((numGoodPts, 1))  # storage for the final Y
        X = np.zeros((numGoodPts, numFeatures))  # storage for the final X
        j = 0
        for i in range(numPts):
            if goodIds[i] == 1:
                Y[j, 0] = Y_inp[i, 0]
                X[j, :] = X_inp[i, :]
                j += 1
        return X, Y

    def getPredMetrics(self, Y_true, Y_pred):
        """ METHODS, INPUTS, & OUTPUTS"""
        ################################################################################################################
        # METHODS:
        # takes in the true and predicted values and outputs the RMSE, R^2, and
        # correlation coefficient.
        # -> Equations:
        #       RMSE: sqrt[ (1/N) * SUM_(i=1->N){ (Yt[i] - Yp[i])^2 }]
        #       R^2: R^2 = 1 - (SSR / SST)
        #           SSR: SUM_(i=1->N){ (Yt[i] - Yp[i])^2 }
        #           SST: SUM_(i=1->N){ (Yt[i] - meanYt)^2 }
        #       Cor: SUM_(i=1->N){ (Yt[i] - meanYt)*(Yp[i] - meanYp) } / sqrt[SUM_(i=1->N){ (Yt[i] - meanYt)^2 }*SUM_(i=1->N){ (Yp[i] - meanYp)^2 } ]
        # INPUTS:
        # - Y_true: the actual Y
        # - Y_pred: predicted Y
        # OUTPUTS:
        # - RMSE: Root-mean squared error
        # - R^2:  Coefficient of Determination
        # - Cor:  Correlation Coefficient
        ################################################################################################################

        # Inital Variables
        N = len(Y_true)  # number of data points
        meanYt = np.mean(Y_true)
        meanYp = np.mean(Y_pred)

        # RMSE: --------------------------------------------------------------------------------------------------------
        rmse_0 = 0
        for i in range(N):
            rmse_0 += (Y_true[i] - Y_pred[i]) ** 2
        RMSE = np.sqrt(rmse_0 / N)

        # R^2: ---------------------------------------------------------------------------------------------------------
        SSR = 0
        SST = 0
        for i in range(N):
            SSR += (Y_true[i] - Y_pred[i]) ** 2
            SST += (Y_true[i] - meanYt) ** 2
        R2 = 1 - (SSR / SST)

        # Cor: ---------------------------------------------------------------------------------------------------------
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i in range(N):
            sum1 += (Y_true[i] - meanYt) * (Y_pred[i] - meanYp)
            sum2 += (Y_true[i] - meanYt) ** 2
            sum3 += (Y_pred[i] - meanYp) ** 2
        #Cor = sum1 / np.sqrt(sum2 * sum3)
        Cor = np.corrcoef(Y_true, Y_pred)[0][1]

        return RMSE, R2, Cor

    def getKfoldSplits(self, X, Nk=5, seed='random'):
        """ METHODS, INPUTS, & OUTPUTS"""
        ################################################################################################################
        # METHODS:
        # Given an input array and the number of K-fold splits for the cross validation,
        # splits the data 'Nk' number of times based on a seed. These values will represent
        # the training-and-testing indices for 'Nk' different splits.
        # INPUTS:
        # - X: input array
        # - Nk: number of splits
        # - seed: seed for the splitting
        # OUTPUTS:
        # - train_idx: list holding 'Nk' number of sets of training indices
        # - test_idx: list holding 'Nk' number of sets of testing indices that correspond to
        #             the indices not in 'train_idx'
        ################################################################################################################

        if seed == 'random':  # Gets the seed for the splits
            seed = np.random.randint(2 ** 31)
            print("SEED: ", seed)

        kf = KFold(n_splits=Nk, shuffle=True, random_state=seed)
        split = list(kf.split(X))  # list (Nk long) holding a pair of testing and training indices

        train_idx = []
        test_idx = []

        for i in range(Nk):
            train_idx.append(split[i][0])  # split[i][0] == i-th pair, training set
            test_idx.append(split[i][1])  # split[i][1] == i-th pair, testing set

        return train_idx, test_idx
