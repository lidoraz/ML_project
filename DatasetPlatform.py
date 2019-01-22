from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# from joblib import Parallel, delayed
# import multiprocessing
import numpy as np
import pandas as pd

# # Set random seed
# np.random.seed(0)

def evaluate_best_est(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy
def ds1_heart():
    df = pd.read_table('ds/heart.dat', delimiter=' ', header=0)
    X = df.drop('y',axis=1)
    y = df['y']
    y[y == 2] = -1

    return X, y, "ds1_heart"


def ds2_whiteWine():
    df = pd.read_table('ds/winequality-white.csv', delimiter=';', header=0)
    X = df.drop('quality', axis=1)
    y = df['quality']
    return X, y, "ds2_whiteWine"


def ds3_fertility():
    # converted x1 to: M-0,F-1,I-2
    df = pd.read_table('ds/fertility_Diagnosis.txt', delimiter=',', header=0)
    X = df.drop('y', axis=1)
    y = df['y']
    y = y.replace('O', 1)
    y = y.replace('N', -1)
    return X, y, "ds3_fertility"


def ds4_hepatitis():
    df = pd.read_table('ds/hepatitis.data', delimiter=',', header=None)
    df = df[(df != '?').all(axis=1)]
    X = df.drop(0, axis=1)
    y = df[0]
    # fill missing values
    X = X.replace('?', np.nan)
    X.sample(5)
    X = X.fillna(X.mean()).convert_objects(convert_numeric=True)
    y = y.replace(2, -1)
    return X, y, "ds4_hepatitis"


def ds5_iris():
    # converted x1 to: M-0,F-1,I-2
    df = pd.read_table('ds/iris.data', delimiter=',', header=0)
    # keep it binary:
    df = df.loc[df['y'] != 'Iris-virginica']
    df = df.replace('Iris-setosa', -1)
    df = df.replace('Iris-versicolor', 1)

    df = df.sample(frac=1) # this will shuffle the data set
    X = df.drop('y', axis=1)
    y = df['y']
    return X, y, "ds5_iris"


def ds6_soybean():
    df = pd.read_table('ds/soybean-large.data', delimiter=',', header=None)
    df = df[(df != '?').all(axis=1)]
    df = df.fillna(0)
    X = df.drop(0, axis=1)
    y = df[0]
    return X, y, "ds6_soybean"


def ds7_ozone():
    df = pd.read_table('ds/eighthr.data', delimiter=',', header=None)
    df = df[(df != '?').all(axis=1)]
    df = df.drop(0, axis=1)
    X = df[df.columns[:-1]]
    y = df.iloc[:,-1]
    return X, y, "ds7_ozone"


def ds8_yeast():
    # dropped x1 for same reason as ds4
    df = pd.read_table('ds/yeast.data', delimiter=',', header=0)
    df = df.drop('x1', axis=1)
    X = df.drop('y', axis=1)
    y = df['y']
    return X, y, "ds8_yeast"


def ds9_ecoli():
    # converted x8 to: cp=0,im=1,1S=2,1L=3,1U=4,om=5,oml=6,pp=7
    # dropped x1:  Sequence Name: Accession number for the SWISS-PROT database.
    df = pd.read_table('ds/ecoli.data', delimiter=',', header=0)
    df = df.drop('x1', axis=1)
    X = df.drop('y', axis=1)
    y = df['y']
    return X, y, "ds9_ecoli"


def ds10_redWine():
    df = pd.read_table('ds/winequality-red.csv', delimiter=';', header=0)
    X = df.drop('quality', axis=1)
    y = df['quality']
    return X, y, "ds10_redWine"


def evaluate(X, y,t_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size)

    from SpectralMirror import MixtureLRSpectralMirroring
    mixture = MixtureLRSpectralMirroring()
    mixture.fit(X_train, y_train, 2)
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    best_params_predictions = mixture.predict(X_test)
    accuracy = accuracy_score(y_test, best_params_predictions)

    print("accuracy:",accuracy)
    auc_score = roc_auc_score(y_test, best_params_predictions)
    print("acc_score:", auc_score)
    print("f1_score", f1_score)

if __name__ == '__main__':

    dataSets = [
        ds1_heart,
        # ds2_whiteWine, #not binary
        ds3_fertility,
        ds4_hepatitis,
        ds5_iris,
        ds6_soybean,
        ds7_ozone,
        ds8_yeast,
        ds9_ecoli,
        ds10_redWine

    ]
    test_size = 0.2
    print("Results on test data: test_size =", test_size, "from the entire ds:")

    names = []
    np.random.seed(0)

    for idx, datasetFunc in enumerate(dataSets):
        X, y, ds_name = datasetFunc()
        print("ds_name = ", ds_name)
        evaluate(X, y, test_size)