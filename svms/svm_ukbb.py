import  numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import argparse

import warnings
warnings.filterwarnings("ignore")


def get_corr_vector(df, atlas, label, nrois):

    n_corr_mat = int(nrois * (nrois + 1) / 2)
    total_subjects = len(df)
    X = np.zeros((total_subjects, n_corr_mat))
    Y = np.zeros(total_subjects, dtype=int)

    c = 'corrmat_file'


    for i in range(total_subjects):

        corr_file = df[c].iloc[i].replace('ATLAS', atlas)
        corr_vals = np.load(corr_file)
        cc_triu_ids = np.triu_indices(nrois)
        cc_vector = corr_vals[cc_triu_ids]
        X[i] = cc_vector
        Y[i] =  0 if df[label].iloc[i] == 'Male' else 1


    return X, Y




class KernelSVC_Model:

    def __init__(self, kernel='linear'):

        self.kernel = kernel

    def get_model(self):

        return SVC(kernel=self.kernel, max_iter=1000)

    def run(self, x_train, y_train, x_test, y_test):
        model = self.get_model()
        if self.kernel == 'rbf':
            params = {'gamma': [1e-2, 1e-3, 1e-4],
                      'C': [1, 10, 100, 1000]}
        else:
            params = {'C': [1, 10, 100, 1000]}
        rnd_search_cv = RandomizedSearchCV(model, params, n_iter=12, cv=3, random_state=0)
        n_samples = x_train.shape[0]
        cv_samples = int(0.5*n_samples)
        #print('fitting ...')
        rnd_search_cv.fit(x_train[:cv_samples],y_train[:cv_samples])
        rnd_search_cv.best_estimator_.fit(x_train,y_train)
        #print('testing ...')
        y_pred = rnd_search_cv.best_estimator_.predict(x_test)
        acc = balanced_accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn+fp)
        sens = tp / (tp + fn)
        return {"Acc": round(acc, 3),
                "Sens": round(sens, 3),
                "Spec": round(spec, 3)}


def run_main(N, kernel, dataset):
    results_df = pd.DataFrame(columns=['Acc','Sens','Spec'])
    label = 'Sex'
    atlas = 'AAL'
    nrois = 116
    train_df = pd.read_csv(f'../csvfiles/ukbb_{N}.csv')
    test_df = pd.read_csv('../csvfiles/ukbb_test.csv')
    results_df = pd.DataFrame(columns=['Acc','Sens','Spec'])

    for i in range(5):
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        x_train, y_train = get_corr_vector(train_df, atlas, label, nrois)
        x_test, y_test = get_corr_vector(test_df, atlas, label, nrois)

        model = KernelSVC_Model(kernel=kernel)
        r = model.run(x_train, y_train, x_test, y_test)
        results_df.loc[i] = r

    #results_df.to_csv('svm_linear_age.csv')
    print('-'*50)
    print("Test Results of svm-{} on Ukbb {}:".format(kernel, N))
    print('Acc = {:.4f}, {:.5f}'.format(np.mean(results_df.Acc.values), np.std(results_df.Acc.values)))
    print('Sens = {:.4f}, {:.5f}'.format(np.mean(results_df.Sens.values), np.std(results_df.Sens.values)))
    print('Spec = {:.4f}, {:.5f}'.format(np.mean(results_df.Spec.values), np.std(results_df.Spec.values)))
    print('*'*100)




if __name__ == "__main__":
    dataset = 'UKBB'
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', "--kernel", type=str)
    parser.add_argument('-n', "--N", type=int)

    args = parser.parse_args()


    run_main(args.N, args.kernel, dataset)
