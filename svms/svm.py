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
import warnings
warnings.filterwarnings("ignore")


def get_corr_vector(df, atlas, label, nrois):

    n_corr_mat = int(nrois * (nrois + 1) / 2)
    total_subjects = len(df)
    X = np.zeros((total_subjects, n_corr_mat))
    Y = np.zeros(total_subjects, dtype=int)

    c = 'cc_file'


    for i in range(total_subjects):

        corr_file = df[c].iloc[i].replace('ATLAS', atlas)
        corr_vals = np.load(corr_file)
        cc_triu_ids = np.triu_indices(nrois)
        cc_vector = corr_vals[cc_triu_ids]
        X[i] = cc_vector
        Y[i] = df[label].iloc[i]


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
        cv_samples = int(0.3*n_samples)
        print('fitting ...')
        rnd_search_cv.fit(x_train[:cv_samples],y_train[:cv_samples])
        rnd_search_cv.best_estimator_.fit(x_train,y_train)
        print('testing ...')
        y_pred = rnd_search_cv.best_estimator_.predict(x_test)
        acc = balanced_accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn+fp)
        sens = tp / (tp + fn)
        return {"Acc": round(acc, 3),
                "Sens": round(sens, 3),
                "Spec": round(spec, 3)}


def run_main(kernel, dataset):
    results_df = pd.DataFrame(columns=['Acc','Sens','Spec'])
    label = 'Diagnosis'
    atlas = 'HO_112' if dataset == 'Mddrest' else 'craddock_200'
    nrois = 112 if dataset == 'Mddrest' else 195
    k = 0
    df_path = '../csvfiles/mddrest.csv' if dataset == 'Mddrest' else '../csvfiles/abide.csv'
    df = pd.read_csv(df_path)
    df = df.sample(frac=1).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=5)
    kf.get_n_splits(df,df[label])
    for train_index, test_index in kf.split(df,df[label]):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        print('preparing data ...')
        x_train, y_train = get_corr_vector(df_train, atlas, label, nrois)
        x_test, y_test = get_corr_vector(df_test, atlas, label, nrois)

        '''scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)'''

        model = KernelSVC_Model(kernel=kernel)

        r = model.run(x_train, y_train, x_test, y_test)
        print('fold {}:'.format(k), r)
        results_df.loc[k] = r
        k += 1
    #results_df.to_csv('svm_linear_age.csv')
    print('-'*50)
    print("Test Results:")
    print('Acc = {:.4f}, {:.5f}'.format(np.mean(results_df.Acc.values), np.std(results_df.Acc.values)))
    print('Sens = {:.4f}, {:.5f}'.format(np.mean(results_df.Sens.values), np.std(results_df.Sens.values)))
    print('Spec = {:.4f}, {:.5f}'.format(np.mean(results_df.Spec.values), np.std(results_df.Spec.values)))




if __name__ == "__main__":
    dataset = 'Abide'
    kernel = 'rbf'
    run_main(kernel, dataset)
