# coding: UTF-8
import glob
import os
import time

import cv2

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from utils import classifier


save_to = 'classifiers/'
if not os.path.exists(save_to):
    os.makedirs(save_to)

# DataFrame to record cross validation result
df_cv_linear_svc = pd.DataFrame()


def main():
    # Load the images
    t0 = time.time()
    images_rgb, labels = classifier.load_data(cv2.COLOR_BGR2RGB)
    print('Loading = {} seconds'.format(time.time() - t0))

    t0 = time.time()
    hog_features_list = []
    for image in images_rgb:
        y_cr_cb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        hog_features_list.append(classifier.get_hog_features(y_cr_cb, hog_channels=[0, 1, 2]))
    hog_features = np.array(hog_features_list)
    print('HOG features = {} seconds'.format(time.time() - t0))

    features = hog_features
    X, y = shuffle(features, labels, random_state=0)
    print('X.shape = {}'.format(X.shape))

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    joblib.dump(scaler, os.path.join('classifiers', 'standard_scaler.pkl'))

    # train, test split
    X_train_origin, X_test, y_train_origin, y_test = \
        train_test_split(X_scaled, y, random_state=0, stratify=y)

    # Linear SVM
    # Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for i, (train_index, test_index) in enumerate(skf.split(X_train_origin, y_train_origin)):
        print('Start Fold {}'.format(i + 1))
        t0_fold = time.time()

        X_train, y_train = X_train_origin[train_index], y_train_origin[train_index]
        X_val, y_val = X_train_origin[test_index], y_train_origin[test_index]

        clf = LinearSVC()
        t0_fit = time.time()
        clf.fit(X_train, y_train)
        print('Fit = {} seconds'.format(time.time() - t0_fit))

        df_cv_linear_svc.loc[i, 'acc'] = clf.score(X_train, y_train)
        df_cv_linear_svc.loc[i, 'val_acc'] = clf.score(X_val, y_val)

        print('Fold {} = {} seconds'.format(i + 1, time.time() - t0_fold))

    print(df_cv_linear_svc)

    # Train for test
    clf = LinearSVC()
    t0 = time.time()
    clf.fit(X_train_origin, y_train_origin)
    print('Train = {} seconds'.format(time.time() - t0))
    # Model persistence
    from sklearn.externals import joblib
    joblib.dump(clf, os.path.join(save_to, 'linear_svm.pkl'))
    # Test accuracy
    print('test acc = {}'.format(clf.score(X_test, y_test)))


if __name__ == '__main__':
    main()
