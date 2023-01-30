import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

for predictor in ['XGBoost', 'SVM', 'GB', 'RF']:
    results = pd.DataFrame(index=np.arange(1, 11))
    for i in range(1, 11):
        train_way = str(i) + "_train_set_descriptor_random.csv"
        dev_way = str(i) + "_dev_set_descriptor_random.csv"
        test_way = str(i) + "_test_set_descriptor_random.csv"

        train = pd.read_csv(train_way)
        dev = pd.read_csv(dev_way)
        test = pd.read_csv(test_way)

        X_train = train.iloc[:, 5:211]
        y_train = train.iloc[:, 4]
        X_dev = dev.iloc[:, 5:211]
        y_dev = dev.iloc[:, 4]
        X_test = test.iloc[:, 5:211]
        y_test = test.iloc[:, 4]

        if predictor == 'XGBoost':
            regr = XGBRegressor(learning_rate=0.05, n_estimators=1600, max_depth=7,
                                random_state=0)
            regr.fit(X_train, y_train)
        elif predictor == 'SVM':
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_dev = pd.DataFrame(scaler.transform(X_dev), columns=X_dev.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

            regr = SVR(C=8, epsilon=0.03, gamma=0.01)
            regr.fit(X_train, y_train)

        elif predictor == 'GB':
            regr = GradientBoostingRegressor(learning_rate=0.2, n_estimators=1100,
                                             random_state=0)
            regr.fit(X_train, y_train)

        elif predictor == 'RF':
            regr = RandomForestRegressor(n_estimators=500, random_state=0)
            regr.fit(X_train, y_train)


        y_train_predict = regr.predict(X_train)
        y_dev_predict = regr.predict(X_dev)
        y_test_predict = regr.predict(X_test)

        r2_train = r2_score(y_train, y_train_predict)
        mse_train = mean_squared_error(y_train, y_train_predict)
        mae_train = mean_absolute_error(y_train, y_train_predict)
        r2_dev = r2_score(y_dev, y_dev_predict)
        mse_dev = mean_squared_error(y_dev, y_dev_predict)
        mae_dev = mean_absolute_error(y_dev, y_dev_predict)
        r2_test = r2_score(y_test, y_test_predict)
        mse_test = mean_squared_error(y_test, y_test_predict)
        mae_test = mean_absolute_error(y_test, y_test_predict)

        results.loc[i, "train_score"] = r2_train
        results.loc[i, "train_mse"] = mse_train
        results.loc[i, "train_mae"] = mae_train
        results.loc[i, "test_score"] = r2_test
        results.loc[i, "test_mse"] = mse_test
        results.loc[i, "test_mae"] = mae_test
        results.loc[i, "val_score"] = r2_dev
        results.loc[i, "val_mse"] = mse_dev
        results.loc[i, "val_mae"] = mae_dev

    results.to_csv(r'results\{}_performance.csv'.format(predictor))








