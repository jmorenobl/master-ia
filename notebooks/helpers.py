import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dateparser
import itertools

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from datetime import timedelta
from ta import add_all_ta_features


def normalize_price(df):
    """
    Given a dataframe with prices, return a new dataframe normalized.

    Arguments
    ---------
    df:           dataframe with prices


    Returns
    -------
    Pandas dataframe with the prices normalized


    Usage
    -----
    df = normalize_price(df) # title of graph
    """
    return (df/df[0]) - 1


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          figsize=(18, 10)):
    """
    Given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tick_params(axis='both', which='both',length=0)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


class Featurizer:
    def __init__(self, dataframe):
        self._df = dataframe  # Untouched version
        self.df = dataframe.copy()  # Working copy

    def _apply_lag(self, num_days=0):
        if num_days > 0:
            lag_dict = {}
            df = self.df.copy()

            # 1. Create different dataframes with the lags
            for day in range(1, num_days + 1):
                if day == 1:
                    lag_dict[day] = df.shift()
                    lag_dict[day].rename(columns=lambda x: x + '_lag' + str(day), inplace=True)
                else:
                    previous_day = day - 1
                    lag_dict[day] = lag_dict[previous_day].shift()
                    lag_dict[day].rename(columns=lambda x: x.replace('lag' + str(previous_day), 'lag' + str(day)), inplace=True)

            # 2. Join all the dataframes
            for day in range(1, num_days + 1):
                df = df.join(lag_dict[day], how='outer')

            # 3. Clean empty rows
            df.dropna(inplace=True)

            self.df = df

    def _add_calendar_related_features(self, week_day=True, month_day=True, month=True):
        index_date = pd.DataFrame({
            'stock_date': self.df.index.to_pydatetime()
        }, index=self.df.index)

        if 'week_day' not in self.df.columns:
            self.df['week_day'] = index_date['stock_date'].dt.weekday

        if 'month_day' not in self.df.columns:
            self.df['month_day'] = index_date['stock_date'].dt.day

        if 'month' not in self.df.columns:
            self.df['month'] = index_date['stock_date'].dt.month

    def _add_technical_analysis_features(self):
        self.df = add_all_ta_features(self.df, "Open", "High", "Low", "Close", "Volume", fillna=True)
        self.df['intraday_return'] = 100 * (self.df['Close'] - self.df['Open']) / self.df['Close']
        self.df['close_pct_change'] = self.df['Close'].pct_change()
        self.df['close_log_ret'] = np.log(self.df['Close']) - np.log(self.df['Close'].shift(1))
        self.df['open_pct_change'] = self.df['Open'].pct_change()
        self.df['open_log_ret'] = np.log(self.df['Open']) - np.log(self.df['Open'].shift(1))
        self.df['high_pct_change'] = self.df['High'].pct_change()
        self.df['high_log_ret'] = np.log(self.df['High']) - np.log(self.df['High'].shift(1))
        self.df['low_pct_change'] = self.df['Low'].pct_change()
        self.df['low_log_ret'] = np.log(self.df['Low']) - np.log(self.df['Low'].shift(1))
        self.df['volume_pct_change'] = self.df['Volume'].pct_change()
        self.df['volume_log_ret'] = np.log(self.df['Volume']) - np.log(self.df['Volume'].shift(1))
        self.df['closet1_opent_diff'] = self.df['Open'] - self.df['Close'].shift(1)
        self.df['closet1_opent_trend'] = np.where(
            self.df['closet1_opent_diff'] < 0,
            -1, np.where(
                self.df['closet1_opent_diff'] > 0, 1, 0
            )
        )
        self.df['close_trend'] = np.where(
            self.df['close_pct_change'] < 0,
            -1, np.where(
                self.df['close_pct_change'] > 1, 1, 0
            )
        )

        self.df['percent_b'] =  (
            (self.df['Close'] - self.df['volatility_bbl']) / (self.df['volatility_bbh'] - self.df['volatility_bbl'])
        )

        self.df['bandwidth_b'] =  (self.df['volatility_bbh'] - self.df['volatility_bbl'] / self.df['volatility_bbm'])
        self.df['percentb_signal'] = np.where(
            self.df['percent_b'] < 0,
            -1, np.where(
                self.df['percent_b'] > 1, 1, 0
            )
        )

    def _forecast_horizon(self, numdays_ahead=1):
        self._predicting_days = numdays_ahead
        self.df['PREDICTOR_0dayAhead'] = np.where(
            self.df['others_dr'] < 0, -1, np.where(
                self.df['others_dr'] > 0, 1, 1
            )
        )
        #self.df['PREDICTOR_0dayAhead'] = self.df['closet1_opent_trend']

        for day in range(1, numdays_ahead + 1):
            self.df[f'PREDICTOR_{day}dayAhead'] = self.df[f'PREDICTOR_{day-1}dayAhead'].shift(-1)

        #self.df.drop(columns=['PREDICTOR_0dayAhead'], inplace=True)

    def reset(self):
        self.df = self._df.copy()

    def add_features(self, forecast_horizon=1, with_ndays_lag=0, week_day_feature=True, month_day_feature=True, month_feature=True):
        self._add_calendar_related_features(
            week_day=week_day_feature, month_day=month_day_feature, month=month_feature
        )
        self._add_technical_analysis_features()
        self._apply_lag(num_days=with_ndays_lag)
        self._forecast_horizon(numdays_ahead=forecast_horizon)

    def crossval_split(self, split_date=None, split_session=20, predict_day=1):
        predicting_days = range(1, self._predicting_days + 1)
        if predict_day not in predicting_days:
            raise ValueError(
                'The parameter "predict_day" is not in the valid range'
                f' of predicting days: {predicting_days}'
            )

        remove_cols = [f'PREDICTOR_{day}dayAhead' for day in predicting_days if day != predict_day]

        df = self.df.drop(columns=remove_cols).dropna()

        if split_date:
            parsed_split_date = dateparser.parse(split_date)

            train_until_date = parsed_split_date.strftime('%Y-%m-%d')
            test_from_date = (parsed_split_date + timedelta(days=1)).strftime('%Y-%m-%d')

            training_data = df[:train_until_date]
            test_data = df[test_from_date:]
        else:
            training_data = df[:-split_session]
            test_data = df[-split_session:]

        X_train = training_data.dropna().drop(columns=[f'PREDICTOR_{predict_day}dayAhead'])
        y_train = training_data[f'PREDICTOR_{predict_day}dayAhead']

        X_test = test_data.dropna().drop(columns=[f'PREDICTOR_{predict_day}dayAhead'])
        y_test = test_data[f'PREDICTOR_{predict_day}dayAhead']

        return X_train, y_train, X_test, y_test

    def train_split(self, predict_day=1):
        predicting_days = range(1, self._predicting_days + 1)
        if predict_day not in predicting_days:
            raise ValueError(
                'The parameter "predict_day" is not in the valid range'
                f' of predicting days: {predicting_days}'
            )
        remove_cols = [f'PREDICTOR_{day}dayAhead' for day in predicting_days if day != predict_day]

        df = self.df.drop(columns=remove_cols).dropna()

        X = df.dropna().drop(columns=[f'PREDICTOR_{predict_day}dayAhead'])
        y = df[f'PREDICTOR_{predict_day}dayAhead']

        return X, y

    def timeseries_cross_validator(self, num_samples, test_size=20):
        num_splits = int((num_samples - test_size) / test_size)

        return TimeSeriesSplit(n_splits=num_splits)

    def unpredicted_data(self):
        predicting_days = range(1, self._predicting_days + 1)
        remove_cols = [f'PREDICTOR_{day}dayAhead' for day in predicting_days]

        return self.df.drop(columns=remove_cols)[-1:]

