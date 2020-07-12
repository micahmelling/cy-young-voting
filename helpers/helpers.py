import os
import functools
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

from constants import PLOTS_DIRECTORY, DIAGNOSTICS_DIRECTORY, FEATURES_TO_DROP, TARGET


def _plot_correlation_matrix(df):
    """
    Plots a correlation matrix for all numeric features in a dataframe, excluding those that will be dropped for
    modeling, per constants.py. Saves a png of the plot in the nested plots directory, identified in constant.py.
    :param df: pandas dataframe
    """
    df = df.select_dtypes(exclude=[object])
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIRECTORY, PLOTS_DIRECTORY, 'correlation_matrix_.png'))
    plt.clf()


def _plot_average_by_category(df, fill_na_value='unknown'):
    """
    Produces a bar plot of the average value of the target, defined in constants.py, for each category in the dataframe.
    The plots are saved as png files in the nested plots directory, identified in constants.py. One file per categorical
    column is produced.
    :param df: pandas dataframe
    :param fill_na_value: the value for which to fill categorical nulls
    """
    features = list(df.select_dtypes(include=[object]))
    for feature in features:
        if feature not in FEATURES_TO_DROP:
            df.fillna({feature: fill_na_value}, inplace=True)
            grouped_df = pd.DataFrame(df.groupby(feature)[TARGET].mean())
            grouped_df.columns = ['mean']
            grouped_df.reset_index(inplace=True)
            sns.barplot(x=feature, y='mean', data=grouped_df)
            plt.title('Class Average for ' + feature)
            plt.xticks(rotation=90)
            plt.savefig(os.path.join(DIAGNOSTICS_DIRECTORY, PLOTS_DIRECTORY, 'class_average_for_' + feature + '.png'))
            plt.clf()


def _plot_histogram(df, column):
    """
    Plots a histogram and saves the result as a png file in the nested plots directory, identified in constants.py.
    :param df: pandas dataframe
    :param column: the column for which to make the histogram
    """
    sns.distplot(df[column], kde=False)
    plt.xlabel(column)
    plt.ylabel('density')
    plt.title('Histogram for {}'.format(column))
    plt.savefig(os.path.join(DIAGNOSTICS_DIRECTORY, PLOTS_DIRECTORY, 'histogram_plot_for_' + column + '.png'))
    plt.clf()


def make_diagnostic_plots(df):
    """
    Runs plotting functions in one call to produce a correlation matrix, average value of the target by category,
    and a histogram of the target.
    :param df: pandas dataframe
    """
    _plot_correlation_matrix(df)
    _plot_average_by_category(df)
    _plot_histogram(df, TARGET)


def timer(func):
    """
    Prints the time in seconds it took to run a function. To be used as a decorator.
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def _create_directories_helper(directory, parent):
    """
    Creates a directory if it does not exist.
    :param directory: directory to create
    :param parent: parent directory of the directory argument, which will also be created if necessary
    """
    if not os.path.exists(os.path.join(parent, directory)):
        os.makedirs(os.path.join(parent, directory))


def create_directories(directories_list, parent=os.getcwd()):
    """
    Calls _create_directories_helper to create an arbitrary number of directories
    :param directories_list: list of directories to create
    :param parent: the parent directory for each directory in directories_list
    """
    _create_directories_helper(parent, os.getcwd())
    if parent in directories_list:
        directories_list.remove(parent)
    for directory in directories_list:
        _create_directories_helper(directory, parent)


def drop_features(df, features_drop_list):
    """
    Drops columns from a pandas dataframe.
    :param df: pandas dataframe
    :param features_drop_list: list of features to drop
    :return: pandas dataframe
    """
    return df.drop(features_drop_list, 1)


def get_num_and_cat_feature_names(df):
    """
    Produces lists of numeric and categorical feature names from a dataframe.
    :param df: pandas dataframe
    :return: tuple of two lists, the first containing the names of the numeric features and the second containing the
    names of the categorical features
    """
    numeric_features = list(df.select_dtypes(exclude=[object]))
    categorical_features = list(df.select_dtypes(include=[object]))

    for feature in FEATURES_TO_DROP:
        numeric_features = [f for f in numeric_features if f != feature]
        categorical_features = [f for f in categorical_features if f != feature]
    return numeric_features, categorical_features


class FeaturesToDict(BaseEstimator, TransformerMixin):
    """
    Converts dataframe, or numpy array, into a dictionary oriented by records. This is a necessary pre-processing step
    for DictVectorizer().
    """
    def __int__(self):
        pass

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.to_dict(orient='records')
        return X
