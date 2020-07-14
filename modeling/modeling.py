import random
import joblib
import os
import numpy as np
import pandas as pd
import math
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, Trials, space_eval

from constants import FEATURES_TO_DROP, DIAGNOSTICS_DIRECTORY, MODELS_DIRECTORY, TEST_SET_DIRECTORY, \
    SHAP_VALUES_DIRECTORY, PARAM_SPACE_CONSTANT, INDIVIDUAL_ID
from helpers.helpers import FeaturesToDict, drop_features, get_num_and_cat_feature_names, timer


def create_custom_train_test_split(df, target, test_set_percent, individual_id):
    """
    Creates a custom train-test split to ensure that players used for training are not also used in the holdout
    test set. This will help ensure our test set is not leaked into our training data in any way.
    :param df: pandas dataframe
    :param target: the target column
    :param test_set_percent: the percent of individuals to reserve for the test set
    :param individual_id: the id column to uniquely identify individual players
    :return: dataframes for x_train, y_train, x_test, y_test
    """

    test_set_n = int(df[individual_id].nunique() * test_set_percent)
    unique_ids = list(set(df[individual_id].tolist()))
    test_set_ids = random.sample(unique_ids, test_set_n)
    train_df = df.loc[~df[individual_id].isin(test_set_ids)]
    train_df.reset_index(inplace=True, drop=True)
    test_df = df.loc[df[individual_id].isin(test_set_ids)]
    test_df.reset_index(inplace=True, drop=True)
    y_train = train_df[target]
    y_test = test_df[target]
    x_train = train_df.drop(target, 1)
    x_test = test_df.drop(target, 1)
    return x_train, y_train, x_test, y_test


def create_custom_cv(train_df, individual_id, folds):
    """
    Creates a custom train-test split for cross validation. This helps prevent leakage of individual-level player
    effects the model does not capture.
    :param train_df: pandas dataframe
    :param individual_id: the id column to uniquely identify individual players
    :param folds: the number of folds we want to use in k-fold cross validation
    :return: a list of tuples; each list item represent a fold in the k-fold cross validation; the first tuple element
    contains the indices of the training data, and the second tuple element contains the indices of the testing data
    """
    unique_ids = list(set(train_df[individual_id].tolist()))
    test_set_id_sets = np.array_split(unique_ids, folds)
    cv_splits = []
    for test_set_id_set in test_set_id_sets:
        temp_train_ids = train_df.loc[~train_df[individual_id].isin(test_set_id_set)].index.values.astype(int)
        temp_test_ids = train_df.loc[train_df[individual_id].isin(test_set_id_set)].index.values.astype(int)
        cv_splits.append((temp_train_ids, temp_test_ids))
    return cv_splits


def construct_pipeline(numeric_features, categorical_features, model):
    """
    Constructs a scikit-learn pipeline to be used for modeling. A scikit-learn ColumnTransformer is used to apply
    different transformers to the data. For the numeric data, nulls are imputed with the mean, and then features are
    scaled. For categorical data, nulls are imputed with the constant "unknown", and then the DictVectorizer is applied
    to "dummy code" the features. All features are run through a function to drop specified features and to potentially
    drop irrelevant features based on univariate statistical tests. The last step in the pipeline is a model with a
    predict method.
    :param numeric_features: list of numeric features
    :param categorical_features: list of categorical features
    :param model: instantiated model
    :return: scikit-learn pipeline that can be used for fitting and predicting
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown', add_indicator=True)),
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_transformer', numeric_transformer, numeric_features),
            ('categorical_transformer', categorical_transformer, categorical_features)
        ])

    pipeline = Pipeline(steps=
                        [
                            ('feature_dropper', FunctionTransformer(drop_features, validate=False,
                                                                    kw_args={'features_drop_list':
                                                                             FEATURES_TO_DROP})),
                            ('preprocessor', preprocessor),
                            ('feature_selector', SelectPercentile(f_classif)),
                            ('model', model)
                        ])

    return pipeline


def produce_shap_values(pipe, x_test, num_features, model_name):
    """
    Produces a plot of SHAP values to identify important features. Work is required to properly extract feature names
    from the pipeline. Categorical feature names are extracted from the DictVectorizer and combined with the numeric
    feature names. Features are removed based on the feature_selector routine in the pipeline. Once the final feature
    names have been isolated, they are used to rename the columns in x_test.
    :param pipe: scikit-learn pipeline defined in the construct_pipeline function
    :param x_test: x_test dataframe
    :param num_features: list of numeric features used for modeling
    :param model_name: the string name of the model
    """
    # for every feature, grab boolean of if the feature selector kept it
    support = pipe.named_steps['feature_selector'].get_support()
    model = pipe.named_steps['model']
    # remove model
    pipe.steps.pop(len(pipe) - 1)
    # remove feature_selector
    pipe.steps.pop(len(pipe) - 1)
    # transform the dataframe with the remaining pipeline
    x_test = pipe.transform(x_test)
    x_test = pd.DataFrame(x_test)

    # extract categorical feature names nested in our pipeline and combine with known numeric feature names
    dict_vect = pipe.named_steps['preprocessor'].named_transformers_.get('categorical_transformer').named_steps[
        'dict_vectorizer']
    cat_features = dict_vect.feature_names_
    cols_df = pd.DataFrame({'cols': num_features + cat_features, 'support': support})
    cols = cols_df['cols'].tolist()
    # assign column names to our dataframe
    x_test.columns = cols
    # drop columns eliminated by our feature selector
    remove_df = cols_df.loc[cols_df['support'] == False]
    remove_cols = remove_df['cols'].tolist()
    x_test.drop(remove_cols, 1, inplace=True)

    # produce shap values, which needs a model outside a pipeline and a dataframe with column names
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, show=False)
    plt.savefig(os.path.join(DIAGNOSTICS_DIRECTORY, SHAP_VALUES_DIRECTORY, '{}_shap_values'.format(model_name)),
                bbox_inches='tight')
    plt.tight_layout()
    plt.clf()


@timer
def train_model(x_train, y_train, x_test, y_test, model_name, get_pipeline_function, model, param_space, cv_scheme,
                iterations):
    """
    Trains a regression machine learning model. Hyperparameters are optimized via Hyperopt, a Bayesian optimization
    library. A serialized model is saved to disk along with test set predictions, test set scores, and SHAP values.
    :param x_train: predictors for training
    :param y_train: target for training
    :param x_test: predictors for testing
    :param y_test: target for testing
    :param model_name: string name of the model
    :param get_pipeline_function: callable to produce a scikit-learn pipeline
    :param model: instantiated model
    :param param_space: search space for parameter optimization via Hyperopt
    :param cv_scheme: the cross validation routine we want to use
    :param iterations: number of iterations to use for optimization
    """
    print(f'training model {model_name}...')
    numeric_features, categorical_features = get_num_and_cat_feature_names(x_train)
    pipeline = get_pipeline_function(numeric_features, categorical_features, model)
    param_space.update(PARAM_SPACE_CONSTANT)

    def _model_objective(params):
        pipeline.set_params(**params)
        score = cross_val_score(pipeline, x_train, y_train, cv=cv_scheme, scoring='neg_mean_squared_error', n_jobs=-1)
        return 1 - score.mean()

    trials = Trials()
    best = fmin(_model_objective, param_space, algo=tpe.suggest, max_evals=iterations, trials=trials)
    best_params = space_eval(param_space, best)
    pipeline.set_params(**best_params)
    pipeline.fit(x_train, y_train)

    predictions_df = pd.concat([pd.DataFrame(pipeline.predict(x_test), columns=['prediction']), y_test, x_test],
                               axis=1)
    predictions_df['prediction'] = predictions_df['prediction'].clip(0, 250)
    mse = mean_squared_error(y_test, predictions_df['prediction'])
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions_df['prediction'])

    joblib.dump(pipeline, os.path.join(DIAGNOSTICS_DIRECTORY, MODELS_DIRECTORY, f'{model_name}.pkl'))
    predictions_df.to_csv(os.path.join(DIAGNOSTICS_DIRECTORY, TEST_SET_DIRECTORY, f'{model_name}_predictions.csv'),
                          index=False)
    pd.DataFrame({'mae': [mae], 'mse': [mse], 'rmse': [rmse]}).to_csv(os.path.join(DIAGNOSTICS_DIRECTORY,
                                                                                   TEST_SET_DIRECTORY,
                                                                                   f'test_set_results_{model_name}.csv'
                                                                                   ), index=False)
    try:
        produce_shap_values(pipeline, x_test, numeric_features, model_name)
    except Exception as e:
        print(e)
