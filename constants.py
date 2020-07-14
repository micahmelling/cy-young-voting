from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from hyperopt import hp


INDIVIDUAL_ID = 'playerID'
TARGET = 'pointsWon'
FEATURES_TO_DROP = ['playerID', 'yearID']
DIAGNOSTICS_DIRECTORY = 'diagnostics'
MODELS_DIRECTORY = 'models'
PLOTS_DIRECTORY = 'plots'
TEST_SET_DIRECTORY = 'test_set_scores'
SHAP_VALUES_DIRECTORY = 'shap_values'
DATA_DIRECTORY = 'data'
CV_SPLITS = 5


PARAM_SPACE_CONSTANT = {'feature_selector__percentile': hp.randint('feature_selector__percentile', 10, 100)}


GRADIENT_BOOSTING_PARAM_GRID = {
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5),
    'model__n_estimators': hp.randint('model__n_estimators', 75, 150),
    'model__max_depth': hp.randint('model__max_depth', 2, 16)
}

XGBOOST_PARAM_GRID = {
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5),
    'model__n_estimators': hp.randint('model__n_estimators', 75, 150),
    'model__max_depth': hp.randint('model__max_depth', 2, 16),
}

CAT_BOOST_PARAM_GRID = {
    'model__depth': hp.randint('model__depth', 2, 16),
    'model__l2_leaf_reg': hp.randint('model__l2_leaf_reg', 1, 10),
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5)
}

LITEGBM_PARAM_GRID = {
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5),
    'model__n_estimators': hp.randint('model__n_estimators', 75, 150),
    'model__max_depth': hp.randint('model__max_depth', 2, 16),
    'model__num_leaves': hp.randint('model__num_leaves', 10, 100)
}

MODEL_TRAINING_DICT = {
    'sklearn_gradient_boosting': [GradientBoostingRegressor(), GRADIENT_BOOSTING_PARAM_GRID, 200],
    'xgboost': [XGBRegressor(), XGBOOST_PARAM_GRID, 200],
    'lightgbm': [LGBMRegressor(), LITEGBM_PARAM_GRID, 200],
    'catBoost': [CatBoostRegressor(silent=True, n_estimators=250), CAT_BOOST_PARAM_GRID, 10]
}
