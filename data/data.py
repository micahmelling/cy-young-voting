import pandas as pd
import os
import joblib


def get_data():
    """
    Reads csv files from the data directory and merges them into a dataset for modeling. The dataset considers
    the number of points won in the Cy Young races for each year, which will be our target for modeling. It also
    includes many standard pitching metrics, such as wins and ERA. The files are from the open-source Lahman Database.
    :return: pandas dataframe
    """
    awards_df = pd.read_csv(os.path.join('data', 'awards_share.csv'))
    awards_df = awards_df.loc[awards_df['awardID'] == 'Cy Young']
    awards_df = awards_df[['yearID', 'playerID', 'pointsWon']]
    pitching_df = pd.read_csv(os.path.join('data', 'pitching_stats.csv'))
    people_df = pd.read_csv(os.path.join('data', 'people.csv'))[['playerID', 'throws']]
    merged_df = pd.merge(awards_df, pitching_df, how='inner', on=['playerID', 'yearID'])
    merged_df = pd.merge(merged_df, people_df, how='inner', on='playerID')
    joblib.dump(merged_df, os.path.join('data', 'training_data.pkl'))
    return merged_df
