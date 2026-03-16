import os
import random
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    # Train, test, and raw data CSVs go in artifacts folder
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        # ingestion_config is the combination of the three train, test, and raw paths
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered initiate_data_ingestion")
        try:
            df = pd.read_csv('artifacts/wta_matches.csv')
            # Make sure to define match_winner column before any transformations
            df['match_winner'] = 0 
            logging.info("Read dataset as a DataFrame")

            # Before splitting: drop columns, rename winner/loser, scramble winner/loser
            #df = self.drop_columns(df)
            #df = self.rename_winner_loser(df)
            #df = self.scramble_winner_loser(df)

            # Initialize directories for the three CSVs
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train, test, split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # helper function: drop unnecessary columns
    def drop_columns(self, df):
        # define list of columns to drop based on EDA and model training
        columns_to_drop = ['tourney_name', 'surface', 'draw_size', 'tourney_level', 'tourney_date', 
                           'winner_hand', 'loser_hand', 'winner_ht', 'loser_ht', 'winner_ioc', 'loser_ioc',
                           'match_num', 'best_of', 'round', 'minutes', 'score', 'winner_entry', 'loser_entry', 
                           'winner_rank', 'loser_rank', 'w_bpSaved', 'l_bpSaved']
        df = df.drop(columns=columns_to_drop)
        # also drop rows with nan values
        df.dropna()
        return df

    # helper function: rename winner/loser columns to player1/player2
    def rename_winner_loser(self, df):
        for col in df.columns:
            if 'winner' in col:
                df.rename(columns={col: col.replace('winner', 'p1')}, inplace=True)
            elif 'w' in col:
                df.rename(columns={col: col.replace('w', 'p1')}, inplace=True)
            elif 'loser' in col:
                df.rename(columns={col: col.replace('loser', 'p2')}, inplace=True)
            elif 'l' in col:
                df.rename(columns={col: col.replace('l', 'p2')}, inplace=True)
        df['match_winner'] = 0 # create match winner column with all 0s (player 1 for all obs.)
        return df

    # helper function: scramble winner/loser columns to remove bias
    def scramble_winner_loser(self, df):
        cols1 = [c for c in df.columns if 'p1' in c]
        cols2 = [c for c in df.columns if 'p2' in c]
        cols2_targ = [c.replace('p1', 'p2') for c in cols1]

        # copy df for scrambling
        copy = df.copy()

        # generate len(df.index) / 2 random indices. for those observations, swap columns and set match_winner to 1
        maskIdx = [random.randint(0, len(df.index) - 1) for i in range((int)(len(df.index) / 2))]
        df.loc[maskIdx, cols1] = copy.loc[maskIdx, cols2_targ].values
        df.loc[maskIdx, cols2_targ] = copy.loc[maskIdx, cols1].values
        df.loc[maskIdx, 'match_winner'] = 1

        return df

        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    # Note: not currently using preprocessor_path from this return
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
    print(f"r2 score: {r2_score}")