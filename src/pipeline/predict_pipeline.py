# The predict pipeline is used to give the model new data for predictions
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transform input data and predict
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomException(e, sys)

# CustomData maps inputs from frontend (app.py, html) to the backend (components) for predictions
class CustomData:
    def __init__(
        self,
        #tourney_id: str,
        #surface: str,
        #draw_size: int,
        #tourney_level: str,
        #tourney_date: int,
        #p1_id: int,
        p1_seed: int,
        #p1_ht: int,
        #p1_age: int,
        #p2_id: int,
        p2_seed: int,
        #p2_ht: int,
        #p2_age: int,
        #best_of: int,
        #round: str,
        p1_ace: int,
        p1_df: int,
        p1_svpt: int,
        p1_1stIn: int,
        p1_1stWon: int,
        p1_2ndWon: int,
        #p1_SvGms: int,
        #p1_bpSaved: int,
        p1_bpFaced: int,
        p2_ace: int,
        p2_df: int,
        p2_svpt: int,
        p2_1stIn: int,
        p2_1stWon: int,
        p2_2ndWon: int,
        #p2_SvGms: int,
        #p2_bpSaved: int,
        p2_bpFaced: int,
        #p1_rank: int,
        p1_rank_points: int,
        #p2_rank: int,
        p2_rank_points: int
    ):
        #self.tourney_id = tourney_id # could drop
        #self.surface = surface # could drop
        #self.draw_size = draw_size # could drop
        #self.tourney_level = tourney_level # could drop
        #self.tourney_date = tourney_date # could drop
        #self.p1_id = p1_id # could drop
        self.p1_seed = p1_seed
        #self.p1_ht = p1_ht # could drop
        #self.p1_age = p1_age # could drop
        #self.p2_id = p2_id # could drop
        self.p2_seed = p2_seed
        #self.p2_ht = p2_ht # could drop
        #self.p2_age = p2_age # could drop
        #self.best_of = best_of # could drop
        #self.round = round # could drop
        self.p1_ace = p1_ace
        self.p1_df = p1_df
        self.p1_svpt = p1_svpt
        self.p1_1stIn = p1_1stIn
        self.p1_1stWon = p1_1stWon
        self.p1_2ndWon = p1_2ndWon
        #self.p1_SvGms = p1_SvGms # could drop
        #self.p1_bpSaved = p1_bpSaved # could drop
        self.p1_bpFaced = p1_bpFaced
        self.p2_ace = p2_ace
        self.p2_df = p2_df
        self.p2_svpt = p2_svpt
        self.p2_1stIn = p2_1stIn
        self.p2_1stWon = p2_1stWon
        self.p2_2ndWon = p2_2ndWon
        #self.p2_SvGms = p2_SvGms # could drop
        #self.p2_bpSaved = p2_bpSaved # could drop
        self.p2_bpFaced = p2_bpFaced
        #self.p1_rank = p1_rank # could drop
        self.p1_rank_points = p1_rank_points
        #self.p2_rank = p2_rank # could drop
        self.p2_rank_points = p2_rank_points

    # Reformats custom data as Pandas DataFrame for backend
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                #"tourney_id": [self.tourney_id],
                #"surface": [self.surface],
                #"draw_size": [self.draw_size],
                #"tourney_level": [self.tourney_level],
                #"tourney_date": [self.tourney_date],
                #"p1_id": [self.p1_id],
                "p1_seed": [self.p1_seed],
                #"p1_ht": [self.p1_ht],
                #"p1_age": [self.p1_age],
                #"p2_id": [self.p2_id],
                "p2_seed": [self.p2_seed],
                #"p2_ht": [self.p2_ht],
                #"p2_age": [self.p2_age],
                #"best_of": [self.best_of],
                #"round": [self.round],
                "p1_ace": [self.p1_ace],
                "p1_df": [self.p1_df],
                "p1_svpt": [self.p1_svpt],
                "p1_1stIn": [self.p1_1stIn],
                "p1_1stWon": [self.p1_1stWon],
                "p1_2ndWon": [self.p1_2ndWon],
                #"p1_SvGms": [self.p1_SvGms],
                #"p1_bpSaved": [self.p1_bpSaved],
                "p1_bpFaced": [self.p1_bpFaced],
                "p2_ace": [self.p2_ace],
                "p2_df": [self.p2_df],
                "p2_svpt": [self.p2_svpt],
                "p2_1stIn": [self.p2_1stIn],
                "p2_1stWon": [self.p2_1stWon],
                "p2_2ndWon": [self.p2_2ndWon],
                #"p2_SvGms": [self.p2_SvGms],
                #"p2_bpSaved": [self.p2_bpSaved],
                "p2_bpFaced": [self.p2_bpFaced],
                #"p1_rank": [self.p1_rank],
                "p1_rank_points": [self.p1_rank_points],
                #"p2_rank": [self.p2_rank],
                "p2_rank_points": [self.p2_rank_points]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
