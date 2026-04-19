from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Define a Flask application
application = Flask(__name__)
app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

## Route for GET or POST
# updated comment
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # User requested the /predictdata (GET)
    if request.method == 'GET':
        return render_template('home.html')
    # User entered data and is requesting prediction (POST)
    else:
        data = CustomData(
            #tourney_id = request.form.get('tourney_id'),
            #surface = request.form.get('surface'),
            #draw_size = int(request.form.get('draw_size')),
            #tourney_level = request.form.get('tourney_level'),
            #tourney_date = int(request.form.get('tourney_date')),
            #p1_id = int(request.form.get('p1_id')),
            p1_seed = request.form.get('p1_seed'),
            #p1_ht = int(request.form.get('p1_ht')),
            #p1_age = int(request.form.get('p1_age')),
            #p2_id = int(request.form.get('p2_id')),
            p2_seed = request.form.get('p2_seed'),
            #p2_ht = int(request.form.get('p2_ht')),
            #p2_age = int(request.form.get('p2_age')),
            #best_of = int(request.form.get('best_of')),
            #round = request.form.get('round'),
            p1_ace = int(request.form.get('p1_ace')),
            p1_df = int(request.form.get('p1_df')),
            #p1_svpt = int(request.form.get('p1_svpt')),
            p1_1stIn = int(request.form.get('p1_1stIn')),
            p1_1stWon = int(request.form.get('p1_1stWon')),
            p1_2ndWon = int(request.form.get('p1_2ndWon')),
            #p1_SvGms = int(request.form.get('p1_SvGms')),
            #p1_bpSaved = int(request.form.get('p1_bpSaved')),
            p1_bpFaced = int(request.form.get('p1_bpFaced')),
            p2_ace = int(request.form.get('p2_ace')),
            p2_df = int(request.form.get('p2_df')),
            #p2_svpt = int(request.form.get('p2_svpt')),
            p2_1stIn = int(request.form.get('p2_1stIn')),
            p2_1stWon = int(request.form.get('p2_1stWon')),
            p2_2ndWon = int(request.form.get('p2_2ndWon')),
            #p2_SvGms = int(request.form.get('p2_SvGms')),
            #p2_bpSaved = int(request.form.get('p2_bpSaved')),
            p2_bpFaced = int(request.form.get('p2_bpFaced')),
            #p1_rank = int(request.form.get('p1_rank')),
            p1_rank_points = int(request.form.get('p1_rank_points')),
            #p2_rank = int(request.form.get('p2_rank')),
            p2_rank_points = int(request.form.get('p2_rank_points'))
        )

        prediction_df = data.get_data_as_dataframe()
        # View pred_df in console
        #print(prediction_df)

        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(prediction_df)
        # 'prediction' is the predicted score given to the frontend in home.html
        prediction_percentage = predictions[0]
        if prediction_percentage > 50:
            interpretation = f'Player 2 is favored to win by {prediction_percentage - 50:.2f}%.'
        else:
            interpretation = f'Player 1 is favored to win by {50 - prediction_percentage:.2f}%.'
        return render_template('home.html', prediction=predictions[0], interpretation=interpretation)
    
if __name__ == "__main__":
    app.run()
