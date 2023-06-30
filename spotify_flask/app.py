from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import pandas as pd
from spotify import get_track_info, get_token, headers, client_id, client_secret



app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        print(request.form)
        # Get the values from the form
        danceability = float(request.form['danceability'])
        energy = float(request.form['energy'])
        key = int(request.form['key'])
        loudness = float(request.form['loudness'])
        mode = int(request.form['mode'])
        speechiness = float(request.form['speechiness'])
        acousticness = float(request.form['acousticness'])
        instrumentalness = float(request.form['instrumentalness'])
        liveness = float(request.form['liveness'])
        valence = float(request.form['valence'])
        tempo = float(request.form['tempo'])
        duration_ms = int(request.form['duration_ms'])
        
        # Load the saved model, scaler, and encoder
        with open('static/utils/model.pkl', 'rb') as file:
            model = pickle.load(file)

        with open('static/utils/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        with open('static/utils/encoder.pkl', 'rb') as file:
            le = pickle.load(file)

        # Create a function to predict the genre of a song
        def predict_genre(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms):   
            x = [[danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, 0]]  # Add 0 for the missing feature (mode)
            x = scaler.transform(x)
            genre = model.predict(x)
            genre = le.inverse_transform(genre)
            return genre[0]

        
        
        
        # Predict the genre of the song
        genre = predict_genre(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms)
        print(genre)
        
        df = pd.read_csv('static/utils/genres_v2.csv')
        # Get all songs with the predicted genre
        df = df[df['genre'] == genre]
        # Get 10 random songs
        df = df.sample(10)
        print(df['id'])
        image_list = df['id'].tolist()


        # Get token
        token = get_token(client_id, client_secret)

        # Get track info
        track_images = {}
        for track_id in image_list:
            track_info = get_track_info(headers, track_id)
            # print(track_info)
            # Save the track title and image url
            track_images[track_info['name']] = [track_info['album']['images'][0]['url'], track_info['artists'][0]['name']]
        

        return render_template('results.html', genre=genre, image_list=image_list, track_images=track_images)
    return render_template('prediction.html')

@app.route('/results', methods=["GET", "POST"])
def results():
    return render_template('results.html')

if __name__ == '__main__':
    # Run on host 6000
    app.run(port=5500, debug=True)






