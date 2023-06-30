import requests
# Set up Spotify API credentials
from dotenv import load_dotenv
import os
load_dotenv()
client_id = os.getenv('client_id') 
client_secret = os.getenv('client_secret')
print(client_id)

# Get an access token
def get_token(client_id, client_secret):
    url = 'https://accounts.spotify.com/api/token'
    params = {'grant_type': 'client_credentials'}
    response = requests.post(url, auth=(client_id, client_secret), data=params)
    token = response.json()['access_token']
    return token

token = get_token(client_id, client_secret)
print(token)

# Set the authorization header with the access token
headers = {
    "Authorization": f"Bearer {token}"
}

# Make a GET request to the track endpoint
def get_track_info(headers, track_id):
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    response = requests.get(url, headers=headers)
    return response.json()

