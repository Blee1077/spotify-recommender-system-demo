import os
import pickle
import spotipy
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from spotipy.oauth2 import SpotifyOAuth

MODEL_PATH = r"./model"

def load_pickle(fn: str):
    """Loads in a pickle file given its filepath
    
    Args:
        fn (str): Path of pickle file
    
    Returns:
        unpickled object
    """
    with open(fn, 'rb') as f:
        return pickle.load(f)


def get_artist_genres(response: dict) -> list:
    """Gets the genres of a track's artist.
    
    Args:
        response (dict): Response from Spotify search API of track name and artist
        
    Returns:
        proc_genre_list (list): List of booleans denoting genres associated with track artist
    """
    artist_urn = []
    for artist in response['tracks']['items'][0]['artists']:
        artist_urn.append(artist["id"])
        
    genre_list = []
    for artist in sp.artists(artist_urn)["artists"]:
        genre_list += artist['genres']
    genre_list = list(set(genre_list))
        
    proc_genre_list = []
    for genre in ['pop', 'rock', 'hip hop', 'indie', 'rap', 'alternative']:
        proc_genre_list.append(any([genre in g.lower() for g in genre_list]) * 1)
        
    return proc_genre_list


def query(name: str, artist: str):
    """Queries Spotify's search API for a track given the track name and artist.
    From this the track's audio features and artist's genres are obtained.
    
    Args:
        name (str): Name of track
        artist (str) Name of track artist
    
    Returns:
        feats (dict): Spotify audio features of the track 
        track_artist (str): String of track name and track artist
        genres (list): Genres associated with the track's artist
    """
    query_str = f"{name} {artist}"
    res = sp.search(q=query_str, type='track')
    feats = sp.audio_features(res['tracks']['items'][0]['id'])[0]
    track_artist = f"{res['tracks']['items'][0]['name']} - {res['tracks']['items'][0]['artists'][0]['name']}"
    genres = get_artist_genres(res)
    
    return feats, genres, track_artist


def proc_language(lang_box: str):
    """Processes the track language from a string to a list of booleans to use in KNN model.
    
    Args:
        lang_box (str): Language of track selected by user on Gradio UI
    
    Returns:
        lang_proc_lst (str): List of booleans representing track language
    """
    mapping = {
        'English': 0,
        'Korean': 1,
        'Japanese': 2,
        'Cantonese': 3,
    }
    
    lang_proc_lst = [0, 0, 0, 0]
    mapping_idx = mapping.get(lang_box, None)
    if mapping_idx is not None:
        lang_proc_lst[mapping[lang_box]] = 1
        
    return lang_proc_lst


def create_features(name: str, artist: str, lang_box: str):
    """Creates features to use in KNN model for obtaining most similar tracks in my library to the track given by user.
    
    Args:
        name (str): Name of track given by user
        artist (str) Name of track artist given by user
        lang_box (str): Language of track selected by user on Gradio UI
        
    Returns:
        (numpy.array): Processed Spotify audio features
        track_artist (str): String of track name and track artist
    """
    feats, proc_genres, track_artist = query(name, artist)
    proc_lang_list = proc_language(lang_box)
    
    all_feats = [
        feats['danceability'],
        feats['energy'],
        minmax_scaler_loudness.transform(np.array(feats['loudness']).reshape(1, -1))[0][0],
        feats['acousticness'],
        feats['valence'],
        minmax_scaler_tempo.transform(np.array(feats['tempo']).reshape(1, -1))[0][0],
    ]
    all_feats += proc_lang_list
    all_feats += proc_genres
    
    return np.array(all_feats).reshape(1, -1), track_artist


def plot(output_df: pd.DataFrame, features: np.ndarray, track_artist: str):
    """Creates a Plotly 2D UMAP embedding scatter of chosen track, nearest neighbours, and tracks from my library.
    
    Args:
        output_df (pd.DataFrame): DataFrame of most similar tracks to user's chosen track
        features (np.ndarray): Spotify audio features of user's chosen track
        track_artist (str): String of track name and artist
        
    Returns:
        (plotly.graph_objects.Figure): Figure object of plot
    """
    embedded_feat = reducer.transform(features)[0]
    selected_track, selected_artist = track_artist.split('-')
    fig = go.Figure()
    
    idx_mask = ~embedding_df.index.isin(output_df.index)
    track_mask = embedding_df['name_artist'] != track_artist
    filt_embedding_df = embedding_df.iloc[idx_mask].loc[track_mask]
    fig.add_trace(
        go.Scattergl(
            x=filt_embedding_df['dim_1'],
            y=filt_embedding_df['dim_2'],
            text=filt_embedding_df['name_artist'],
            customdata=filt_embedding_df[['name', 'artist']].values,
            hovertemplate=
                "<b>%{text}</b><br><br>" +
                "Name: %{customdata[0]}<br>" +
                "Artist: %{customdata[1]}<br>" +
                "<extra></extra>",
            mode='markers',
            marker=dict(
                size=4,
                color='grey'
            ),
            showlegend=False
        ),
    )
    
    nn_df = embedding_df.iloc[output_df.index]
    fig.add_trace(
        go.Scattergl(
            x=nn_df['dim_1'],
            y=nn_df['dim_2'],
            text=nn_df['name_artist'],
            customdata=nn_df[['name', 'artist']].values,
            hovertemplate=
                "<b>%{text}</b><br><br>" +
                "Name: %{customdata[0]}<br>" +
                "Artist: %{customdata[1]}<br>" +
                "<extra></extra>",
            mode='markers',
            marker=dict(
                size=7,
                color='blue'
            ),
            name='Nearest Neighbours' # For the legend
        ),
    )
    
    fig.add_trace(
        go.Scattergl(
            x=[embedded_feat[0]],
            y=[embedded_feat[1]],
            text=[track_artist],
            customdata=[[[selected_track.strip(),], [selected_artist.strip(),]]],
            hovertemplate=
                "<b>%{text}</b><br><br>" +
                "Name: %{customdata[0]}<br>" +
                "Artist: %{customdata[1]}<br>" +
                "<extra></extra>",
            mode='markers',
            marker=dict(
                size=10,
                color='red'
            ),
            name='Chosen track' # For the legend
        ),
    )
    
    fig.update_layout(
        xaxis=dict(
            range=[embedded_feat[0]-1, embedded_feat[0]+1]
        ),
        yaxis=dict(
            range=[embedded_feat[1]-1, embedded_feat[1]+1]
        ),
        legend={'traceorder': 'reversed'},
    )
    
    return fig


def predict(name: str, artist: str, lang_box: str, plot_box: str):
    """
    
    Args:
        name (str): Name of track given by user
        artist (str) Name of track artist given by user
        lang_box (str): Language of track selected by user on Gradio UI
        plot_box (str): Whether to plot figure of 2D UMAP embedding

    Returns:
        track_artist (str): String of track name and artist
        output_df (pd.DataFrame): DataFrame of most similar tracks to user's chosen track
        Update of {map} variable depending on value of {plot_box} given
    """
    features, track_artist = create_features(name, artist, lang_box)
    nn_dist, nn_idx = model.kneighbors(features)
    
    output_df = (
        embedding_df[['name', 'artist']]
        .iloc[nn_idx[0]]
        .assign(Distance=np.around(nn_dist[0], 3))
        .rename(columns={'name': 'Track Name', 'artist': 'Track Artist'})
    )
    output_df = (
        output_df
        .loc[(output_df['Track Name'] != track_artist.split('-')[0].strip()) &
            (output_df['Track Artist'] != track_artist.split('-')[1].strip())]
        .head(5)
        .assign(Rank=list(range(1, 6)))
        [['Rank', 'Track Name', 'Track Artist', 'Distance']]
    )
    
    if plot_box != 'Yes':
        return track_artist, output_df, gr.update(visible=False)
    else:
        fig = plot(output_df, features, track_artist)
        return track_artist, output_df, gr.update(value=fig, visible=True)


# Authenticate Spotipy instance
scope = "user-library-read,user-top-read,user-read-recently-played"
access_token = spotipy.CacheFileHandler(cache_path='.cache').get_cached_token()
cache_handler = spotipy.MemoryCacheHandler(token_info=access_token)
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, redirect_uri=os.getenv('SPOTIPY_REDIRECT_URI'), cache_handler=cache_handler))

# Load in preproc and model artifacts
minmax_scaler_tempo = load_pickle(f"{MODEL_PATH}/minmax_scaler_tempo.pkl")
minmax_scaler_loudness = load_pickle(f"{MODEL_PATH}/minmax_scaler_loudness.pkl")
model = load_pickle(f"{MODEL_PATH}/nearest_neighbours.pkl")
reducer = load_pickle(f"{MODEL_PATH}/umap_reducer.pkl")
embedding_df = pd.read_csv(f"{MODEL_PATH}/embeddings.csv")

with gr.Blocks(analytics_enabled=False) as demo:
    
    gr.Markdown(
        """
        # Spotify Recommender System Demo
        Provide a track name and its corresponding track artist and you'll receive 5 of the closest tracks from my Spotify library.
        For more information on how this was created, head over to [Philipp Schmid's blog article](https://www.philschmid.de/serverless-gradio).  
        You can specify just a track name or just an artist. In the former case, the first result from Spotify's search algorithm will be used.
        In the latter case, the artists' most popular track will be used.  
        Click on the examples at the bottom to get a quick start.  
        
        This demo is part of my blog series on leveraging Spotify audio features to create a simple recommender system using the K-nearest neighbour algorithm and deploying it using the severless computing paradigm.  
        Check out part 1 which covers the [exploratory data analysis here](https://solverism.com/posts/2022-10-21-spotify.html) and part 2 which covers creating this app (coming soon).
        
        Note: initial runs with and without the plot will take longer than subsequent runs, be prepared to wait for up to a minute for first time runs.  
        Tip: double click on the plot to zoom out and view all clusters.  
        """
    )
    
    with gr.Row():
        
        # Inputs column
        with gr.Column():
            name_box = gr.Textbox(label="Track Name", interactive=True)
            artist_box = gr.Textbox(label="Track Artist", interactive=True)
            lang_box = gr.Radio(choices=["English", "Japanese", "Korean", "Cantonese", "Other"], value="English", label="Language", interactive=True)
            plot_box = gr.Radio(choices=["Yes", "No"], value="No", label="Include output plot of most tracks?", interactive=True)
            submit_btn = gr.Button("Submit")
            
        # Output column of table
        with gr.Column():
            chosen_track_box = gr.Textbox(label="Selected Track", interactive=False)
            output_box = gr.components.Dataframe(type="pandas", interactive=False)

    # Additional output row for wide plot
    with gr.Row():
        map = gr.Plot(visible=False)
    
    # Submission button
    submit_btn.click(
        predict,
        [name_box, artist_box, lang_box, plot_box],
        [chosen_track_box, output_box, map],
    )
    
    # Examples for users to get started with
    gr.Examples(
        examples=[
            ["Call Me", "NAV", "English", "No"],
            ["Glimpse of Us", "Joji", "English", "Yes"],
            ["You outside my window", "きのこ帝国", "Japanese", "Yes"],
            ["INDUSTRY BABY", "Lil Nas X", "English", "No"],
        ],
        inputs=[name_box, artist_box, lang_box, plot_box],
        outputs=[chosen_track_box, output_box, map],
        fn=predict,
        cache_examples=False
    )

demo.launch(server_port=int(os.getenv('PORT')), enable_queue=False)
