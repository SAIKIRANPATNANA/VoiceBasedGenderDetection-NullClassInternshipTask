import streamlit as st
import tensorflow as tf
import numpy as np 
import pandas as pd 
import librosa 
from st_audiorec import st_audiorec

def predict(audio_file):
    model = tf.keras.models.load_model('voice_clzfier.hdf5')
    audio, sample_rate = librosa.load(audio_file) 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    features = np.array(mfccs_scaled_features)
    features = np.expand_dims(features, axis=0)
    return int(np.round(model.predict(features)))

st.title("VOICE BASED GENDER PREDICTION")
st.header('Trained & Developed by Sai Kiran Patnana')

uploaded_file = None 
recorded_file = None

if st.toggle('Upload Audio'):
    uploaded_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'ogg'])
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        st.success("Audio uploaded successfully!")
        if st.button('Predict Gender'):
            if predict(uploaded_file)==1:
                st.success("Your uploaded audio is of a Male.")
            else:
                st.success("Your uploaded audio is of a Female.")

elif st.toggle('Record Audio'):
    recorded_file = st_audiorec()
    if st.button('Predict Gender') and recorded_file is not None:
        with open("tested_recordings/recorded_audio.wav", "wb") as f:
            f.write(recorded_file)
        st.success("Audio recorded successfully!")
        recorded_file = 'tested_recordings/recorded_audio.wav'
        if predict(recorded_file)==1:
            st.success("Your recorded audio is of a Male.")
        else:
            st.success("Your recorded audio is of a Female.")

# model size: 0.5 mb
