import joblib
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource
def load_joblib_model(model_path):
    """Loads and returns the pre-trained joblib model."""
    return joblib.load(model_path)

@st.cache_resource
def load_encoder(encoder_path='./model/encoder.joblib'):
    """Loads and returns the pre-trained categorical encoder."""
    return joblib.load(encoder_path)

def preprocess_data(df, target_col):
    """
    Separates features and targets, then applies custom preprocessing.    
    """
    # Separate features and target
    X_raw = df.drop(columns=[target_col])
    y_true = df[target_col]

    # Load the ordinal encoder
    encoder = load_encoder('./model/encoder.joblib')

    # Transform the categorical columns
    X_processed = encoder.transform(X_raw)

    target_map = {'e': 0, 'p': 1}
    y_true = y_true.map(target_map).astype(int)   

    return X_processed, y_true