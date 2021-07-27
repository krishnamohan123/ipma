# -*- coding: utf-8 -*-
"""
Created on Fri May  7 08:05:48 2021

@author: iampr
"""

from folder_paths_linux import *
import streamlit as st
import joblib

import torch

from utils import *

import pandas as pd
import numpy as np

def app():

    cols = st.beta_columns(3)
    input1 = cols[0].text_input('input1')
    input2 = cols[1].text_input('input2')

    scores_df = get_simulation_scores(input1, input2)

    if not scores_df.empty:

        cols = ['glove6b', 'glove840b','komninos','universal_sentence_encoder', 'mpnet', 'paraphrase_mpnet']
        
        scores_df['model_predictions'] = rf_model.predict(scores_df[cols])

        scores_df['model_predictions'] = scores_df.model_predictions.map({1: 'similar', 0: 'not similar'})

        st.table(scores_df)


