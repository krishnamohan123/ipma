# -*- coding: utf-8 -*-
"""
Created on Fri May  7 08:05:48 2021

@author: iampr
"""

### WARNING: RUNNING THIS SCRIPT TAKES LONGER AS NGRAMS TO RISK-PHRASE EMBEDDINGS TO BE COMPUTED FOR EACH OF THE TRAIN RISK DESCRIPTIONS

from folder_paths_linux import *
import numpy as np
import pandas as pd

import ast
import time

from sklearn.metrics.pairwise import cosine_similarity

from utils import ( get_similar_riskphrases, load_data,
                    preprocess_text,
                    get_ngrams_each_rd, 
                    get_updated_risk_db
                    )

def covert_to_riskscore(x):
  if x:
    tmp = pd.DataFrame(x)
    return tmp[['risk_phrase', 'avg_score']].to_dict()

def literal_evaluate(x):
  if x:
    return ast.literal_eval(x)

df_train = load_data(train_file, train_file_sheet)

df_train['phrase'] = df_train.risk_description.apply(preprocess_text)

df_train['ngrams'] = df_train.phrase.apply(get_ngrams_each_rd)

risk_db = get_updated_risk_db()

risk_lists = [x for x in risk_db if x] # removing empty risks if any

df_train['sim_matches'] = df_train['ngrams'].apply(lambda x: get_similar_riskphrases(x, risk_db)) # This step takes longer 

df_train['sim_matches'].dropna(inplace=True)

# df_train['sim_matches'] = df_train.sim_matches.apply(literal_evaluate)

df_train['risk_phrases_score'] = df_train.sim_matches.apply(covert_to_riskscore)

timestr = time.strftime("%Y%m%d_%H%M%S")
filename = TRAINED_DATA_PATH + 'df_train_' + timestr + '.csv'

df_train.to_csv(filename, index=False)