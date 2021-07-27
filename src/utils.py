######## Import the required Libraries & Load the data ######

from folder_paths_linux import *
import pandas as pd
import streamlit as st
import ast

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import util

import joblib

mpnet_model = joblib.load(mpnet_file)
use_model = joblib.load(use_file)
glove6b_model = joblib.load(glove6b_file)
glove840b_model = joblib.load(glove840_file)
komninos_model = joblib.load(komninos_file)
paraphrase_mpnet_model = joblib.load(paraphrase_mpnet_file)
rf_model = joblib.load(rf_model_file)

df_train = pd.read_csv(trained_data)

def get_score_simulations(input1, input2, model):
  input1_emb = model.encode(input1)
  input2_emb = model.encode(input2)
  score = util.pytorch_cos_sim(input1_emb, input2_emb)
  return score

def get_simulation_scores(input1, input2):

  score2 = get_score_simulations(input1, input2, glove6b_model)
  score3 = get_score_simulations(input1, input2, glove840b_model)
  score4 = get_score_simulations(input1, input2, komninos_model)
  score5 = get_score_simulations(input1, input2, use_model)
  score6 = get_score_simulations(input1, input2, mpnet_model)
  score7 = get_score_simulations(input1, input2, paraphrase_mpnet_model)

  scores_df = pd.DataFrame()

  scores_df = scores_df.append({
      'ngram': input1,
      'riskphrase': input2,
      'glove6b':round(float(score2),3),            
      'glove840b':round(float(score3),3), 
      'komninos': round(float(score4),3),
      'universal_sentence_encoder':round(float(score5),3),
      'mpnet': round(float(score6),3), 
      'paraphrase_mpnet':round(float(score7),3)
  }, ignore_index=True)

  return scores_df[['ngram', 'riskphrase', 'glove6b', 'glove840b','komninos','universal_sentence_encoder', 'mpnet', 'paraphrase_mpnet']]


def get_updated_risk_db(add_rs_list = [], remove_rs_list = []):
    
    clusters_df = pd.read_csv(risks_metadata_file)

    clusters_df.dropna(inplace=True)
    shortlisted_risk_phrases =  list(clusters_df.risk_phrase)

    updated_risk_phrases = shortlisted_risk_phrases + add_rs_list
    
    return updated_risk_phrases


def load_data(file_path, sheet_name):

  # load the data
  df = pd.read_excel(file_path, sheet_name = sheet_name)

  ## convert column names to lower case
  df.columns = [i.strip().lower().replace(' ', '_') for i in df.columns]

  ## rename columns
  df.rename(columns= {'risk_category_-rewritten': 'risk_category'}, inplace=True)

  return df

def preprocess_text(rd):
  return rd.lower()

def get_ngrams_each_rd(x):

  tfidf = CountVectorizer(ngram_range=(1,5))

  tfidf.fit_transform(pd.Series(x))

  return tfidf.get_feature_names()


def get_sim_score(ngrams, risk_db, model, outputscore):

  encodings = model.encode(risk_db + ngrams)

  sim_df = pd.DataFrame(cosine_similarity(encodings))

  nrows = len(risk_db)

  sim_df = sim_df.iloc[nrows:,:nrows]

  sim_df['ngram'] = ngrams

  sim_df = sim_df.set_index('ngram')

  sim_df.columns  = risk_db

  sim_df = sim_df.stack()#.sort_values(ascending = False)

#  sim_df = sim_df[sim_df >= 0.6]

  sim_df = pd.DataFrame(sim_df).reset_index().rename({'level_1': 'risk_phrase', 
                                                          0: outputscore}, axis=1)

 # sim_df['ngram_length'] = sim_df.ngram.apply(lambda x: len(x.split()))
 # sim_df['riskword_length'] = sim_df.risk_phrase.apply(lambda x: len(x.split()))

  return sim_df


def get_similar_riskphrases(ngrams, risk_db):
  glove6b_df = get_sim_score(ngrams, risk_db, glove6b_model, 'glove6b')
  glove840b_df = get_sim_score(ngrams, risk_db, glove840b_model, 'glove840b')
  komninos_df = get_sim_score(ngrams, risk_db, komninos_model, 'komninos')
  use_df = get_sim_score(ngrams, risk_db, use_model, 'universal_sentence_encoder')
  mpnet_df = get_sim_score(ngrams, risk_db, mpnet_model, 'mpnet')
  paraphrase_mpnet_df = get_sim_score(ngrams, risk_db, paraphrase_mpnet_model, 'paraphrase_mpnet')

  res_df = glove6b_df.copy()
  del glove6b_df

  res_df['glove840b'] = glove840b_df.glove840b
  del glove840b_df

  res_df['komninos'] = komninos_df.komninos
  del komninos_df

  res_df['universal_sentence_encoder'] = use_df.universal_sentence_encoder
  del use_df

  res_df['mpnet'] = mpnet_df.mpnet
  del mpnet_df

  res_df['paraphrase_mpnet'] = paraphrase_mpnet_df.paraphrase_mpnet
  del paraphrase_mpnet_df

  ## filter for records if any of the models score is less than 70%

  res_df_70 = res_df[(res_df.glove6b>=0.7) | 
                     (res_df.glove840b>=0.7) | 
                     (res_df.komninos>=0.7) | 
                     (res_df.universal_sentence_encoder>=0.7) | 
                     (res_df.mpnet>=0.7) | 
                     (res_df.paraphrase_mpnet>=0.7)]

  cols = [#'ngram_length', 
        #'riskword_length',
        'universal_sentence_encoder',
        'glove6b', 
        'glove840b', 
        'komninos',
        'mpnet', 
        'paraphrase_mpnet']

  df1 = res_df_70[cols]

  if df1.empty:
    
    return
  
  else:
  
    res_df_70['sim_pred'] = rf_model.predict(df1)

    res_df_70_match = res_df_70[res_df_70.sim_pred ==1].drop('sim_pred', axis=1)

    res_df_70_filtered = res_df_70_match.sort_values(by = 'mpnet', ascending=False).drop_duplicates(subset=['risk_phrase'], keep='first')

    #res_df_70_filtered = res_df_70_filtered[res_df_70_filtered.mpnet >=0.45]

    res_df_70_filtered['avg_score'] = (res_df_70_filtered.glove6b + res_df_70_filtered.glove840b + res_df_70_filtered.komninos + res_df_70_filtered.universal_sentence_encoder + res_df_70_filtered.mpnet + res_df_70_filtered.paraphrase_mpnet)/6

    return res_df_70_filtered.reset_index(drop=True).to_dict()


def covert_to_riskscore(x):
  if x:
    tmp = pd.DataFrame(x)
    return tmp[['risk_phrase', 'avg_score']].to_dict()

def literal_evaluate(x):
  if x:
    return ast.literal_eval(x)

def myfunc(x):
  if x:
    tmp_df = pd.DataFrame(x)
    tmp_df = tmp_df[tmp_df.mpnet>=0.45]
    return tmp_df.to_dict()

def get_matching_phrases(train_phrases, test_phrases):
    tmp_df = pd.DataFrame(train_phrases)
    tmp_df_matches = tmp_df[tmp_df.risk_phrase.isin(test_phrases)]
    return tmp_df_matches.risk_phrase.values

def get_matching_phrases_score(train_phrases, test_phrases):
    tmp_df = pd.DataFrame(train_phrases)
    tmp_df_matches = tmp_df[tmp_df.risk_phrase.isin(test_phrases)]
    return tmp_df_matches.avg_score.sum()

def get_matching_risk_descriptions(test_phrases,df_train=df_train):
    df_train1 = df_train[~df_train.risk_phrases_score.isnull()]
    df_train1['risk_phrases_score'] =  df_train1.risk_phrases_score.apply(lambda x: ast.literal_eval(x))
    df_train1['matching_risk_phrases'] =  df_train1.risk_phrases_score.apply(lambda x: get_matching_phrases(x,test_phrases))
    df_train1['n_matches'] =  df_train1.matching_risk_phrases.apply(len)
    df_train1 = df_train1[df_train1.n_matches >= 1]
    df_train1['avg_score'] =df_train1.risk_phrases_score.apply(lambda x: get_matching_phrases_score(x,test_phrases))
    df_train1 = df_train1.sort_values(by = ['n_matches', 'avg_score'], ascending = [False, False])

    return df_train1