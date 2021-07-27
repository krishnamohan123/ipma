# -*- coding: utf-8 -*-
"""
Created on Fri May  7 08:05:48 2021

@author: iampr
"""
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from folder_paths_linux import *

# MPnet model
mpnet_model = SentenceTransformer('stsb-mpnet-base-v2')
joblib.dump(mpnet_model,mpnet_file)

# Universal Sentence Encoder
use_model = SentenceTransformer('distiluse-base-multilingual-cased')
joblib.dump(use_model,use_file)

# Glove6B Word Embeddings
glove6b_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
joblib.dump(glove6b_model,glove6b_file)

# Glove840B Word Embeddings
glove840b_model = SentenceTransformer('average_word_embeddings_glove.840B.300d')
joblib.dump(glove840b_model,glove840b_file)

# Komninos Word Embeddings
komninos_model = SentenceTransformer('average_word_embeddings_komninos')
joblib.dump(komninos_model, komninos_file)

# Paraphrase MPnet 
paraphrase_mpnet_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
joblib.dump(paraphrase_mpnet_model,paraphrase_mpnet_file)