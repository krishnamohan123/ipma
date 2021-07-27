# -*- coding: utf-8 -*-
"""
Created on Fri May  7 08:05:48 2021

@author: iampr
"""

# Defining the project home directory
#PROJECT_HOME = "/workspace/ipma/"
PROJECT_HOME = "D:/Krishna/IPMA/v4/ipma_ensemble/"

# Defining the folder paths for data, models, simulations, results directories
INPUT_DATA_PATH = PROJECT_HOME + 'data/input/'
MODEL_PATH = PROJECT_HOME + 'data/saved_models/windows/'
SIMULATIONS_PATH = PROJECT_HOME + 'data/simulations/'
RESULTS_PATH = PROJECT_HOME + 'results/'
TRAINED_DATA_PATH = PROJECT_HOME + 'data/data_after_training/'
RISK_METADATA_PATH = PROJECT_HOME + 'data/risk_metadata/'

# Defining the file paths

# to save the simulation results
simulations_file = SIMULATIONS_PATH + 'simulations_history.csv'


# to save & load the model file
mpnet_file = MODEL_PATH + 'mpnet_model.dump' 
glove6b_file = MODEL_PATH + 'glove6b_model.dump'
glove840_file = MODEL_PATH + 'glove840b_model.dump'
komninos_file = MODEL_PATH + 'komninos_model.dump'
use_file = MODEL_PATH + 'use_model.dump'
paraphrase_mpnet_file = MODEL_PATH + 'paraphrase_mpnet_model.dump'

rf_model_file = MODEL_PATH + 'random_forest_final_model.pkl'
#rf_model_file = MODEL_PATH + 'random_forest_final_model_sunitha.pkl'
#rf_model_file = MODEL_PATH + 'random_forest_final_model_sunitha_v2.pkl'



# Inputs Provided by Mahesh from CGI
data_with_target = INPUT_DATA_PATH + 'final_input_with_target.xlsx' 

# input file that is being used for testing
test_file = INPUT_DATA_PATH + 'IPMA_new_risks.xlsx'
test_file_sheet = 'Top 20 Risks'

# input file to be used only for training, not required for at the time of predictions
train_file = INPUT_DATA_PATH + 'IPMA_new_risks.xlsx'
train_file_sheet = 'All RIsks'

# trained data is the file that should contian the train data along with risk phrases
trained_data = TRAINED_DATA_PATH + 'df_train_08072021.csv' 
#trained_data = TRAINED_DATA_PATH + 'df_train_25072021.csv' 
# latest metadata file that is used for both training and testing
# in order to add more risk-phrases in batch, open this file in excel and manually add the rows each containing a risk-phrase
risks_metadata_file = RISK_METADATA_PATH + 'updated_risk_phrases.csv'