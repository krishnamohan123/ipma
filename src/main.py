# -*- coding: utf-8 -*-
"""
Created on Fri May  7 08:05:48 2021

@author: iampr
"""

from folder_paths_linux import *
import streamlit as st

import base64
from io import BytesIO

st.set_page_config(layout="wide")

import ast

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

from utils import ( get_matching_risk_descriptions, 
                    get_ngrams_each_rd,
                    get_similar_riskphrases, 
                    get_updated_risk_db
                    )

def app(rd_selected):
    
    if rd_selected:

############################ LAYOUT PLAN #################################################

#### r00 : to display the selected or written testcase discription

#### rc01 : to display what domain expert said are the similar mathces

        rc00, _ ,rc01 = st.beta_columns((7,1,7))

#### rc10 : to display nearest riskphrases to the ngrams

        rc10 = st.beta_columns(1)[0]

#### rc20 : to display nearest risk-descriptions for the given risk-description based on risk-phrases predicted above

        rc20 = st.beta_columns(1)[0]

#### rc30 : to display the matching risk-phrases for any of the above recommended risk-descriptions

        rc30 = st.beta_columns(1)[0]

############################ ROW0, COLUMN0 ###############################################


        #rd_selected_pp = preprocess_text(rd_selected)

        #st.subheader("Extracted ngrams:")

        ngrams = get_ngrams_each_rd(rd_selected)

        with rc00:
            st.subheader("Selected Risk Description:")
            st.write(rd_selected)
            #st.subheader("N-grams:")
            #st.write(ngrams)
            #annotated_text("This ", ("is", "risk", "#f72d4f","#ffffff"))

############################ ROW0, COLUMN1 ###############################################

        with rc01:

            st.subheader("Similar Risks provided by Domain Expert:")

            df_hs = pd.read_excel(data_with_target, sheet_name = 'selected')

            df_hs = df_hs[['risk_description', 'human_selection']]

            df_hs.human_selection = df_hs.human_selection.apply(lambda x: ast.literal_eval(x))

            #st.write(fuzz.ratio(df_hs.risk_description[1], rd_selected))

            def func1(x):
                if fuzz.ratio(x, rd_selected) > 95:
                    return True
                else:
                    return False

            hs = df_hs[df_hs.risk_description.apply(func1)]['human_selection'].explode().reset_index(drop=True)

            #hs = df_hs.risk_description == rd_selected #['human_selection'].explode()

            st.table(hs)

############################ ROW1, COLUMN0 ###############################################

        rc10.subheader("Risk-phrases predicted by AI System:")


        risk_db = get_updated_risk_db() 

        risk_db = [x for x in risk_db if x] # to remove nulls if any

        tmp = get_similar_riskphrases(ngrams, risk_db)

        tmp_df = pd.DataFrame(tmp)


        #rc10.table(tmp_df)

        test_phrases = tmp_df['risk_phrase'].values
        
            
        with rc10:
                st.warning( "To mark the below as Non-Similar Check them: Do it only when appropriate !")
                df_similar = pd.DataFrame(tmp_df)
                df_similar.sort_values("avg_score",inplace = True,ascending=False)
                ngram_dict = dict(zip(df_similar.ngram, df_similar.risk_phrase))
        
                for a,b in ngram_dict.items():
                        if st.checkbox('ngram = ' + a +  ' :      : '   + 'Risk_phrase = ' + b , key=a):
                            ngram_list = a
                            risk_phrase = b
                            tmp1 = get_similar_riskphrases(ngrams, risk_db)
                            tmp_df1 = pd.DataFrame(tmp1)
                            tmp_df1['sim_match'] = 0
                            k_df = tmp_df1[(tmp_df1.ngram == ngram_list) & (tmp_df1.risk_phrase == risk_phrase)]
                            k_df.drop('avg_score',axis=1,inplace=True)
                            Master_df = pd.DataFrame()
                            # copy all these temporary records into a master file - # MASTER FILE
                            Master_df = Master_df.append(k_df)
                            Master_final = Master_df.drop_duplicates()

                            with open('/Users/sunithaadiraju/Desktop/ipma_ensemble/data/input/test_final.csv', 'a') as f:
                                    f.write('\n')  
                                    #Master_final.head(1).to_csv(f, header=False,mode = 'a',index=False)
                                    k_df.head(1).to_csv(f, header=False,mode = 'a',index=False)
                                    st.markdown(f'**_{ngram_list}, {risk_phrase} _** successfully added to Database as Non-Similar!')
                                    #Final_DB = pd.read('/Users/sunithaadiraju/Desktop/ipma_ensemble/data/input/test_final.csv',header = True)
                            #Final_DB.drop_duplicates() # Need this for Retraining and merging with df_combined of ensemble model 
                        


############################ ROW2, COLUMN0 ###############################################
        
        with rc20:
            
            st.subheader("Similar Risk-Descriptions provided by AI System:")

            st.markdown("Based on the Active Risk Phrases in the testcase, the closest Risk Descriptions are listed below. If the risk-descriptions are matching with what domain expert provided, The row will be highlighted in green background.")

            res_df = get_matching_risk_descriptions(test_phrases)

            nearest_rd_df = res_df[['matching_risk_phrases','n_matches', 'avg_score', 'risk_description','risk_mitigation']]

            nearest_rd_df = nearest_rd_df.reset_index(drop=True)

            def func2(x):
                status='No'
                for i in hs:
                    if fuzz.ratio(x, i) > 95:
                        status ='Yes'
                        break
                    else:
                        status = 'No'
                return status

            tmp = nearest_rd_df[~nearest_rd_df.risk_description.apply(func1)]
            
            tmp['status'] = tmp.risk_description.apply(func2)

            tmp = tmp.sort_values(by=['status','n_matches'], ascending=[False, False]).head(5)

            tmp1 = tmp[['status', 'risk_description', 'risk_mitigation', 'n_matches','avg_score']]

            tmp1_yes = tmp1[tmp1.status == 'Yes'].drop(['status'], axis =1)
            tmp1_no = tmp1[tmp1.status == 'No'].drop(['status'], axis =1)

            x = st.table(tmp1_yes.style.set_properties(**{"background-color": "lightgreen", "color": "black"}))
            
            x.add_rows(tmp1_no)


            if not hs.dropna().empty:

                hs_list = list(hs)

                hs_match = sum(nearest_rd_df.risk_description.isin(hs_list))

                total_list = hs_list + list(tmp[tmp['status'] == 'No']['risk_description'])
            else:
                total_list = list(tmp[tmp['status'] == 'No']['risk_description'])

############################ ROW3, COLUMN0 ###############################################

            with rc30:
                
                st.subheader("Matching Risk-phrases for the above Risk Descriptions:")

                rd = st.selectbox('Choose the risk description to get the risk-phrases from AI system:', total_list)

                st.write(rd)
                
                matches = res_df[res_df.risk_description == rd].sim_matches.values[0]

                matches = ast.literal_eval(matches)

                matches_df = pd.DataFrame(matches)

                if matches_df.size == 0:
                    st.markdown("---")
                    st.markdown("*No Risk Phrases Identified in the selected risk description*")

                else:

                    matches_df['match'] =  matches_df.risk_phrase.apply(lambda x: 'Yes' if x in test_phrases else 'No')

                    tmp2 = matches_df.sort_values(['match'], ascending=False).reset_index(drop=True)

                    tmp2_yes = tmp2[tmp2['match'] == 'Yes']
                    tmp2_yes = tmp2_yes.drop(['match'], axis=1)

                    ind = len(tmp2_yes)

                    x = st.table(tmp2_yes.style.set_properties(**{"background-color": "lightgreen", "color": "black"}))

                    tmp2_rem = tmp2[tmp2.match == 'No'].drop(['match'], axis=1)

                    if not tmp2_rem.empty:
                        x.add_rows(tmp2_rem)