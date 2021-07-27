
from sklearn.utils.extmath import stable_cumsum
from utils import *
from folder_paths_linux import *

def app():
           
############################ ROW2, COLUMN0 ###############################################

        st.subheader('Risk Metadata Management')

        add_rs = st.text_input('Add Risk Phrases copied from Risk Description here (Only one risk at a time):')

        if add_rs:

            add_rs_list = add_rs.split(',')

            risk_db = get_updated_risk_db()

            risk_db = [x for x in risk_db if x]

            tmp = get_similar_riskphrases(add_rs_list, risk_db)

            res_df = pd.DataFrame(tmp)

            test_phrases = res_df['risk_phrase'].values

            if res_df.empty:
                
                if add_rs_list[0]!='':
                    st.write('No Similar Risk Phrases present in Risk Database.')

                    try:
                        option = st.radio(f'Do you wish to add the risk : {add_rs_list[0]} ?',['no','yes'])
                        
                        if option == 'yes':
                            risk_db = get_updated_risk_db(add_rs_list)
                            risk_db = [x for x in risk_db if x]
                            risk_db = pd.DataFrame(risk_db, columns=['risk_phrase'])
                            risk_db.to_csv(risks_metadata_file, index=False)
                            st.markdown(f'**_{add_rs_list[0]}_** successfully added!')

                    except:
                        st.write('Unable to add Risk to the Database.')
            else:
            
                st.markdown(f'The Following are closest risk-phrases found in Risk Database!')
                st.markdown(f'WARNING: If you check the box, the risk-phrase will be deleted from the database. Do only when appropriate.')

                k = res_df.risk_phrase.values

                remove_rs_list=[]

                for i, c in enumerate(k):
                    if st.checkbox(c,key=i):
                        remove_rs_list.append(c)
                
                try:
                    risk_db = get_updated_risk_db(remove_rs_list=remove_rs_list)

                    risk_db = [x for x in risk_db if x]

                    risk_db = pd.DataFrame(risk_db, columns=['risk_phrase'])

                    risk_db.to_csv(risks_metadata_file, index=False)

                    if len(remove_rs_list)>0:
                        st.markdown(f'**_{remove_rs_list[0]}_** successfully removed!')
                except:
                    st.markdown(f'Unable to remove **_{add_rs_list[0]}_**')
                
                option = st.radio(f'Do you wish to add the risk : {add_rs_list[0]} ?',['no','yes'])

                if option == 'yes':

                        risk_db = get_updated_risk_db(add_rs_list)

                        risk_db = [x for x in risk_db if x]

                        risk_db = pd.DataFrame(risk_db, columns=['risk_phrase'])

                        st.write()

                        risk_db.to_csv(risks_metadata_file, index=False)

                        st.markdown(f'**_{add_rs_list[0]}_** successfully added!')
