#app.py
from folder_paths_linux import *
import main
import simulations
import add_risk
from utils import load_data

import streamlit as st

st.sidebar.title('IPMA')

Options = {
    "Select the Risk Description from the list of testcases": main,
    "Write a new Risk Description": main,
    "Risk Metadata Management": add_risk,
    "Risk-phrase Simulations": simulations,
}

choice = st.sidebar.radio("Your Choice:", list(Options.keys()))

if choice ==  "Select the Risk Description from the list of testcases":

    df_test = load_data(test_file, test_file_sheet)
    test_rd1 = st.sidebar.selectbox('Choose from the list:', df_test.risk_description)
    page = Options[choice]
    page.app(test_rd1)

elif choice ==  "Write a new Risk Description":
    
    test_rd2 = st.sidebar.text_input('Manually Write/Paste a new Risk Description here:')
    page = Options[choice]
    page.app(test_rd2)

elif choice == "Risk-phrase Simulations":
    page = Options[choice]
    page.app()

elif choice == "Risk Metadata Management":
    page = Options[choice]
    page.app()