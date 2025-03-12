import streamlit as st
from assets.styles.styler import apply_global_style
from utils.data_visualization import display_dataset, generate_eda_report

# Page Style
apply_global_style()

# Page Header 
st.title(':material/bar_chart_4_bars: Exploratory Data Analysis')
st.write('**Exploratory Data Analysis (EDA)** in Datalyze runs automatically, providing a comprehensive report with just one click. Simply select a dataset, and Datalyze will generate **detailed summaries**, **visualizations**, and **statistical insights** to help you uncover patterns and relationships effortlessly.')

if 'current_dataset' not in st.session_state: # Ensure that the dataset has been uploaded
    st.warning(':material/warning: **No dataset found**. Please upload a dataset on the Upload Dataset page first.')
else: # Main Code Start From Here
    
    # Select and view dataset
    dataset_choice = st.selectbox('**Select Dataset**', ['Current Dataset', 'Raw Dataset'])
    if dataset_choice == 'Current Dataset':
        selected_df = st.session_state['current_dataset']['df_file'].copy()
        df_report = {
            'df_type': 'current_dataset',
            'report_status': st.session_state['current_dataset']['report_status'],
            'report_file': st.session_state['current_dataset']['report_file']
        }
    else:
        selected_df = st.session_state['raw_dataset']['df_file'].copy()
        df_report = {
            'df_type': 'raw_dataset',
            'report_status': st.session_state['raw_dataset']['report_status'],
            'report_file': st.session_state['raw_dataset']['report_file']
        }
        
    display_dataset(selected_df)
    
    # Generate EDA Report
    st.write('---')
    st.subheader(':material/description: **Generated EDA Report**')
    generate_eda_report(selected_df, df_report)
    del selected_df, df_report, generate_eda_report