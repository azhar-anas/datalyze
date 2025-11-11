import streamlit as st
import gc
from assets.styles.styler import apply_global_style
from utils.data_visualization import display_dataset, generate_eda_report, show_interactive_scatter_plot

# Page Style
apply_global_style()

# Page Header 
st.title(':material/bar_chart_4_bars: Exploratory Data Analysis')
st.write('**Exploratory Data Analysis (EDA)** in Datalyze runs automatically, providing a comprehensive report with just one click. Datalyze will generate **detailed summaries**, **visualizations**, and **statistical insights** to help you uncover patterns and relationships effortlessly. Additionally, you can also **download the EDA report** as an HTML file for further analysis.')

if 'current_dataset' not in st.session_state: # Ensure that the dataset has been uploaded
    st.warning(':material/warning: **No Dataset Found**. Please upload your dataset on the *Upload Dataset* page.')
else: # Main Code Start From Here
    
    # Display Current Dataset
    selected_df = st.session_state['current_dataset']['df_file'].copy()
    df_report = {
        'df_type': 'current_dataset',
        'report_status': st.session_state['current_dataset']['report_status'],
        'report_file': st.session_state['current_dataset']['report_file']
    }

    # Display selected dataset
    display_dataset(selected_df)
    
    # Display EDA Actions
    st.write('')
    tab1, tab2 = st.tabs(['Generated EDA Report', 'Interactive Scatter Plot'])

    with tab1:
        # Show Generated EDA Report
        if not df_report['report_status']:
            st.warning(':material/warning: **EDA Report Unavailable**. Your dataset appears to be new or recently modified. Click the button below to proceed.')
            if st.button(':material/description: Generate EDA Report', key='eda_report_btn', use_container_width=True):
                generate_eda_report(selected_df, df_report)
                gc.collect()
                st.rerun()
        elif df_report['report_status']:
                generate_eda_report(selected_df, df_report)
                gc.collect()
        else:
            st.error(':material/error: **Error in Generating EDA Report!** Please try again.')

    with tab2:
        # Show Scatter Plot
        show_interactive_scatter_plot(selected_df)
