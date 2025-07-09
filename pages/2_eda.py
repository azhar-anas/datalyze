import streamlit as st
import gc
from streamlit_pandas_profiling import st_profile_report
from assets.styles.styler import apply_global_style
from utils.data_visualization import display_dataset, generate_eda_report, download_eda_report_button

# Page Style
apply_global_style()

# Page Header 
st.title(':material/bar_chart_4_bars: Exploratory Data Analysis')
st.write('**Exploratory Data Analysis (EDA)** in Datalyze runs automatically, providing a comprehensive report with just one click. Simply select a dataset, and Datalyze will generate **detailed summaries**, **visualizations**, and **statistical insights** to help you uncover patterns and relationships effortlessly. Additionally, you can also **download the EDA report** as an HTML file for further analysis.')

if 'current_dataset' not in st.session_state: # Ensure that the dataset has been uploaded
    st.warning(':material/warning: **No Dataset Found**. Please upload your dataset on the *Upload Dataset* page.')
else: # Main Code Start From Here
    
    # Select dataset to generate EDA report
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
    
    # Display selected dataset
    display_dataset(selected_df)
    
    # Display EDA Actions
    st.write('')
    tab1, tab2 = st.tabs(['Generated EDA Report', 'Interactive Scatter Plot'])

    with tab1:
        # Show Generated EDA Report
        if not df_report['report_status']:
            st.warning(':material/warning: **EDA Report Not Generated because Your Dataset is New or has been Modified**. Click the button below to generate the EDA report.')
            if st.button(':material/description: Generate EDA Report', use_container_width=True):
                generate_eda_report(selected_df, df_report)
                gc.collect()
                # Update the session state after generating the report
                if dataset_choice == 'Current Dataset':
                    st.session_state['current_dataset']['report_status'] = True
                else:
                    st.session_state['raw_dataset']['report_status'] = True
                st.rerun()
        elif df_report['report_status']:
            # Display the existing report
            if dataset_choice == 'Current Dataset':
                report_file = st.session_state['current_dataset']['report_file']
            else:
                report_file = st.session_state['raw_dataset']['report_file']
            
            if report_file:
                st_profile_report(report=report_file)
                download_eda_report_button(report_file, df_report['df_type'])
                gc.collect()
        else:
            st.error(':material/error: **Error in Generating EDA Report!** Please try again.')

    with tab2:
        # Show Scatter Plot
        numeric_cols = selected_df.select_dtypes(include='number').columns
        if len(numeric_cols) < 2:
            st.warning(':material/warning: **Not Enough Numeric Columns**. Please select a dataset with at least two numeric columns for scatter plot visualization.')
            x_col, y_col = None, None
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                x_col = st.selectbox('**Select X-axis Column**', numeric_cols)
                y_col = st.selectbox('**Select Y-axis Column**', numeric_cols)

            with col2:
                # Filter columns: non-numeric and <= 10 unique values
                color_candidates = [
                    col for col in selected_df.columns
                    if selected_df[col].nunique() < 10
                ]
                color_col = st.selectbox('**Select Color Column (Optional)**', ['None'] + color_candidates)
        
        if x_col and y_col:
            st.write('\n')
            if x_col == y_col:
                st.warning(':material/warning: **X-axis and Y-axis columns must be different.** Please select two different columns.')
            else:
                st.markdown(
                    f"<div style='text-align: center;'><b>Scatter Plot of {x_col} vs {y_col}</b></div>",
                    unsafe_allow_html=True
                )
                if color_col != 'None':
                    st.scatter_chart(data=selected_df[[x_col, y_col, color_col]], x=x_col, y=y_col, color=color_col, size=70, height=400)
                    gc.collect()
                else:
                    st.scatter_chart(data=selected_df[[x_col, y_col]], x=x_col, y=y_col, size=70, height=400)
                    gc.collect()
