import streamlit as st
import gc
from assets.styles.styler import apply_global_style
from utils.data_visualization import display_dataset
from utils.data_quality_check import find_missing_values, find_duplicate_values, find_outliers

# Page Style
apply_global_style()

# Page Header 
st.title(':material/mop: Data Cleaning')
st.write('Data cleaning is a crucial step in data preparation, ensuring that your dataset is accurate, consistent, and ready for analysis. Datalyze provides an intuitive interface to handle common data quality issues efficiently.')

if 'current_dataset' not in st.session_state: # Ensure that the dataset has been uploaded
    st.warning(':material/warning: **No dataset found**. Please upload a dataset on the Upload Dataset page first.')
else: # Main Code Start From Here
    df = st.session_state['current_dataset']['df_file'].copy()
    tab1, tab2, tab3, tab4= st.tabs(['Change Data Type', 'Handle Missing Values', 'Remove Duplicate Rows', 'Remove Outliers'])
    
    # 1. Change Data Type
    with tab1:
        st.write('This section allows you to **modify the data types** of selected features to match the appropriate format. You can convert data into **Boolean, Integer (int64), Float (float64), or Object** types. This ensures that your dataset is structured correctly for further analysis and modeling.')
        
        features_to_change = st.multiselect('**Select features to change data type**', df.columns)
        new_dtype = st.selectbox('**Select new data type**', options=['boolean', 'int64', 'float64', 'object'])
        if st.button(label='Change Data Type', icon=':material/mop:'): # Change Data Type Button
            if not features_to_change:
                st.error(':material/error: **No features selected**. Please select at least one feature to change data type.')
            else:
                df_processed = df.copy()
                success = True
                for column in features_to_change:
                    try:
                        if new_dtype == 'boolean' and df_processed[column].dtype == 'object':
                            unique_values = df_processed[column].dropna().unique()
                            if len(unique_values) == 2:
                                mapping = {unique_values[0]: 0, unique_values[1]: 1}
                                df_processed[column] = df_processed[column].map(mapping)
                                df_processed[column] = df_processed[column].astype('boolean')
                            else:
                                raise ValueError(f'Feature object \'{column}\' does not have exactly 2 unique values.')
                        else:
                            df_processed[column] = df_processed[column].astype(new_dtype)
                    except Exception as e:
                        st.error(f':material/error: **Data type cannot be changed for feature** \'{column}\'. {e}')
                        success = False
                gc.collect()
                if success:
                    st.session_state['current_dataset']['df_file'] = df_processed
                    st.session_state['current_dataset']['report_status'] = False
                    st.session_state['current_dataset']['report_file'] = None
                    del df_processed, success
                    gc.collect()
                    st.rerun()
        
        st.write('')
        st.subheader(':material/table: Current Dataset')
        display_dataset(df)
    
    # 2. Handle Missing Values
    with tab2:
        st.write('This section helps you **detect & manage incomplete data** by providing three methods: **Delete** -> Remove rows containing missing values; **Fill with Mean Value** -> Replace missing values with the mean (for integer & float types); And **Fill with Most Frequent Value** -> Replace missing values with the most frequently occurring value, applicable to **integer & float, Boolean only, or all data types**.')
        
        null_data, total_missing, percent_missing = find_missing_values(df)
        st.write(f'**Number of missing value: {total_missing}/{df.shape[0]} ({percent_missing:.2f}%)**')
        st.dataframe(null_data, use_container_width=True)
        if not null_data.empty:
            solution_method = st.selectbox('**Select Solution Method**', ['Delete', 'Fill with Mean Value (integer & float data type)', 'Fill with Most Frequent Value'])
            if solution_method == 'Fill with Most Frequent Value':
                data_type_option = st.radio('**Select Data Type to Apply**', ['Integer & Float', 'Boolean', 'All of above'])
            if st.button(label='Handle Missing Values', icon=':material/mop:'): # Handle Missing Value Button
                df_processed = df.copy()
                if solution_method == 'Delete':
                    df_processed = df_processed.dropna()
                elif solution_method == 'Fill with Mean Value (integer & float data type)':
                    df_processed = df_processed.apply(lambda x: x.fillna(x.mean()) if x.dtype in ['int64', 'float64'] else x, axis=0)
                elif solution_method == 'Fill with Most Frequent Value':
                    if data_type_option == 'Integer & Float':
                        df_processed = df_processed.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype in ['int64', 'float64'] else x, axis=0)
                    elif data_type_option == 'Boolean':
                        df_processed = df_processed.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'boolean' or x.dtype == 'bool' else x, axis=0)
                    elif data_type_option == 'All of above':
                        df_processed = df_processed.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype in ['int64', 'float64', 'boolean', 'bool'] else x, axis=0)
                df_processed.reset_index(drop=True, inplace=True)
                st.session_state['current_dataset']['df_file'] = df_processed
                st.session_state['current_dataset']['report_status'] = False
                st.session_state['current_dataset']['report_file'] = None
                del df_processed
                gc.collect()
                st.rerun()
        else:
            st.success(':material/task_alt: No missing value found')
        
        st.write('')
        st.subheader(':material/table: Current Dataset')
        display_dataset(df)
    
    # 3. Remove Duplicate Data
    with tab3:
        st.write('This section helps you **detect & eliminate duplicate rows** that may distort analysis. The system automatically identifies duplicate entries and you can remove them with a single click.')
        
        duplicates, num_duplicates, percent_duplicates = find_duplicate_values(df)
        st.write(f'**Number of duplicate rows: {num_duplicates}/{df.shape[0]} ({percent_duplicates:.2f}%)**')
        st.dataframe(duplicates, use_container_width=True)
        if not duplicates.empty:
            if st.button(label='Delete Duplicate rows', icon=':material/mop:'):
                df_processed = df.copy()
                df_processed = df_processed.drop_duplicates()
                df_processed.reset_index(drop=True, inplace=True)
                st.session_state['current_dataset']['df_file'] = df_processed
                st.session_state['current_dataset']['report_status'] = False
                st.session_state['current_dataset']['report_file'] = None
                del df_processed
                gc.collect()
                st.rerun()
        else:
            st.success(':material/task_alt: No duplicate row found')
        
        st.write('')
        st.subheader(':material/table: Current Dataset')
        display_dataset(df)
    
    # 4. Remove Outliers
    with tab4:
        st.write('This section enables you to **detect and eliminate extreme values (outliers)** that could skew analysis results. This section provides a **boxplot visualization** to help you identify potential outliers in numerical data. You can review the detected outliers and choose whether to remove them for a cleaner dataset.')
        
        outliers, outlier_plot, num_outliers, percent_outliers = find_outliers(df)
        st.write(f'**Number of outliers: {num_outliers}/{df.shape[0]} ({percent_outliers:.2f}%)**')
        if outlier_plot and not outliers.empty:
            st.pyplot(outlier_plot)
            st.write('')
        st.dataframe(outliers, use_container_width=True)
        
        if st.session_state['outliers_removed'] is True or outliers.empty:
            st.success(':material/task_alt: No outlier found')
        else:
            if st.button(label='Delete Outliers', icon=':material/mop:'):
                df_original = df.copy()
                df_numeric = df.select_dtypes(include=['number'])
                Q1 = df_numeric.quantile(0.25)
                Q3 = df_numeric.quantile(0.75)
                IQR = Q3 - Q1
                df_numeric_processed = df_numeric[~((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)]
                df_processed = df_original.loc[df_numeric_processed.index]
                df_processed.reset_index(drop=True, inplace=True)
                st.session_state['current_dataset']['df_file'] = df_processed
                st.session_state['current_dataset']['report_status'] = False
                st.session_state['current_dataset']['report_file'] = None
                st.session_state['outliers_removed'] = True
                del df_original, df_numeric, Q1, Q3, IQR, df_numeric_processed, df_processed
                gc.collect()
                st.rerun()
        
        st.write('')
        st.subheader(':material/table: Current Dataset')
        display_dataset(df)
