import streamlit as st
import pandas as pd
from assets.styles.styler import apply_global_style
from utils.data_visualization import display_dataset

# Page Style
apply_global_style()

# Page Header 
st.title(':material/handyman: Feature Engineering')
st.write('Feature Engineering is a key step in the data preparation process, where raw data is transformed into meaningful features that improve model performance. This page provides tools to allowing you to optimize your dataset for machine learning tasks.')

if 'current_dataset' not in st.session_state: # Ensure that the dataset has been uploaded
    st.warning(':material/warning: **No dataset found**. Please upload a dataset on the Upload Dataset page first.')
else: # Main Code Start From Here
    df = st.session_state['current_dataset']['df_file'].copy()
    tab1, tab2, tab3, tab4 = st.tabs(['Rename Feature', 'Add Feature', 'Remove Feature', 'One-Hot Encoding'])
            
    # 1. Rename Feature
    with tab1:
        st.write('This section allows you to **rename specific features** to improve readability and organization. Simply select a feature and enter a new name to update it.')
        
        feature_to_rename = st.selectbox('**Select feature to rename**', df.columns)
        new_feature_name = st.text_input('**Enter the new name for the feature**', placeholder='New feature name')
        if st.button(label='Rename Feature', icon=':material/handyman:'):
            if not new_feature_name:
                st.error(':material/error: **No new name entered**. Please enter a new name for the selected feature.')
            else:
                df_processed = df.copy()
                df_processed.rename(columns={feature_to_rename: new_feature_name}, inplace=True)
                st.session_state['current_dataset']['df_file'] = df_processed
                st.session_state['current_dataset']['report_status'] = False
                st.session_state['current_dataset']['report_file'] = None
                del df_processed
                st.rerun()

        st.write('---')
        st.subheader(':material/table: Current Dataset')
        display_dataset(df)
    
                
    # 2. Add Feature
    with tab2:
        st.write('This Section enables you to **create new features using existing ones**. You can apply **Basic Mathematical Operations** (addition, subtraction, multiplication, or division) between two selected features or generate **Polynomial Features** by raising a selected feature to a specified degree (ranging from 2 to 10). These transformations help uncover new patterns and enhance model performance.')
        
        if st.session_state['current_dataset']['df_file'].isnull().sum().sum() > 0: # Ensure that the dataset does not contain missing values
            st.warning(':material/warning: **Your \'Current Dataset\' contains missing values**. Please handle them on the **Data Cleaning** page first.')
        else:
            operation_type = st.selectbox('**Select operation type**', ['Basic Mathematical Operation', 'Polynomial'])
            new_feature_name = st.text_input('**Enter the name for the new feature**', placeholder='New feature name')
            
            if operation_type == 'Basic Mathematical Operation':
                features = st.multiselect('**Select two numerical features**', df.columns)
                operation = st.selectbox('Select operation', ['Addition', 'Subtraction', 'Multiplication', 'Division'])
                if st.button(label='Add Feature', icon=':material/handyman:'):
                    if not new_feature_name:
                        st.error(':material/error: **No new feature name entered**. Please enter a name for the new feature.')
                    elif len(features) != 2:
                        st.error(':material/error: **Please select two numerical features**.')
                    elif not (pd.api.types.is_numeric_dtype(df[features[0]]) and pd.api.types.is_numeric_dtype(df[features[1]]) and not pd.api.types.is_bool_dtype(df[features[0]]) and not pd.api.types.is_bool_dtype(df[features[1]])):
                        st.error(':material/error: **Selected numerical features must be numeric (integer or float)**.')
                    else:
                        df_processed = df.copy()
                        new_feature = None
                        if operation == 'Addition':
                            new_feature = df_processed[features[0]] + df_processed[features[1]]
                        elif operation == 'Subtraction':
                            new_feature = df_processed[features[0]] - df_processed[features[1]]
                        elif operation == 'Multiplication':
                            new_feature = df_processed[features[0]] * df_processed[features[1]]
                        elif operation == 'Division':
                            new_feature = df_processed[features[0]] / df_processed[features[1]]
                        df_processed[new_feature_name] = new_feature
                        st.session_state['current_dataset']['df_file'] = df_processed
                        st.session_state['current_dataset']['report_status'] = False
                        st.session_state['current_dataset']['report_file'] = None
                        del df_processed
                        st.rerun()
            
            elif operation_type == 'Polynomial':
                feature = st.selectbox('**Select a numerical feature**', df.columns)
                degree = st.number_input('**Select polynomial degree**', min_value=2, max_value=10, value=2, step=1)
                if st.button(label='Add Feature', icon=':material/handyman:'):
                    if not new_feature_name:
                        st.error(':material/error: **No new feature name entered**. Please enter a name for the new feature.')
                    elif not (pd.api.types.is_numeric_dtype(df[feature]) and not pd.api.types.is_bool_dtype(df[feature])):
                        st.error(':material/error: **Selected numerical features must be numeric (integer or float)**.')
                    else:
                        df_processed = df.copy()
                        new_feature = df_processed[feature] ** degree
                        df_processed[new_feature_name] = new_feature
                        st.session_state['current_dataset']['df_file'] = df_processed
                        st.session_state['current_dataset']['report_status'] = False
                        st.session_state['current_dataset']['report_file'] = None
                        del df_processed
                        st.rerun()
        
        st.write('---')
        st.subheader(':material/table: Current Dataset')
        display_dataset(df)
            
    # 3. Remove Feature
    with tab3:
        st.write('This section allows you to **eliminate unnecessary features** from the dataset, ensuring that only relevant information is retained for analysis. Simply select the features you want to remove and confirm the action.')
        
        features_to_remove = st.multiselect('**Select features to remove**', df.columns)
        if st.button(label='Remove Feature', icon=':material/handyman:'):
            if not features_to_remove:
                st.error(':material/error: **No features selected**. Please select at least one feature to remove.')
            else:
                df_processed = df.copy()
                df_processed.drop(columns=features_to_remove, inplace=True)
                st.session_state['current_dataset']['df_file'] = df_processed
                st.session_state['current_dataset']['report_status'] = False
                st.session_state['current_dataset']['report_file'] = None
                del df_processed
                st.rerun()
        
        st.write('---')
        st.subheader(':material/table: Current Dataset')        
        display_dataset(df)
    
    # 4. One-Hot Encoding
    with tab4:
        st.write('This section **converts categorical features with multiple unique values (ranging from 3 to 10) into binary features**. This transformation ensures that categorical data is properly formatted for machine learning models by creating separate columns for each unique category, making the dataset more suitable for numerical processing.')
        
        if st.session_state['current_dataset']['df_file'].isnull().sum().sum() > 0: # Ensure that the dataset does not contain missing values
            st.warning(':material/warning: **Your \'Current Dataset\' contains missing values**. Please handle them on the **Data Cleaning** page first.')
        else:
            categorical_features = df.select_dtypes(include=['number', 'object']).loc[:, df.nunique() <= 10].loc[:, df.nunique() > 2]
            categorical_features = categorical_features.loc[:, ~categorical_features.apply(pd.api.types.is_bool_dtype)].columns
            if len(categorical_features) == 0:
                st.warning(':material/warning: **No categorical features available for one-hot encoding**. Ensure there are features with unique values between 3 and 10.')
            else:
                features_to_encode = st.multiselect('**Select categorical features to encode**', categorical_features)
                if st.button(label='One-Hot Encode', icon=':material/handyman:'):
                    if not features_to_encode:
                        st.error(':material/error: **No features selected**. Please select at least one feature to encode.')
                    else:
                        df_processed = df.copy()
                        df_processed = pd.get_dummies(df_processed, columns=features_to_encode, drop_first=True)
                        for col in df_processed.select_dtypes(include=['bool']).columns:
                            df_processed[col] = df_processed[col].astype('boolean')
                        st.session_state['current_dataset']['df_file'] = df_processed
                        st.session_state['current_dataset']['report_status'] = False
                        st.session_state['current_dataset']['report_file'] = None
                        del df_processed
                        st.rerun()
        
        st.write('---')
        st.subheader(':material/table: Current Dataset')          
        display_dataset(df)    