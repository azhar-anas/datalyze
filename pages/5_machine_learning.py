import streamlit as st
import joblib
import io
from assets.styles.styler import apply_global_style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from models.classification_models import logr_param_selector, dt_c_param_selector, rf_c_param_selector, knn_c_param_selector, svc_param_selector, nb_c_param_selector, ada_c_param_selector, gb_c_param_selector, mlp_c_param_selector
from models.regression_models import lr_param_selector, lasso_param_selector, ridge_param_selector, dt_r_param_selector, rf_r_param_selector, knn_r_param_selector, svr_param_selector, ada_r_param_selector, gb_r_param_selector, mlp_r_param_selector
from utils.data_visualization import display_dataset, plot_confusion_matrix, plot_classification_metrics, plot_roc_curve, plot_precision_recall_curve, display_regression_metrics, plot_predicted_vs_actual, plot_predicted_vs_residuals, plot_kde

# Page Style
apply_global_style()

# Page Header 
st.title(':material/function: Machine Learning')
st.write('Unlock the potential of machine learning to make data-driven decisions. This page allows you to configure datasets, select models, and evaluate performance, ensuring that your predictions are optimized for accuracy and efficiency. Choose between manual model configuration or automated hyperparameter tuning to achieve the best results.')

if 'current_dataset' not in st.session_state: # Ensure that the dataset has been uploaded
    st.warning(':material/warning: **No dataset found**. Please upload a dataset on the Upload Dataset page first.')
elif st.session_state['current_dataset']['df_file'].isnull().sum().sum() > 0: # Ensure that the dataset does not contain missing values
    st.warning(':material/warning: **Your \'Current Dataset\' contains missing values**. Please handle them on the **Data Cleaning** page first.')
else: # Main Code Start From Here
    st.subheader(':material/manufacturing: Data Configuration')
    st.write('Configure your dataset for training by selecting the input features (**X**) and target variable (**Y**). Adjust the **train-test split ratio** and apply a **scaling method** (Min-Max Normalization, Z-Score Standardization, or Robust Scaling) to optimize model performance. Proper data configuration ensures your model is well-prepared for training.')
    
    # Data Configuration
    # A. Select Dataset
    dataset_choice = st.selectbox('**Select Dataset**', ['Current Dataset', 'Raw Dataset'])
    if dataset_choice == 'Current Dataset':
        selected_df = st.session_state['current_dataset']['df_file'].copy()
    else:
        selected_df = st.session_state['raw_dataset']['df_file'].copy()
    
    # Check for missing values on Raw Dataset
    if selected_df.isnull().sum().sum() > 0:
        st.error(':material/error: **Your \'Raw Dataset\' contains missing values**. You can\'t proceed with machine learning analysis with this dataset.')
    else:
        with st.container(border=True, key='data_config_container'):
            display_dataset(selected_df, border=False)
            st.write('')
            col1, col2 = st.columns(2)
            with col1:
                # 2. Select Features
                features = st.multiselect('**Select independent features (X variables)**', selected_df.columns, selected_df.columns)
                target = st.selectbox('**Select dependent feature (Y variable)**', selected_df.columns)
            with col2:
                # 3. Data Splitting
                train_size = st.slider('**Train Size**', min_value=0.1, max_value=0.9, step=0.05, value=0.8)
                test_size = st.slider('**Test Size**', min_value=0.1, max_value=0.9, value=1-train_size, disabled=True)
                random_state = st.number_input('**Random State (0 - 100)** -> Control Splitting Reproducibility', min_value=0, max_value=100, step=1, value=42)
                
                # 4. Feature Scaling
                normalization_method = st.selectbox('**Select feature scaling method**', ['None', 'Min-Max Normalization', 'Z-Score Standardization', 'Robust Scaling'])
            
            # Apply Changes Button
            if st.button(label='Apply Changes', icon=':material/manufacturing:', use_container_width=True):
                processed_df = selected_df.copy()
                X = processed_df[features]
                y = processed_df[target]
                if features == []:
                    st.error(':material/error: **Please select at least one independent feature**.')
                elif X.select_dtypes(include=['object', 'datetime']).shape[1] > 0 or y.dtype in ['object', 'datetime']:
                    st.error(':material/error: **Selected features contain non-numeric data types**. Please select only numeric or boolean features.')
                else:
                    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
                    numeric_cols = x_train.select_dtypes(include=['number']).columns
                    scaled_status = False
                    if normalization_method == 'Min-Max Normalization':
                        scaler = MinMaxScaler()
                        scaled_status = True
                    elif normalization_method == 'Z-Score Standardization':
                        scaler = StandardScaler()
                        scaled_status = True
                    elif normalization_method == 'Robust Scaling':
                        scaler = RobustScaler()
                        scaled_status = True
                    else:
                        scaler = None

                    if scaler:
                        x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
                        x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

                    st.session_state['dataset_split'] = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test, 'train_size': train_size, 'is_scaled': scaled_status, 'norm_method': normalization_method}
                    st.success(':material/task_alt: Data has been split successfully!')
                    st.write(f'Training Set Shape ({normalization_method if scaled_status else "Unnormalized"}):', x_train.shape, y_train.shape)
                    st.write(f'Test Set Shape ({normalization_method if scaled_status else "Unnormalized"}):', x_test.shape, y_test.shape)
                    del processed_df, x_train, x_test, y_train, y_test, X, y, numeric_cols, scaler, scaled_status
        
        # B. Model Selection
        st.write(''); st.write('')
        st.subheader(':material/autorenew: Model Selection')
        st.write('Choose the most suitable machine learning model for your problem. Select between **manual configuration**, where you specify the model and its hyperparameters, or **automated hyperparameter tuning**, which optimizes the parameters using advanced search techniques. The hyperparameter tuning process allows you to define categorical and numerical hyperparameters, set search boundaries, specify cross-validation folds, choose an evaluation metric, and define the number of iterations for **Tree-structured Parzen Estimator (TPE)** optimization.')
        
        col1, col2 = st.columns(2)
        with col1:
            ml_problem = st.selectbox('**Machine Learning Problem**', ['Classification (2 Classes)', 'Regression'])
        with col2:
            if ml_problem == 'Classification (2 Classes)':
                model_choice = st.selectbox('**Select Model**', ['Logistic Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine', 'Naive Bayes', 'AdaBoost', 'Gradient Boosting', 'Multi-Layer Perceptron (Neural Network)'])
            else:
                model_choice = st.selectbox('**Select Model**', ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine', 'AdaBoost', 'Gradient Boosting', 'Multi-Layer Perceptron (Neural Network)'])
        
        tab1, tab2 = st.tabs(['Manual', 'Hyperparameter Tuning'])
        with tab1:
            with st.container(border=True, key='model_config_container'):
                model_selection_type = 'Manual'
                if ml_problem == 'Classification (2 Classes)':
                    if model_choice == 'Logistic Regression':
                        model = logr_param_selector()
                    elif model_choice == 'Decision Tree':
                        model = dt_c_param_selector()
                    elif model_choice == 'Random Forest':
                        model = rf_c_param_selector()
                    elif model_choice == 'K-Nearest Neighbors':
                        model = knn_c_param_selector()
                    elif model_choice == 'Support Vector Machine':
                        model = svc_param_selector()
                    elif model_choice == 'Naive Bayes':
                        model = nb_c_param_selector()
                    elif model_choice == 'AdaBoost':
                        model = ada_c_param_selector()
                    elif model_choice == 'Gradient Boosting':
                        model = gb_c_param_selector()
                    elif model_choice == 'Multi-Layer Perceptron (Neural Network)':
                        model = mlp_c_param_selector()
                else:
                    if model_choice == 'Linear Regression':
                        model = lr_param_selector()
                    elif model_choice == 'Lasso Regression':
                        model = lasso_param_selector()
                    elif model_choice == 'Ridge Regression':
                        model = ridge_param_selector()
                    elif model_choice == 'Decision Tree':
                        model = dt_r_param_selector()
                    elif model_choice == 'Random Forest':
                        model = rf_r_param_selector()
                    elif model_choice == 'K-Nearest Neighbors':
                        model = knn_r_param_selector()
                    elif model_choice == 'Support Vector Machine':
                        model = svr_param_selector()
                    elif model_choice == 'AdaBoost':
                        model = ada_r_param_selector()
                    elif model_choice == 'Gradient Boosting':
                        model = gb_r_param_selector()
                    elif model_choice == 'Multi-Layer Perceptron (Neural Network)':
                        model = mlp_r_param_selector()

                if st.button(label='Train Model', icon=':material/autorenew:', use_container_width=True):
                    if st.session_state['dataset_split']['x_train'] is None:
                        st.error(':material/error: **Please split the data before training the model**.')
                    elif ml_problem == 'Classification (2 Classes)' and st.session_state['dataset_split']['y_train'].dtype not in ['bool', 'boolean']:
                        st.error(':material/error: **Selected dependent feature must be boolean type for classification problem**.')
                    else:
                        model.fit(st.session_state['dataset_split']['x_train'], st.session_state['dataset_split']['y_train'])
                        st.session_state['model_detail'] = {'train_status': True, 'model_selection': model_selection_type, 'problem': ml_problem, 'model': model, 'model_name': model_choice, 'model_params': model.get_params(), 'used_dataset': st.session_state['dataset_split']}
                        st.success(':material/task_alt: Model has been trained successfully')
                        del model

        with tab2:
            with st.container(border=True, key='model_tuning_container'):
                st.error('**Error**: Hyperparameter Tuning is not available yet. Please use manual model selection for now.')
                            
    # C. Model Performance
    st.write(''); st.write('')
    st.subheader(':material/search_insights: Model Performance')
    st.write('Assess the effectiveness of your trained model using comprehensive performance metrics. View key details about the **dataset**, **model configuration**, and **visualizations of evaluation results**. Compare model performance using **baseline references** from either the training or test data. Once satisfied, you can **download the trained model** for further use.')
    st.write('')
    
    if st.session_state['model_detail']['train_status'] == True:
        model_detail = st.session_state['model_detail']
        model_selection_type = model_detail['model_selection']
        ml_problem = model_detail['problem']
        model = model_detail['model']
        model_name = model_detail['model_name']
        model_params = model_detail['model_params']
        
        used_dataset = model_detail['used_dataset']
        x_train = used_dataset['x_train']
        y_train = used_dataset['y_train']
        x_test = used_dataset['x_test']
        y_test = used_dataset['y_test']
        train_size = used_dataset['train_size']
        normalization_method = used_dataset['norm_method']
        
        col1, col2 = st.columns([1, 1])
        with col1: # Model Summary
            st.write(f'**Model Problem**: `{ml_problem}`')
            st.write(f'**Model Selection Type**: `{model_selection_type}`')
            st.write(f'**Model Name**: `{model_name}`')
            params_expander = st.expander("**Model Parameters**")
            with params_expander:
                for param, value in model_params.items():
                    st.write(f'{param}: `{value}`')
        with col2: # Dataset Summary
            st.write(f'**Feature Target**: `{target}`')
            st.write(f'**Test Set Shape**: ', x_test.shape)
            st.write(f'**Training Set Shape**: ', x_train.shape)
            st.write(f'**Dataset Split Ratio (Train/Test)**: `{train_size}/{round(1-train_size, 2)}`')
            st.write(f'**Dataset Feature Scaling**: `{normalization_method}`')
        st.write('')
        
        # Select Dataset for Baseline Model Performance
        dataset_type = st.selectbox('**Select Dataset for Baseline Model Performance**', ['Test Set', 'Training Set'])
        if dataset_type == 'Test Set':
            selected_x = x_test
            selected_y = y_test
        else:
            selected_x = x_train
            selected_y = y_train
        
        y_pred = model.predict(selected_x)
        
        # METRICS FOR CLASSIFICATION PROBLEM
        if ml_problem == 'Classification (2 Classes)':
            col1, col2 = st.columns(2)
            with col1:
                # 1. Confusion Matrix Graph
                plot_confusion_matrix(selected_y, y_pred)

            with col2:
                # 2. Metrics Graph
                plot_classification_metrics(selected_y, y_pred)

            col1, col2 = st.columns(2)
            with col1:
                # 3. ROC Curve
                plot_roc_curve(model, selected_x, selected_y)

            with col2:
                # 4. Precision-Recall Curve
                plot_precision_recall_curve(model, selected_x, selected_y)
                
        # METRICS FOR REGRESSION PROBLEM
        else:
            # 1. Display Regression Metrics
            display_regression_metrics(selected_y, y_pred)

            col1, col2 = st.columns(2)
            with col1:
                # 2. Scatter Plot of Predictions vs Actual
                plot_predicted_vs_actual(selected_y, y_pred)

            with col2:
                # 3. Residual Plot
                plot_predicted_vs_residuals(selected_y, y_pred)

            # 4. Distribution Plot (KDE) of Predictions vs Actual
            plot_kde(selected_y, y_pred)

        # Download Model Button
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        if ml_problem == 'Classification (2 Classes)':
            model_filename = 'sklearn_classifier_' + model_name.lower().replace(' ', '_').replace('-', '_') + '.joblib'
        else:
            model_filename = 'sklearn_regressor_' + model_name.lower().replace(' ', '_').replace('-', '_') + '.joblib'
        st.download_button(
            label='Download Model',
            icon=':material/download:',
            data=buffer,
            file_name=model_filename,
            mime='application/octet-stream',
            use_container_width=True
        )

    else:
        st.warning(':material/warning: **The model performance metrics are based on the latest model trained**. If you have not trained a model yet, please train a model first.')
