import streamlit as st
from assets.styles.styler import apply_global_style

# Page Style
apply_global_style()

# Page Header with Logo
st.image('assets/images/logo_name_horizontal_817px.png', use_container_width =True)
st.title('Welcome to Datalyze!')
st.write('**Datalyze** is your all-in-one data analytics platform, designed to transform raw data into meaningful insights with ease. Whether you\'re an analyst, a data scientist, or a business professional, Datalyze empowers you to explore, clean, prepare, and model your data effortlessly. With an intuitive interface, automated processes, and machine learning capabilities, you can make data-driven decisions faster than ever before.')
if st.button('Get Started', icon=':material/prompt_suggestion:', key='header_button'):
    st.switch_page('pages/1_upload_dataset.py')
    
# Section: Why Use Datalyze?
st.markdown('---')
st.subheader('Why Choose Datalyze?')
st.write('Datalyze is built with efficiency, accuracy, and user experience in mind. Here\'s why it stands out:')
st.info('''
- **Dynamic & Accessible**: Access the platform seamlessly on both **laptops** and **mobile phones**.
- **User-Friendly**: Designed for all levels of expertise—**no coding required**.  
- **Comprehensive & Fast**: : Automates workflows from data to insights for **fast analysis**.
- **Advanced AI & ML**: Select various machine learning models for predictive analysis powered by **Scikit-Learn**.  
- **Hyperparameter Tuning**: Optimize model performance with **Tree-structured Parzen Estimator (TPE)** algorithm.
- **Export & Share**: Download reports, models, and visualizations for further analysis or presentation.
''')

# Section: Key Features
st.markdown('---')
st.subheader('Key Features')
st.write('Datalyze provides an end-to-end solution for your data analytics needs. Explore the features below:')

# 1. Upload Dataset
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('#### :material/upload: Upload Dataset')
    st.write('Kickstart your data analytics journey by uploading a dataset in **CSV** or **Excel** format. Datalyze ensures a smooth data ingestion process with the following features')
    st.info('''
    - **Instant Data Preview**: View the dataset to confirm data structure before proceeding.
    - **Automatic Data Recognition**: The system intelligently detects dataset information and structure.
    - **Data Deletion**: Need to start fresh? Delete the uploaded dataset with a single click.  
    ''')
with col2:
    st.image('assets/images/logo_name_vertikal_500px.png', use_container_width=True)

# 2. Exploratory Data Analysis
col1, col2 = st.columns([1, 1])
with col1:
    st.image('assets/images/logo_name_vertikal_500px.png', use_container_width=True)
with col2:
    st.markdown('#### :material/bar_chart_4_bars: Exploratory Data Analysis')
    st.write('Gain deeper insights into your dataset with **automated EDA reports**. Datalyze quickly generates summaries that highlight patterns, trends, and key statistics.')
    st.info('''
    - **Data Assessing**: Check for missing values, duplicates, and more.
    - **Descriptive Statistics**: Show mean, median, mode, and other statistical measures.
    - **Data Visualization**: display data distribution, correlation, and more with interactive plots.
    ''')

# 3. Data Cleaning
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('#### :material/mop: Data Cleaning')
    st.write('Messy data? No problem! Datalyze provides cleaning tools to ensure data consistency and accuracy before analysis.')
    st.info('''
    - **Feature Type Modification**: Convert data types to the correct format.  
    - **Handle Missing Values**: Choose between deletion or imputation methods.  
    - **Remove Duplicates**: Ensure unique and non-redundant data.  
    - **Outlier Detection & Removal**: Detect anomalies and remove them for better analysis.  
    ''')
with col2:
    st.image('assets/images/logo_name_vertikal_500px.png', use_container_width=True)

# 4. Feature Engineering
col1, col2 = st.columns([1, 1])
with col1:
    st.image('assets/images/logo_name_vertikal_500px.png', use_container_width=True)
with col2:
    st.markdown('#### :material/handyman: Feature Engineering')
    st.write('Prepare your data for advanced analytics and machine learning by transforming raw datasets into optimized structures. Feature Engineering in Datalyze includes:')
    st.info('''  
    - **Renaming Features**: Standardize column names for better readability.  
    - **Feature Addition**: Create new features using basic math and polynomial transformations. 
    - **Feature Selection & Removal**: Remove irrelevant or redundant columns.  
    - **One-Hot Encoding**: Convert multi-categorical data into numerical representations.  
    ''')

# 5. Machine Learning
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('#### :material/function: Machine Learning')
    st.write('Take your data to the next level with **built-in machine learning models** powered by **Scikit-Learn**. Datalyze provides a streamlined workflow for predictive modeling:')
    st.info('''  
    - **Dataset Configuration**: Select input features (`X`) and target variable (`Y`).  
    - **Train-Test Split**: Adjust the ratio to optimize model performance.
    - **Scaling & Normalization**: Ensure consistent data distribution for ML models.
    - **Problem Type Selection**: Choose between **Regression** or **Binary Classification** tasks.  
    - **Manual & Automated Hyperparameter Tuning**: Optimize models with either manual settings or  
    **TPE-based tuning** for automated best-parameter selection.  
    - **Performance Metrics Dashboard**: Evaluate model accuracy, precision, recall, and F1-score.  
    - **Model Export**: Download trained models in `.joblib` format for deployment or further analysis.  
    ''')
with col2:
    st.image('assets/images/logo_name_vertikal_500px.png', use_container_width=True)

# Call to Action
st.markdown('---')
st.subheader('Start Your Data Journey Now!')
st.write('Explore your data, gain powerful insights, and make data-driven decisions with ease.')
if st.button('Get Started', icon=':material/prompt_suggestion:', key='footer_button'):
    st.switch_page('pages/1_upload_dataset.py')