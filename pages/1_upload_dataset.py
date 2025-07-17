import streamlit as st
import pandas as pd
import gc
from assets.styles.styler import apply_global_style
from utils.data_visualization import display_dataset

# Page Style
apply_global_style()

# Page Header 
st.title(':material/upload: Upload Dataset')
st.write('Upload your dataset in **CSV** or **Excel** format to begin. Once uploaded, Datalyze automatically creates the **Current Dataset** as the working version for all analysis and transformations. You can download your dataset anytime. If needed, You can reset **Current Dataset** with a single click to start fresh. Try it out with our **[sample dataset](https://drive.google.com/drive/folders/1YjTGIco0dqR8hDHGrOGza0eMZqMVpOAc?usp=sharing)**.')

# Upload Dataset & Cache Initialization
file = st.file_uploader('**Upload Dataset**', type=['csv', 'xlsx'])
if file:
    df = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)

    # Current Dataset Cache Initialization
    st.session_state['current_dataset'] = {
        'df_file': df,
        'df_file_raw': df,
        'report_status': False,
        'report_file': None,
        'outliers_removed': False,
        'label_encoders_list': None
    }

    # Dataset Split Information Cache Initialization
    st.session_state['dataset_split'] = {
        'x_train': None,
        'x_test': None,
        'y_train': None,
        'y_test': None,
        'train_size': None,
        'is_scaled': False,
        'norm_method': None,
        'norm_file': None
    }

    # Model Detail Cache Initialization
    st.session_state['model_detail'] = {
        'train_status': False,
        'model_selection': None,
        'problem': None,
        'model': None,
        'model_name': None,
        'model_params': None,
        'used_dataset': None
    }

# Display Dataset
if 'current_dataset' in st.session_state:
    df = st.session_state['current_dataset']['df_file'].copy()
    display_dataset(df)
    del df
    gc.collect()    

    if st.button('Reset Dataset', key='reset_btn', help='Reset current dataset to original', icon=':material/refresh:', use_container_width=True):
        st.session_state['current_dataset']['df_file'] = st.session_state['current_dataset']['df_file_raw'].copy()
        st.session_state['current_dataset']['report_status'] = False
        st.session_state['current_dataset']['report_file'] = None
        st.session_state['current_dataset']['outliers_removed'] = False
        st.session_state['current_dataset']['label_encoders_list'] = None
        gc.collect()    
        st.rerun()
