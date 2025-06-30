import streamlit as st
import pandas as pd
import gc
from assets.styles.styler import apply_global_style
from utils.data_visualization import display_dataset

# Page Style
apply_global_style()

# Page Header 
st.title(':material/upload: Upload Dataset')
st.write('Upload your dataset in **CSV** or **Excel** format to begin. Once uploaded, Datalyze automatically creates two dataset versions: the **Raw Dataset** and the **Current Dataset**. The **Raw Dataset** remains unchanged as a reference point, preserving the original data, while the **Current Dataset** serves as the working version for all analysis and transformations. You can download your dataset anytime. If needed, You can reset **Current Dataset** or delete **All Datasets** with a single click to start fresh.')
st.info('Initially, both datasets are identical, but as you clean, preprocess, and modify data, only the **Current Dataset** is affected. This dual-dataset approach allows you to compare processed results against the original data, ensuring transparency and traceability. Try it out with our **[sample dataset](https://drive.google.com/drive/folders/1YjTGIco0dqR8hDHGrOGza0eMZqMVpOAc?usp=sharing)**.') 

# Upload Dataset & Cache Initialization
col1, col2 = st.columns([7, 3])
with col1:
    file = st.file_uploader('**Upload Dataset**', type=['csv', 'xlsx'])
    if file:
        df = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
        
        # Raw Dataset Cache Initialization
        st.session_state['raw_dataset'] = {
                                            'df_file': df, 
                                            'report_status': False,
                                            'report_file': None
                                        }
        
        # Current Dataset Cache Initialization
        st.session_state['current_dataset'] = { 
                                                'df_file': df,
                                                'report_status': False,
                                                'report_file': None
                                            }
        
        # Outliers Removed Status Cache Initialization
        st.session_state['outliers_removed'] = False

        # Dataset Split Information Cache Initialization
        st.session_state['dataset_split'] = {
                                                'x_train': None,
                                                'x_test': None,
                                                'y_train': None,
                                                'y_test': None,
                                                'train_size': None,
                                                'is_scaled': False,
                                                'norm_method': None
                                            }
        
        # Model Detail Cache
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
if 'raw_dataset' in st.session_state:
    with col2:
        # Show Dropdown to select dataset
        dataset_option = st.selectbox('**Select Dataset to View**', ['Raw Dataset', 'Current Dataset'])
        
    if dataset_option == 'Raw Dataset':
        # Show Raw Dataset
        st.write('')
        df = st.session_state['raw_dataset']['df_file'].copy()
        display_dataset(df)
        
        del df
        gc.collect()
        
        # Delete All Datasets Button
        if st.button('Delete All Datasets', icon=':material/delete:'):
            del st.session_state['raw_dataset']
            del st.session_state['current_dataset']
            del st.session_state['outliers_removed']
            del st.session_state['dataset_split']
            del st.session_state['model_detail']
            st.cache_data.clear()
            gc.collect()
            st.rerun()
        
    else:
        # Show Current Dataset
        st.write('')
        df = st.session_state['current_dataset']['df_file'].copy()
        display_dataset(df)
        
        del df
        gc.collect()
        
        # Reset Current Dataset Button
        if st.button('Reset Current Dataset', icon=':material/refresh:'):
            st.session_state['current_dataset'] = st.session_state['raw_dataset'].copy()
            st.session_state['outliers_removed'] = False
            st.rerun()
