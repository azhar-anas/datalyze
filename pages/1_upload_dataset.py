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
            keys_to_keep = ['logged_in']
            keys_to_delete = [k for k in st.session_state.keys() if k not in keys_to_keep]
            for k in keys_to_delete:
                del st.session_state[k]
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



# OLD CODE --------------------------------------------------------------------------------------------------------------------------
# import streamlit as st

# st.set_page_config(page_title='Datalyze', page_icon='assets/images/logo_only_500px_circle.png')

# # Sidebar logo
# st.logo('assets/images/logo_name_horizontal_265px_withoutbg.png',icon_image='assets/images/logo_only_350px_withoutbg.png', size='large')

# # Sidebar menu:
# home = st.Page('pages/0_home.py', title='Home', icon=':material/home:', default=True)
# upload_dataset = st.Page('pages/1_upload_dataset.py', title='Upload Dataset', icon=':material/upload:')
# eda = st.Page('pages/2_eda.py', title='Exploratory Data Analysis', icon=':material/bar_chart_4_bars:')
# data_cleaning = st.Page('pages/3_data_cleaning.py', title='Data Cleaning', icon=':material/mop:')
# feature_engineering = st.Page('pages/4_feature_engineering.py', title='Feature Engineering', icon=':material/handyman:')
# machine_learning = st.Page('pages/5_machine_learning.py', title='Machine Learning', icon=':material/function:')

# pg = st.navigation(
#     [
#         home,
#         upload_dataset,
#         eda,
#         data_cleaning,
#         feature_engineering,
#         machine_learning,
#     ]
# )

# pg.run()
# OLD CODE --------------------------------------------------------------------------------------------------------------------------
