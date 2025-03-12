import streamlit as st

st.set_page_config(page_title='Datalyze', page_icon='assets/images/logo_only_500px_circle.png')

# Sidebar logo
st.logo('assets/images/logo_name_horizontal_265px_withoutbg.png',icon_image='assets/images/logo_only_350px_withoutbg.png', size='large')

# Sidebar menu:
home = st.Page('pages/0_home.py', title='Home', icon=':material/home:', default=True)
upload_dataset = st.Page('pages/1_upload_dataset.py', title='Upload Dataset', icon=':material/upload:')
eda = st.Page('pages/2_eda.py', title='Exploratory Data Analysis', icon=':material/bar_chart_4_bars:')
data_cleaning = st.Page('pages/3_data_cleaning.py', title='Data Cleaning', icon=':material/mop:')
feature_engineering = st.Page('pages/4_feature_engineering.py', title='Feature Engineering', icon=':material/handyman:')
machine_learning = st.Page('pages/5_machine_learning.py', title='Machine Learning', icon=':material/function:')

pg = st.navigation(
    [
        home,
        upload_dataset,
        eda,
        data_cleaning,
        feature_engineering,
        machine_learning,
    ]
)

pg.run()
