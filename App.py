import streamlit as st
import hashlib
import re
import gc
from supabase import create_client, Client

st.set_page_config(page_title='Datalyze', page_icon='assets/images/logo_only_500px_circle.png')

# # Login & Register section with Supabase
# SUPABASE_URL = "https://ramtjckjvdhbamxedovs.supabase.co"
# SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhbXRqY2tqdmRoYmFteGVkb3ZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTEyMTk3NjUsImV4cCI6MjA2Njc5NTc2NX0.DUL-fUhTleUS8XLneh7svW4Rw0jT1OijgeDwlMkLLnI"
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# def show_auth():
#     """
#     Displays authentication UI with Login and Register tabs using Streamlit.
#     - Login tab: Allows users to log in using their username or email and password.
#     - Register tab: Allows new users to register by providing name, username, email, and password.
#     - User data is fetched and stored using Supabase.
#     - Passwords are hashed using SHA-256 for security.
#     - Includes basic validation for registration fields and error handling for database operations.
#     """
#     tab1, tab2 = st.tabs(["Login", "Register"])

#     def load_users():
#         try:
#             res = supabase.table("app_users").select("*").execute()
#             users = res.data if res.data else []
#             del res
#             gc.collect()
#             return users
#         except Exception as e:
#             st.error(f":material/error: Failed to load users. Error: {e}")
#             gc.collect()
#             return []

#     def verify_password(input_password, hashed_password):
#         input_hash = hashlib.sha256(input_password.strip().encode()).hexdigest()
#         return input_hash == hashed_password.strip()

#     with tab1:
#         # Login tab
#         st.title("Login")

#         username_or_email = st.text_input("Username/Email", placeholder="Enter your username or email", key="login_username")
#         password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
#         if st.button("Login", use_container_width=True):
#             try:
#                 users = load_users()
#                 user = next(
#                     (u for u in users if u["username"] == username_or_email or u.get("email") == username_or_email),
#                     None
#                 )
#                 if user and verify_password(password, user["password"]):
#                     st.session_state.logged_in = True
#                     del users, user
#                     gc.collect()
#                     st.rerun()
#                 else:
#                     st.error(":material/error: Username atau password salah.")
#                 del users
#                 gc.collect()
#             except Exception as e:
#                 st.error(f":material/error: Login failed. Error: {e}")
#                 gc.collect()

#     with tab2:
#         # Register tab
#         st.title("Register")

#         reg_name = st.text_input("Name", placeholder="Enter your full name", key="reg_name")
#         reg_username = st.text_input("Username", placeholder="Choose a username", key="reg_username")
#         reg_email = st.text_input("Email", placeholder="Enter your email", key="reg_email")
#         reg_password = st.text_input("Password", type="password", placeholder="Create a password", key="reg_password")
#         reg_password_confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="reg_password_confirm")

#         def is_valid_email(email):
#             return re.match(r"[^@]+@[^@]+\.[^@]+", email)

#         def save_user(new_user):
#             try:
#                 supabase.table("app_users").insert(new_user).execute()
#                 gc.collect()
#                 return True, None
#             except Exception as e:
#                 gc.collect()
#                 return False, e

#         if st.button("Register", use_container_width=True):
#             try:
#                 users = load_users()
#                 if not reg_name or not reg_username or not reg_email or not reg_password or not reg_password_confirm:
#                     st.error(":material/error: Please fill in all fields.")
#                 elif not is_valid_email(reg_email):
#                     st.error(":material/error: Invalid email format.")
#                 elif any(u["username"] == reg_username for u in users):
#                     st.error(":material/error: Username already exists.")
#                 elif any(u.get("email") == reg_email for u in users):
#                     st.error(":material/error: Email already registered.")
#                 elif reg_password != reg_password_confirm:
#                     st.error(":material/error: Passwords do not match.")
#                 elif len(reg_password) < 6:
#                     st.error(":material/error: Password must be at least 6 characters.")
#                 else:
#                     hashed_pw = hashlib.sha256(reg_password.encode()).hexdigest()
#                     new_user = {
#                         "name": reg_name,
#                         "username": reg_username,
#                         "email": reg_email,
#                         "password": hashed_pw
#                     }
#                     success, err = save_user(new_user)
#                     if success:
#                         st.success(":material/check_circle: Registration successful! Please login.")
#                     else:
#                         st.error(f":material/error: Registration failed. Error: {err}")
#                 del users
#                 gc.collect()
#             except Exception as e:
#                 st.error(f":material/error: Registration failed. Error: {e}")
#                 gc.collect()

# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False

# if not st.session_state.logged_in:
#     st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns([1,9,1])
#     with col2:
#         st.image('assets/images/logo_name_horizontal_817px.png', use_container_width=True)
    
#     show_auth()
#     gc.collect()

# else:
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
