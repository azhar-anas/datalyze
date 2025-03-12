import streamlit as st

# Global Page Style
def apply_global_style():
    st.markdown("""
    <style>
        .block-container {
            max-width: 80%;
            padding-top: 4.5rem;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1rem;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)