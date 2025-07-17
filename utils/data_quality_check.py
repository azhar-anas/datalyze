import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns

def find_missing_values(df):
    total_missing = df.isna().sum().sum()
    null_data = df[df.isna().any(axis=1)]
    percent_missing = (total_missing / df.shape[0]) * 100
    return null_data, total_missing, percent_missing

def find_duplicate_values(dataframe):
    duplicates = dataframe[dataframe.duplicated()]
    num_duplicates = duplicates.shape[0]
    percent_duplicates = (num_duplicates / dataframe.shape[0]) * 100
    return duplicates, num_duplicates, percent_duplicates

def find_outliers(df):
    if st.session_state['current_dataset']['outliers_removed'] == False:
        numeric_df = df.select_dtypes(include=['number'])
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
        num_outliers = outliers.shape[0]
        percent_outliers = (num_outliers / df.shape[0]) * 100
        
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=numeric_df, ax=ax)
        ax.set_xlabel('Numerical Features')
        ax.set_ylabel('Values')
        # set x-axis labels with 45 degree rotation
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        outlier_plot = fig
            
        return outliers, outlier_plot, num_outliers, percent_outliers
    else:
        empty_df = df.iloc[0:0]
        outlier_plot = None
        return empty_df, outlier_plot, 0, 0
