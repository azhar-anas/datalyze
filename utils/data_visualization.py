import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

def display_dataset(df, border=True):
    with st.container(border=border):
        col1, col2 = st.columns([7, 3])
        with col1:
            st.dataframe(df, use_container_width=True)
        with col2:
            data_info = pd.DataFrame({
                'Data Type': df.dtypes,
                'Unique Values': df.nunique(),
            })
            st.write(data_info, use_container_width=True)
        
        col1, col2 = st.columns([7, 3])
        with col1:
            with st.expander('Show Descriptive Statistics'):
                st.write(df.describe()) 
        with col2:
            st.write('Dataset Shape: ', df.shape)
    del df, data_info
    gc.collect()



def download_eda_report_button(report, df_type):
    export_html = report.to_html()
    st.download_button(
        label='Download EDA Report', 
        icon=':material/download:', 
        data=export_html, 
        file_name=f'eda_report_{df_type}.html', 
        key=f'eda_report_{df_type}', 
        use_container_width=True
    )



def generate_eda_report(df, df_report):
    df_type = df_report['df_type']
    if df_report['report_status'] == False:
        st.session_state[df_type]['report_file'] = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
        st.session_state[df_type]['report_status'] = True
        gc.collect()
    
    report = st.session_state[df_type]['report_file']
    st_profile_report(report=report)
    gc.collect()
    
    download_eda_report_button(report, df_type)

    del df, df_report
    gc.collect()




def show_interactive_scatter_plot(selected_df):
    numeric_cols = selected_df.select_dtypes(include='number').columns
    if len(numeric_cols) < 2:
        st.warning(':material/warning: **Not Enough Numeric Columns**. Please select a dataset with at least two numeric columns for scatter plot visualization.')
    else:
        col1, col2 = st.columns([3, 10])
        with col1:
            x_col = st.selectbox('**Select X-axis Column**', numeric_cols)
            y_col = st.selectbox('**Select Y-axis Column**', numeric_cols, index=1)
            color_candidates = [col for col in selected_df.columns if selected_df[col].nunique() < 10]
            color_col = st.selectbox('**Select Color Column (Optional)**', ['None'] + color_candidates)
        with col2:
            # if x_col and y_col:
            if x_col == y_col:
                st.warning(':material/warning: **X-axis and Y-axis columns must be different.** Please select two different columns.')
            else:
                st.markdown(
                f"<div style='text-align: center;'><b>Scatter Plot of {x_col} vs {y_col}</b></div>",
                unsafe_allow_html=True
                )
                if color_col != 'None':
                    st.scatter_chart(data=selected_df[[x_col, y_col, color_col]], x=x_col, y=y_col, color=color_col, size=70, height=400)
                    gc.collect()
                else:
                    st.scatter_chart(data=selected_df[[x_col, y_col]], x=x_col, y=y_col, size=70, height=400)
                    gc.collect()




def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return st.pyplot(fig)



def plot_classification_metrics(selected_y, y_pred):
    accuracy = accuracy_score(selected_y, y_pred)
    precision = precision_score(selected_y, y_pred)
    recall = recall_score(selected_y, y_pred)
    f1 = f1_score(selected_y, y_pred)

    metrics = {
        'F1-Score': f1,
        'Recall': recall,
        'Precision': precision,
        'Accuracy': accuracy,
        }

    fig, ax = plt.subplots()
    ax.barh(list(metrics.keys()), list(metrics.values()), color='skyblue')
    for index, value in enumerate(metrics.values()):
        ax.text(value, index, f'{value:.3f}')
    ax.set_xlim([0, 1])
    ax.set_xlabel('Score')
    ax.set_title('Classification Metrics')
    return st.pyplot(fig)



def plot_roc_curve(model, selected_x, selected_y):
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(selected_x)[:, 1]
    else:
        y_pred_proba = model.decision_function(selected_x)
    fpr, tpr, _ = roc_curve(selected_y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    return st.pyplot(fig)



def plot_precision_recall_curve(model, selected_x, selected_y):
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(selected_x)[:, 1]
    else:
        y_pred_proba = model.decision_function(selected_x)
    precision, recall, _ = precision_recall_curve(selected_y, y_pred_proba)
    auc_score = auc(recall, precision)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {auc_score:.3f})')
    ax.plot([0, 1], [1, 0], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('True Positive Rate (Recall)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    return st.pyplot(fig)



def display_regression_metrics(selected_y, y_pred):
    mse = mean_squared_error(selected_y, y_pred)
    rmse = root_mean_squared_error(selected_y, y_pred)
    mae = mean_absolute_error(selected_y, y_pred)
    mape = (abs((selected_y - y_pred) / selected_y).mean()) * 100
    r2 = r2_score(selected_y, y_pred)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("MSE", f"{mse:.3f}")
    with col2:
        st.metric("RMSE", f"{rmse:.3f}")
    with col3:
        st.metric("MAE", f"{mae:.3f}")
    with col4:
        st.metric("MAPE", f"{mape:.2f}%")
    with col5:
        st.metric("R2", f"{r2:.3f}")



def plot_predicted_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_true)
    ax.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], color='red', linestyle='--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Predicted vs Actual')
    return st.pyplot(fig)



def plot_predicted_vs_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(y=0, color='red', linestyle='--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')
    ax.set_title('Predicted vs Residuals')
    return st.pyplot(fig)



def plot_kde(selected_y, y_pred):
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.kdeplot(selected_y, label='Actual', ax=ax)
    sns.kdeplot(y_pred, label='Predicted', ax=ax)
    ax.set_xlabel('Value')
    ax.set_title('Kernel Density Estimation (KDE) Plot')
    ax.legend()
    return st.pyplot(fig)
