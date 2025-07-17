import io
import joblib
import zipfile
import streamlit as st

def download_model_and_scaler(model, model_name, is_scaled, scaler=None):
    # Save model to a buffer memory
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)
    zip_buffer = io.BytesIO()
    if is_scaled:
        # save scaler to a buffer memory if scaling is applied
        scaler_buffer = io.BytesIO()
        joblib.dump(scaler, scaler_buffer)
        scaler_buffer.seek(0)
        # Create a zip file containing both model and scaler
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr(
                f'sklearn_classifier_{model_name.lower().replace(" ", "_").replace("-", "_")}.pkl',
                model_buffer.getvalue()
            )
            zip_file.writestr('scaler.pkl', scaler_buffer.getvalue())
    else:
        # Create a zip file containing only the model
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr(
                f'sklearn_classifier_{model_name.lower().replace(" ", "_").replace("-", "_")}.pkl',
                model_buffer.getvalue()
            )
    zip_buffer.seek(0)

    # Download button for the zip file
    st.download_button(
        label='Download Model',
        icon=':material/download:',
        data=zip_buffer,
        file_name=f'sklearn_model_{model_name.lower().replace(" ", "_").replace("-", "_")}.zip',
        mime='application/zip',
        use_container_width=True
    )