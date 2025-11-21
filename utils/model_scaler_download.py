import io
import joblib
import zipfile
import streamlit as st
import json

def download_model_and_scaler(model, model_name, ml_problem, data_type, min_max_values, target_var_name, is_scaled, scaler=None,):
    # Save model to a buffer memory
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)
    zip_buffer = io.BytesIO()

    # Create structured metadata for model input and target
    metadata = {
        'features': [
            {
                'name': col,
                'type': data_type.get(col),
                'min': min_max_values.get(col, {}).get('min'),
                'max': min_max_values.get(col, {}).get('max')
            }
            for col in data_type
        ],
        'target': {
            'name': target_var_name,
            'type': data_type.get(target_var_name),
        },
        'case': ml_problem
    }


    if is_scaled:
        # save scaler to a buffer memory if scaling is applied
        scaler_buffer = io.BytesIO()
        joblib.dump(scaler, scaler_buffer)
        scaler_buffer.seek(0)
        model_buffer.seek(0)
        # Create a zip file containing model, scaler, and metadata
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr(
                f'sklearn_{model_name.lower().replace(" ", "_").replace("-", "_")}.pkl',
                model_buffer.getvalue()
            )
            zip_file.writestr('scaler.pkl', scaler_buffer.getvalue())
            zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))
    else:
        # Create a zip file containing only the model and metadata
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr(
                f'sklearn_{model_name.lower().replace(" ", "_").replace("-", "_")}.pkl',
                model_buffer.getvalue()
            )
            zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))
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