<p align="center">
  <img src="assets/images/logo_name_horizontal_817px.png" alt="Datalyze Header" width="700">
</p>

<h1 align="center">Datalyze</h1>

<p align="center">
  <a href="https://www.python.org" target="_blank"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> </a>
  <a href="https://streamlit.io/" target="_blank"> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"> </a>
  <a href="https://pandas.pydata.org/" target="_blank"> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"> </a>
  <a href="https://matplotlib.org/" target="_blank"> <img src="https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=matplotlib&logoColor=white"> </a>
  <a href="https://seaborn.pydata.org/" target="_blank"> <img src="https://img.shields.io/badge/Seaborn-80b6ff?style=for-the-badge&logo=seaborn&logoColor=white"> </a>
  <a href="https://scikit-learn.org/" target="_blank"> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"> </a>
  <a href="https://optuna.org/" target="_blank"> <img src="https://img.shields.io/badge/Optuna-00468B?style=for-the-badge&logo=optuna&logoColor=white"> </a>
  <a href="https://joblib.readthedocs.io/en/latest/" target="_blank"> <img src="https://img.shields.io/badge/Joblib-00468B?style=for-the-badge&logo=joblib&logoColor=white"> </a>
</p>

Datalyze is a powerful no-code data analytics platform that simplifies the entire data science pipeline, from initial data ingestion to the development of sophisticated machine learning models. Built for analysts, data scientists, and business professionals alike, its intuitive web interface allows users to seamlessly conduct exploratory data analysis, perform automated data cleaning, and carry out robust feature engineering. With Datalyze, you can effortlessly build and optimize machine learning models in a single, integrated environment, transforming raw data into actionable insights without writing a single line of code.

---

## 🚀 Key Capabilities

1.  Upload and preview **CSV/Excel files** up to **200 MB** with automatic schema detection and basic statistical reporting.
2.  Automatically generate advanced statistical reports, detect missing values, imbalanced data, correlation, and create visualizations.
3.  Edit column data types, handle missing values, remove duplicates, and detect/remove outliers.
4.  Rename, transform, delete, and encode specific columns for modeling.
5.  Train **regression** and **binary classification models** with scaling, train-test split, and hyperparameter tuning (manual or automated via the **Tree-structured Parzen Estimator (TPE)** algorithm).
6.  Download the optimized model as a **.pkl** file for reuse or deployment.
7.  Export **Exploratory Data Analysis reports** (HTML), data visualizations (images), and the cleaned dataset (CSV) for further analysis or presentation.

---

## 🛠️ Tech Stack

Datalyze is built using the following technologies:
- **Python**: Core language for backend processing
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-Learn**: Machine learning models Development
- **Optuna**: Hyperparameter optimization (TPE algorithm)
- **Joblib**: Model serialization and deployment

---

## 📖 Getting Started

You can run Datalyze either directly from the web or by setting it up locally on your machine.

### 1️⃣ Run Online

The easiest way to get started is by using the live web application, hosted on Streamlit Community Cloud.

🔗 **[Launch Datalyze App](https://datalyze-zkrjbtewkocqze5mt5gh55.streamlit.app/)**

### 2️⃣ Run Locally

For a local setup, ensure you have **Python 3.11** installed. Then, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/azhar-anas/datalyze.git
    ```

2.  **Install dependencies:**
    ```bash
    cd datalyze
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run App.py
    ```

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
