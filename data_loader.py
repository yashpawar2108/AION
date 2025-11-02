# import os
# import pandas as pd
# from sklearn.datasets import fetch_openml, load_breast_cancer, fetch_california_housing, load_wine
# from datasets import load_dataset
# import tensorflow_datasets as tfds

# # Create dataset folder
# DATA_DIR = "datasets"
# os.makedirs(DATA_DIR, exist_ok=True)

# def save_dataframe(name, df):
#     path = os.path.join(DATA_DIR, f"{name}.csv")
#     df.to_csv(path, index=False)
#     print(f"✅ Saved {name} → {path}")

# # -------------------
# # 1. Tabular Datasets
# # -------------------
# try:
#     titanic = fetch_openml("titanic", version=1, as_frame=True)
#     save_dataframe("titanic", titanic.frame)
# except Exception as e:
#     print("❌ Titanic:", e)

# try:
#     adult = fetch_openml("adult", version=2, as_frame=True)
#     save_dataframe("adult_income", adult.frame)
# except Exception as e:
#     print("❌ Adult:", e)

# try:
#     heart = fetch_openml("heart-disease-uci", version=1, as_frame=True)
#     save_dataframe("heart_disease", heart.frame)
# except Exception as e:
#     print("❌ Heart Disease:", e)

# try:
#     bc = load_breast_cancer(as_frame=True)
#     df = pd.concat([bc.data, pd.Series(bc.target, name="target")], axis=1)
#     save_dataframe("breast_cancer", df)
# except Exception as e:
#     print("❌ Breast Cancer:", e)

# try:
#     housing = fetch_california_housing(as_frame=True)
#     df = pd.concat([housing.data, pd.Series(housing.target, name="target")], axis=1)
#     save_dataframe("california_housing", df)
# except Exception as e:
#     print("❌ California Housing:", e)

# try:
#     wine = load_wine(as_frame=True)
#     df = pd.concat([wine.data, pd.Series(wine.target, name="target")], axis=1)
#     save_dataframe("wine_quality", df)
# except Exception as e:
#     print("❌ Wine Quality:", e)

# # -------------------
# # 2. Text Datasets
# # -------------------
# try:
#     imdb = load_dataset("imdb")
#     imdb["train"].to_csv(os.path.join(DATA_DIR, "imdb_train.csv"), index=False)
#     imdb["test"].to_csv(os.path.join(DATA_DIR, "imdb_test.csv"), index=False)
#     print("✅ Saved IMDB Sentiment Dataset")
# except Exception as e:
#     print("❌ IMDB:", e)

# try:
#     sms = load_dataset("sms_spam")
#     sms["train"].to_csv(os.path.join(DATA_DIR, "sms_spam.csv"), index=False)
#     print("✅ Saved SMS Spam Dataset")
# except Exception as e:
#     print("❌ SMS Spam:", e)

# try:
#     agnews = load_dataset("ag_news")
#     agnews["train"].to_csv(os.path.join(DATA_DIR, "ag_news_train.csv"), index=False)
#     print("✅ Saved AG News Dataset")
# except Exception as e:
#     print("❌ AG News:", e)

# # -------------------
# # 3. Image Datasets
# # -------------------
# try:
#     for name in ["mnist", "fashion_mnist", "cifar10"]:
#         ds, info = tfds.load(name, split="train", with_info=True)
#         print(f"✅ {name} downloaded → size: {info.splits['train'].num_examples} examples")
# except Exception as e:
#     print("❌ Image datasets:", e)

# # -------------------
# # 4. Time Series
# # -------------------
# try:
#     airpass = load_dataset("timeseries", "air_passengers")
#     airpass["train"].to_csv(os.path.join(DATA_DIR, "air_passengers.csv"), index=False)
#     print("✅ Air Passengers Time Series saved")
# except Exception as e:
#     print("❌ Air Passengers:", e)

# print("\n🎯 All dataset downloads attempted.")





# data_loader.py
# Functions for loading data.

import pandas as pd
import config

def load_data(path: str) -> pd.DataFrame:
    """
    Loads a CSV file from the given path.
    
    Args:
        path (str): The file path to the CSV.

    Returns:
        pd.DataFrame: The loaded data.
    """
    print(f"Attempting to load data from {path}...")
    # In a real framework, you might add more logic here
    # (e.g., handling different file types like parquet, excel, etc.)
    df = pd.read_csv(path)
    print("Data loaded successfully.")
    return df

