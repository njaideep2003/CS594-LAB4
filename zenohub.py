## If you don't have a Zeno account already, create one on Zeno Hub (https://hub.zenoml.com/signup). 
## After logging in to Zeno Hub, generate your API key by clicking on your profile at the top right to navigate to your account page.

# !pip install zeno_client

import os
from dotenv import load_dotenv
from zeno_client import ZenoClient, ZenoMetric
import pandas as pd

# Load API Key from .env file
load_dotenv()
API_KEY = os.getenv("ZENO_API_KEY")

if not API_KEY:
    raise ValueError("❌ API Key not found. Make sure you have a .env file with ZENO_API_KEY set.")

# Initialize a client with the API key
client = ZenoClient(API_KEY)

df = pd.read_csv('tweets.csv')
df = df.reset_index()

project = client.create_project(
    name="Tweet Sentiment Analysis",
    view="text-classification",
    metrics=[
        ZenoMetric(name="accuracy", type="mean", columns=["correct"]),
    ]
)

project.upload_dataset(df, id_column="index", data_column="text", label_column="label")

models = ['roberta', 'gpt2']
for model in models:
    df_system = df[['index', model]]
    
    # Measure accuracy for each instance, which is averaged by the ZenoMetric above
    df_system["correct"] = (df_system[model] == df["label"]).astype(int)
    
    project.upload_system(df_system, name=model, id_column="index", output_column=model)

print("✅ ZenoHub setup complete. Project uploaded successfully!")