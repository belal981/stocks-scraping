from simple_llm import SimpleInvokeLLM
import json 
import pandas as pd
import time

prompt = """
you are an AI agent your task is to classify the articles to one of the following categories: 
['Real Estate' 'Education Services' 'Health Care & Pharmaceuticals'
 'Basic Resources' 'Food, Beverages and Tobacco' 'Building Materials'
 'Shipping & Transportation Services' 'Energy & Support Services'
 'Non-bank financial services' 'Banks' 'Trade & Distributors'
 'Contracting & Construction Engineering' 'Textile & Durables'
 'Industrial Goods , Services and Automobiles'
 'IT , Media & Communication Services' 'Travel & Leisure' 'Utilities']
 
with one only word 

article: 
"""

def load_config():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

config = load_config()
llm = SimpleInvokeLLM(model=config['model'], api_key=config['google_api_key'], temperature=0.0)
df = pd.read_csv("/home/ahmed-hereiz/self/doc-discriminator/predict-stock-price/data/agent_data/scrapped_news_enriched.csv")
if 'llm_predicted_class' not in df.columns:
    llm_predicted_class = []
    num_requests = 0
    for sample, headline in df[['Content', 'Headline']].values:
        response = llm.llm_generate(prompt + "giveen the Headline: " + headline + " and the Content: " + sample + " classify the article :")
        llm_predicted_class.append(response)
        num_requests += 1
        if num_requests % 10 == 0:
            print(f"processed {(num_requests/len(df)) * 100:.1f}%, paussing for 60 seconds")
            time.sleep(60)

        df['llm_predicted_class'] = pd.Series(llm_predicted_class + [float('nan')] * (len(df) - len(llm_predicted_class)))
        df.to_csv("/home/ahmed-hereiz/self/doc-discriminator/predict-stock-price/data/agent_data/scrapped_news_enriched_with_llm_predicted_class.csv", index=False)
else:
    llm_predicted_class = []
    num_requests = 0
    for i, (sample, headline) in enumerate(df[['Content', 'Headline']].values):
        if pd.isna(df.at[i, 'llm_predicted_class']):
            response = llm.llm_generate(prompt + "giveen the Headline: " + headline + " and the Content: " + sample + " classify the article :")
            llm_predicted_class.append(response)
            num_requests += 1
            if num_requests % 10 == 0:
                print(f"processed {(num_requests/len(df)) * 100:.1f}%, paussing for 60 seconds")
                time.sleep(60)
        else:
            llm_predicted_class.append(df.at[i, 'llm_predicted_class'])

        df['llm_predicted_class'] = pd.Series(llm_predicted_class)
        df.to_csv("/home/ahmed-hereiz/self/doc-discriminator/predict-stock-price/data/agent_data/scrapped_news_enriched_with_llm_predicted_class.csv", index=False)
        