from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import RedirectResponse
from bs4 import BeautifulSoup
from datetime import datetime
from pyarabic.araby import strip_tashkeel, normalize_hamza
from thefuzz import fuzz, process
import requests
import pandas as pd
import numpy as np
import csv
import time
from typing import List, Dict, Tuple
from pydantic import BaseModel
import uvicorn
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential

app = FastAPI(
    title="Stock Market Headlines Scraper",
    description="An API for scraping and analyzing stock market headlines from Alborsa News",
    version="1.0.0"
)

# Headers for web scraping
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_url(url: str) -> requests.Response:
    return requests.get(url, headers=headers, timeout=10)

def get_date(link: str) -> str:
    date_parts = link.split("/")[3:6]
    return datetime.strptime("/".join(date_parts), "%Y/%m/%d").strftime("%d/%m/%Y")

def get_article_content(article_url: str) -> str:
    try:
        response = fetch_url(article_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join(p.get_text(strip=True) for p in soup.select('div.entry-content p'))
    except Exception as e:
        print(f"Failed to scrape {article_url}: {str(e)}")
        return ""

def get_headlines(page_url: str) -> List[Tuple[str, str, str]]:
    try:
        response = fetch_url(page_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return [
            (get_date(link['href']), link['aria-label'].split('Read article:')[1].strip(), link['href'])
            for div in soup.select('div.thumbnail-container.animate-lazy.size-715')
            if (link := div.find_parent('a', {'aria-label': True}))
        ]
    except Exception as e:
        print(f"Error on {page_url}: {str(e)}")
        return []

def scrape_page(page_num: int, base_url: str) -> List[List[str]]:
    page_url = base_url.format(page_num)
    print(f"Scraping: {page_url}")
    articles = get_headlines(page_url)
    return [[date, title, get_article_content(url)] for date, title, url in articles]

class MappingFiles(BaseModel):
    egx_listings_path: str
    egx30_path: str

@app.post("/scrape-headlines/")
async def scrape_headlines(start_page: int = 2, end_page: int = 10):
    try:
        base_url = "https://www.alborsaanews.com/category/%d8%a7%d9%84%d8%a8%d9%88%d8%b1%d8%b5%d8%a9-%d9%88%d8%a7%d9%84%d8%b4%d8%b1%d9%83%d8%a7%d8%aa/page/{}/"
        all_headlines = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(scrape_page, page_num, base_url) 
                      for page_num in range(start_page, end_page + 1)]
            
            for future in as_completed(futures):
                try:
                    all_headlines.extend(future.result())
                except Exception as e:
                    print(f"Error processing page: {str(e)}")

        with open('alborsa_headlines.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Headline', 'Content'])
            writer.writerows(all_headlines)

        return {"status": "success", "headlines_count": len(all_headlines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/map-headlines/")
async def map_headlines(
    similarity_threshold: float = 80.0
):
    try:
        mapping_df = pd.read_csv('assets/egx_listings.csv').merge(
            pd.read_excel('assets/egx30.xlsx')
        )
        headlines_df = pd.read_csv('alborsa_headlines.csv')

        company_list = mapping_df['name_ar'].tolist()

        def find_companies(row):
            text = f"{row['Headline']} {row['Content']}"
            matches = process.extractBests(
                text, company_list, scorer=fuzz.token_set_ratio,
                score_cutoff=similarity_threshold, limit=1
            )
            return matches[0][0] if matches else pd.NA

        headlines_df['matched_company'] = headlines_df['Headline'].apply(find_companies)

        return {"status": "success", "mapped_headlines": headlines_df.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)