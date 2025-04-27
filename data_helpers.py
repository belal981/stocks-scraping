import pandas as pd
import numpy as np
lpersfrom thefuzz import fuzz, process
from typing import List, Union, Dict, Any # Added Dict, Any
import json # Added json
import time # Added time
import os # Added os
# Add imports for your LLM library, e.g., google.generativeai
# import google.generativeai as genai

# --- SimpleInvokeLLM Class (from simple_llm.py) ---
class SimpleInvokeLLM:
    """
    A simple wrapper class to interact with a Generative Language Model.
    (Structure needs to be adapted based on your actual simple_llm.py)
    """
    def __init__(self, model: str, api_key: str, temperature: float = 0.0):
        """
        Initializes the LLM client.

        Args:
            model (str): The name of the model to use.
            api_key (str): The API key for the LLM service.
            temperature (float): The sampling temperature for generation.
        """
        self.model_name = model
        self.api_key = api_key
        self.temperature = temperature
        print(f"Placeholder: LLM Initialized with model: {self.model_name}") # Placeholder

    def llm_generate(self, prompt: str) -> str:
        """
        Generates text based on the provided prompt using the LLM.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: The generated text response from the LLM.

        Raises:
            Exception: If the LLM generation fails.
        """
        print(f"Placeholder: Generating response for prompt (first 50 chars): {prompt[:50]}...") # Placeholder
        return "Placeholder Category" # Placeholder response

# --- Helper Functions ---
def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Loads configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file. Defaults to 'config.json'.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file is not found.
        json.JSONDecodeError: If the config file is not valid JSON.
    """
    config_abs_path = os.path.abspath(config_path) # Ensure absolute path for clarity
    if not os.path.exists(config_abs_path):
        raise FileNotFoundError(f"Configuration file not found at {config_abs_path}")
    try:
        with open(config_abs_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        return config
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON from {config_abs_path}: {e.msg}", e.doc, e.pos)

def fetch_company_list(mapping_file_path: str = r'assets\mapping.csv') -> List[str]:
    """
    Fetches the list of company names from the specified mapping file.

    Args:
        mapping_file_path (str): Relative or absolute path to the mapping CSV file.
                                 Defaults to r'assets\mapping.csv'.

    Returns:
        List[str]: A list of Arabic company names from the mapping file.

    Raises:
        FileNotFoundError: If the mapping file is not found.
        KeyError: If the 'company_ar' column is missing.
    """
    abs_mapping_path = os.path.abspath(mapping_file_path)
    if not os.path.exists(abs_mapping_path):
        raise FileNotFoundError(f"Mapping file not found at {abs_mapping_path}")
    try:
        mapping_df = pd.read_csv(abs_mapping_path)
        if 'company_ar' not in mapping_df.columns:
            raise KeyError(f"'company_ar' column not found in {abs_mapping_path}")
        company_list = mapping_df['company_ar'].tolist()
        return company_list
    except Exception as e:
        print(f"Error reading mapping file {abs_mapping_path}: {e}")
        raise

def find_companies(row: pd.Series, company_list: List[str],
                  threshold: float = 90.0, content_col: str = 'Content') -> Union[str, float]:
    """
    Finds matching companies in the text content using fuzzy matching.

    Args:
        row (pd.Series): DataFrame row containing the content column.
        company_list (List[str]): List of company names to match against.
        threshold (float, optional): Minimum similarity score. Defaults to 90.0.
        content_col (str, optional): Name of the column containing the text content.
                                     Defaults to 'Content'.

    Returns:
        Union[str, float]: Matched company name or np.nan if no match found or content is invalid.
    """
    if content_col not in row or pd.isna(row[content_col]):
        return np.nan # Return NaN if content column is missing or NaN

    text = str(row[content_col])
    if not text.strip():
        return np.nan # Return NaN if content is empty string

    try:
        matches = process.extractBests(
            text,
            company_list,
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold,
            limit=1
        )
        return matches[0][0] if matches else np.nan
    except Exception as e:
        print(f"Error during fuzzy matching for row: {e}")
        return np.nan # Return NaN on unexpected error during matching

DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE = """
You are an AI agent. Your task is to classify the article into one of the following categories based on its Headline and Content:
{categories}

Classify the article strictly into ONE category from the list above. Output only the category name.

Article Headline: {headline}
Article Content: {content}

Category:"""

DEFAULT_CATEGORIES = [
    'Real Estate', 'Education Services', 'Health Care & Pharmaceuticals',
    'Basic Resources', 'Food, Beverages and Tobacco', 'Building Materials',
    'Shipping & Transportation Services', 'Energy & Support Services',
    'Non-bank financial services', 'Banks', 'Trade & Distributors',
    'Contracting & Construction Engineering', 'Textile & Durables',
    'Industrial Goods , Services and Automobiles',
    'IT , Media & Communication Services', 'Travel & Leisure', 'Utilities'
]

def classify_articles_with_llm(
    input_csv_path: str,
    output_csv_path: str,
    config_path: str = 'config.json',
    content_col: str = 'Content',
    headline_col: str = 'Headline',
    output_col: str = 'llm_predicted_class',
    prompt_template: str = DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE,
    categories: List[str] = None,
    requests_per_minute: int = 10, # Adjust based on API limits
    save_interval: int = 50 # Save progress every N rows
) -> None:
    """
    Classifies articles in a CSV file using an LLM based on content and headline.

    Reads articles from an input CSV, uses an LLM to classify each article into
    predefined categories, handles rate limiting, supports resuming processing,
    and saves the results to an output CSV.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the output CSV file with classifications.
        config_path (str): Path to the JSON configuration file containing 'model' and 'google_api_key'.
        content_col (str): Name of the column containing the article content.
        headline_col (str): Name of the column containing the article headline.
        output_col (str): Name of the column to store the LLM's predicted class.
        prompt_template (str): A format string for the LLM prompt. Must include
                               {categories}, {headline}, and {content} placeholders.
        categories (List[str], optional): List of classification categories.
                                          Defaults to DEFAULT_CATEGORIES.
        requests_per_minute (int): Maximum number of LLM requests per minute.
        save_interval (int): Save progress every N processed rows.
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES

    # --- Input/Output Path Handling ---
    abs_input_path = os.path.abspath(input_csv_path)
    abs_output_path = os.path.abspath(output_csv_path)
    abs_config_path = os.path.abspath(config_path)

    if not os.path.exists(abs_input_path):
        print(f"Error: Input CSV file not found at '{abs_input_path}'")
        return

    try:
        config = load_config(abs_config_path)
        # Ensure required keys are in config
        if 'model' not in config or 'google_api_key' not in config:
             raise KeyError("Config must contain 'model' and 'google_api_key'")
        llm = SimpleInvokeLLM(model=config['model'], api_key=config['google_api_key'], temperature=0.0)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error initializing LLM or loading config: {e}")
        return # Stop execution if LLM setup fails

    try:
        if os.path.exists(abs_output_path):
            print(f"Output file '{abs_output_path}' found. Attempting to resume.")
            df = pd.read_csv(abs_output_path)
            # Check if essential columns exist from input
            if content_col not in df.columns or headline_col not in df.columns:
                 print(f"Warning: Essential columns ('{content_col}', '{headline_col}') missing in existing output file. Reloading from input.")
                 df = pd.read_csv(abs_input_path)
                 df[output_col] = pd.NA
            elif output_col not in df.columns:
                print(f"Output column '{output_col}' not found in existing file. Adding it.")
                df[output_col] = pd.NA
        else:
            df = pd.read_csv(abs_input_path)
            df[output_col] = pd.NA # Initialize output column for new processing

        if content_col not in df.columns or headline_col not in df.columns:
            print(f"Error: Input CSV must contain '{content_col}' and '{headline_col}' columns.")
            return

    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{abs_input_path}' is empty.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    num_requests_made_this_minute = 0
    minute_start_time = time.time()
    sleep_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0

    category_str = "\n".join([f"- {cat}" for cat in categories])

    total_rows = len(df)
    processed_since_last_save = 0
    start_index = df[output_col].isna().idxmax() if df[output_col].isna().any() else total_rows
    print(f"Starting/Resuming processing from index {start_index}...")

    for i in range(start_index, total_rows):
        row = df.iloc[i]

        current_time = time.time()
        if current_time - minute_start_time >= 60.0:
            minute_start_time = current_time
            num_requests_made_this_minute = 0

        if requests_per_minute > 0 and num_requests_made_this_minute >= requests_per_minute:
            time_to_wait = 60.0 - (current_time - minute_start_time)
            if time_to_wait > 0:
                print(f"Rate limit reached ({requests_per_minute}/min). Pausing for {time_to_wait:.2f} seconds.")
                time.sleep(time_to_wait)
                minute_start_time = time.time() # Reset timer after waiting
                num_requests_made_this_minute = 0

        content = str(row[content_col]) if pd.notna(row[content_col]) else ""
        headline = str(row[headline_col]) if pd.notna(row[headline_col]) else ""

        if not content.strip() and not headline.strip():
             print(f"Row {i+1}/{total_rows}: Skipping due to empty headline and content.")
             df.loc[df.index[i], output_col] = "SKIPPED_EMPTY"
             processed_since_last_save += 1
             continue

        prompt = prompt_template.format(
            categories=category_str,
            headline=headline,
            content=content
        )

        try:
            response = llm.llm_generate(prompt)
            cleaned_response = response.strip() if response else "LLM_EMPTY_RESPONSE"
            df.loc[df.index[i], output_col] = cleaned_response
            num_requests_made_this_minute += 1
            processed_since_last_save += 1
            print(f"Processed row {i+1}/{total_rows}. Category: {cleaned_response}")
        except Exception as e:
            print(f"Error processing row {i+1}: {e}. Marking as LLM_ERROR.")
            df.loc[df.index[i], output_col] = "LLM_ERROR"
            processed_since_last_save += 1
            time.sleep(2)

        if processed_since_last_save >= save_interval:
            print(f"Saving progress ({processed_since_last_save} rows processed since last save)...")
            try:
                 df.to_csv(abs_output_path, index=False, encoding='utf-8')
                 processed_since_last_save = 0 # Reset counter
            except Exception as e:
                 print(f"Error saving progress to {abs_output_path}: {e}")

    print("Classification complete. Saving final results...")
    try:
        df.to_csv(abs_output_path, index=False, encoding='utf-8')
        print(f"Output successfully saved to {abs_output_path}")
    except Exception as e:
        print(f"Error saving final results to {abs_output_path}: {e}")

# --- Example Usage Placeholder (Optional) ---
#     input_file = r'data\input_articles.csv'
#     output_file = r'data\output_articles_classified.csv'
#     config_file = 'config.json'
#
#     print(f"Starting classification for {input_file}...")
#     classify_articles_with_llm(
#         input_csv_path=input_file,
#         output_csv_path=output_file,
#         config_path=config_file,
#         requests_per_minute=15 # Adjust as needed
#     )
#     print("Classification process finished.")

