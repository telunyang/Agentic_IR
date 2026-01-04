# from googlesearch import search
from playwright.sync_api import sync_playwright
from time import time
import json, os, re, sys
from typing import Annotated
from pprint import pprint
from urllib.parse import unquote
from bs4 import BeautifulSoup as bs
import requests as req
from modules.Model import generate
from modules.Prompts import get_triplets_prompt, get_summarization_prompt
from dotenv import load_dotenv
load_dotenv(override=True)

# Env variables for Search API
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")


# Get search results
def get_search_results(
        user_intent: Annotated[str, "user intent"], 
        hop_type: Annotated[str, "FORMAT: single-hop or multi-hop"], 
        user_query: Annotated[str, "The original question posed by the user. Do not modify the content of user_query under any circumstances."], 
        query: Annotated[str | list[str], "Return the single-hop question, or if the query is multi-hop, return the sub-questions. Strip all multiple-choice option markers (e.g., (A), (B), (C), (D), …) and any associated option formatting. Keep only the raw question sentence(s). Do not rewrite, paraphrase, or otherwise modify the question text beyond removing the option markers."],
        num_results: Annotated[int, "number of search results"],
        model_name: Annotated[str, "model name for generation"]
    ) -> str:

    # Initialize return data
    json_string = ''

    # Calculate execution time
    t1 = time()

    try:
        # Get user intent, hop_type, user question, query
        d_info = {
            "user_intent": user_intent,
            "hop_type": hop_type,
            "user_query": user_query,
            "query": query,
            "model_name": model_name,
            "search_results": [],
            "proper_knowledge": [],
        }

        # Set date range (optional)
        # date_range = "after:2025-05-31 before:2026-01-01 "
        date_range = ""

        '''
        Determine the type of query:
        - If it is a list, it represents multi-hop sub-questions
        - If it is a string, it represents a single-hop question
        '''
        if isinstance(query, list): 
            sentence = ""
            last_q = ""

            for index, q in enumerate(query):
                if sentence != "":
                    q = f"{q}     {last_q}{sentence}"
                    sentence = ""

                # Temporarily store the previous query; if it is multi-hop, background knowledge should be included for the search
                last_q = q + ' '
                
                # Search
                li_context = run_web_search(date_range + q.strip(), num_results, "us", model_name)

                # Organize paragraphs after summarization into a list for easy re-ranking
                li_sentences = []
                li_urls = []
                for d in li_context:
                    # 取得摘要
                    summaries = d['summaries']
                    for summary in summaries:
                        li_sentences.append(summary.strip())
                        li_urls.append(d['url'])
                
                # Re-ranking, get the most relevant document or sentence to the query
                ranks = req.post('http://127.0.0.1:5004/rerank', json={"q": q, "li_sentences": li_sentences, "li_urls": li_urls}).json()['ranks']

                # Based on the previously customized query structure characteristics, get the main query 
                q = q.split("     ")[0]

                # Record search results
                d_info["search_results"].append(li_context)
                d_proper_knowledge = {
                    "q": last_q.strip(),
                    "references": ranks[:5],
                }

                # Use the top-ranked sentence from the re-ranker as background knowledge for the next sub-question
                sentence = ranks[0]['text']
                
                # Draw knowledge graph
                kg_prompt = get_triplets_prompt(d_proper_knowledge)

                # Your triples data
                str_triplets = generate(kg_prompt, model_name)
                triplets = re.search(r"\[(.|\s)+\]", str_triplets)[0]
                triplets = eval(triplets)

                # Add the returned data to the triples information
                d_proper_knowledge['triplets'] = triplets
                d_info["proper_knowledge"].append(d_proper_knowledge)

        elif isinstance(query, str):
            # Search
            li_context = run_web_search(date_range + query, num_results, "us", model_name)

            # Organize paragraphs after summarization into a list for easy re-ranking
            li_sentences = []
            li_urls = []
            for d in li_context:
                # Get summaries
                summaries = d['summaries']
                for summary in summaries:
                    li_sentences.append(summary.strip())
                    li_urls.append(d['url'])

            # Re-ranking, get the most relevant document or sentence to the query
            ranks = req.post('http://127.0.0.1:5004/rerank', json={"q": user_query, "li_sentences": li_sentences, "li_urls": li_urls}).json()['ranks']

            # Organize the data needed for return
            d_info["search_results"].append(li_context)
            d_proper_knowledge = {
                "q": user_query,
                "references": ranks,
            }
            
            # Draw knowledge graph
            kg_prompt = get_triplets_prompt(d_proper_knowledge)

            # Your triples data
            str_triplets = generate(kg_prompt, model_name)
            triplets = re.search(r"\[(.|\s)+\]", str_triplets)[0]
            triplets = eval(triplets)

            # Add the returned data to the triples information
            d_proper_knowledge['triplets'] = triplets
            d_info["proper_knowledge"].append(d_proper_knowledge)

        # Convert the organized results data to JSON format
        json_string = json.dumps(d_info, ensure_ascii=False, indent=None)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    # Calculate program execution time
    t2 = time()

    print(f"Program execution time: {t2 - t1} seconds")

    return json_string



# Search engine
def run_web_search(q: str, num_results: int, lang: str, model_name: str) -> list:
    try:
        # Place search results
        li_context = []
        urls = []

        # Use Google Custom Search API to get search results
        search_engine_id = SEARCH_ENGINE_ID
        api_key = SEARCH_API_KEY
        limit = num_results
        pagination = []
        for start in range(1, limit + 1, 10):
            pagination.append(start)

        # Get search results based on pagination
        for start in pagination:
            web_api_prefix =f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&num={min(10, limit - start + 1)}&start={start}"
            res = req.get(f"{web_api_prefix}&q={q}")
            data = res.json()
            urls.extend( [item['link'] for item in data['items']] )

        # Use playwright to get web page content
        with sync_playwright() as playwright:
            # Set up browser
            pw = playwright.chromium # "chromium" or "firefox" or "webkit".
            browser = pw.launch(headless=True, args=["--start-maximized"])
            context = browser.new_context(accept_downloads=False, no_viewport=True)
            page = context.new_page()
        
            # Visit each link
            for url in urls:
                # Go to page
                page.goto(url, timeout=15*1000)
                
                print("=" * 50)
                print("Query:", q)
                print(f"Getting content from {unquote(url)}...")

                if url.lower().endswith(".pdf") or 'arxiv' in url.lower():
                    # m = hashlib.md5()
                    # m.update(url.encode('utf-8'))
                    # file_name = m.hexdigest()
                    # file_path = f"./tmp/{file_name}.pdf"
                    # if not os.path.exists(file_path):
                    #     print('Downloading PDF file...')
                    #     wget.download(url, file_path)
                    #     print('Download complete!')
                    # else:
                    #     print('PDF file already exists, skipping download.')
                    # try:
                    #     doc = pymupdf.open(file_path)
                    #     context = ''
                    #     for p in doc:
                    #         context += p.get_text()
                    #     print()
                    # except Exception as e:
                    #     raise Exception(f"Cannot read PDF file {url} -> {file_path}: {e}")

                    print(f"Skipping PDF file: {url}")
                    continue
                else:
                    # Get HTML elements
                    html = page.content()

                    # Use BeautifulSoup to parse HTML
                    soup = bs(html, "lxml")

                    # Get web page content
                    context = soup.get_text(strip=True)
                    # context = html
                    context = re.sub(r"\s+", " ", context)

                # Based on the previously customized query structure characteristics, get the main query 
                q = q.split("     ")[0]

                # Get the prompt for summarization
                user_prompt = get_summarization_prompt(q, context)

                # Generate content
                generated_text = generate(user_prompt, model_name) # 'gemini-2.5-flash'
                generated_text = generated_text.replace("\n", "")
                generated_text = json.loads(re.search(r"\[.*?\]", generated_text)[0])

                # Organize the data needed for return
                d = {
                    "q": q,
                    "url": unquote(url),
                    "summaries": generated_text
                }
                li_context.append(d)
                print("Knowledge obtained:")
                pprint(d)

            # Close the browser
            browser.close()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # Error event name
        print(exc_type, fname, exc_tb.tb_lineno)

        # Error code location
        print(f"Error in Search: {e}")

    return li_context


