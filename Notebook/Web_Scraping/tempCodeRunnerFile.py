### Libraries import
import contractions
from nltk import TreebankWordTokenizer, sent_tokenize
import re
import spacy
import os
from groq import Groq
import json
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from readability import Document
from urllib.parse import urlparse, parse_qs, unquote
import base64
import time
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_community.vectorstores import FAISS
from pathlib import Path

load_dotenv(r"D:\Coding\Learning\LLMs\RAG\.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_query_variations(original_query, tokenized_query, semantic_data):

    system_prompt = """
You are a Query Reformulation Engine.

OUTPUT RULES:
- Output ONLY a valid JSON array of exactly 5 strings.
- No explanations.
- No markdown.
- No backticks.
- Preserve named entities exactly.
- Maintain original intent.
- Do NOT introduce new topics.
- Each query must be <= 20 words.
- Make sure to divide the inten
- If the original query contains multiple distinct questions or intents that cannot be naturally combined into one clear query, split them into separate focused queries instead of forcing a single long sentence.
- However, still return exactly 5 total queries.
- Each query must remain semantically equivalent to a part of the original intent.
- Since this is for web scraping, make sure the queries you make are alligned to search in web and gain the best results.
"""

    user_prompt = f"""
Original Query: {original_query}

Semantic Anchors:
{json.dumps(semantic_data)}

Task: Generate 5 equivalent search queries.
Return ONLY JSON array.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0.25,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    text = response.choices[0].message.content.strip()
    # print("RAW LLM OUTPUT:\n", text)

    try:
        return extract_json_array(text)
    except:
        return ["LLM_OUTPUT_PARSE_ERROR", text]

def semantic_data(sent1_doc):    
    return {
    "entities": [(ent.text, ent.label_) for ent in sent1_doc.ents],
    "noun_chunks": [chunk.text for chunk in sent1_doc.noun_chunks],
    "pos_tags": [(token.text, token.pos_) for token in sent1_doc],
    "root_verbs": [token.text for token in sent1_doc if token.head == token],
    }
    
def extract_json_array(text):
    match = re.search(r'\[.*\]', text, re.S)
    if match:
        return json.loads(match.group())
    return []

def query_generation(query):
    """
    takes in query (str), cleans it, process it, pass it to llm, and produces 5 variations, using those 5 queries, web search is done and html data is retrieved. This data will be processed and then converted to 384-dimensioned embeddings.    
    :param query: Description
    """
    
    ### Contractions dictionary
    contractions_dict = contractions.contractions_dict
    contractions_dict
    
    ### Spacy text to doc
    model = spacy.load("en_core_web_sm")
    query_doc = model(query)
    doc = semantic_data(query_doc)
    
    # --- Step 1: Prepare protected phrases ---
    tokenizer = TreebankWordTokenizer()
    entities = [e[0] for e in doc["entities"]]
    noun_chunks = [c for c in doc["noun_chunks"] if len(c.split()) > 1]

    protected_phrases = set(entities + noun_chunks)

    # --- Step 2: Mask phrases ---
    phrase_map = {}
    masked_text = query

    for i, phrase in enumerate(protected_phrases):
        key = f"__ENT{i}__"
        phrase_map[key] = phrase.replace(" ", "_")  # join words
        masked_text = re.sub(re.escape(phrase), key, masked_text)

    # --- Step 3: Sentence split ---
    sentences = sent_tokenize(masked_text)

    tokenized_query = []

    for sentence in sentences:
        temp = []

        # --- Step 4: Contraction expand ---
        for word in sentence.split():
            lw = word.lower()
            if lw in contractions_dict:
                temp.append(contractions_dict[lw])
            else:
                temp.append(word)

        # --- Step 5: Tokenize ---
        tokens = tokenizer.tokenize(" ".join(temp))

        # --- Step 6: Restore phrases ---
        restored = [phrase_map.get(tok, tok) for tok in tokens]

        tokenized_query.append(restored)

    variations = generate_query_variations(
        original_query=query,
        tokenized_query=tokenized_query,
        semantic_data=doc
    )   
    return variations, doc["entities"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

TRUSTED_DOMAINS = [
    "wikipedia.org",
    "ibm.com",
    "google.com",
    "microsoft.com",
    "openai.com",
    "geeksforgeeks.org",
    "towardsdatascience.com",
    "kaggle.com",
    "medium.com",
    "arxiv.org",
    "stanford.edu",
    "mit.edu",
    "britannica.com",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "sciencedirect.com",
    "biologydictionary.net",
    "healthline.com",
    "medicalnewstoday.com",
    "byjus.com"
]

# ---------- URL HELPERS ----------
def normalize_url(url):
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except:
        return url

def is_trusted_domain(url):
    domain = urlparse(url).netloc
    return any(td in domain for td in TRUSTED_DOMAINS)

def decode_bing_url(url):
    try:
        parsed = urlparse(url)

        if "bing.com" not in parsed.netloc:
            return url

        qs = parse_qs(parsed.query)
        if "u" not in qs:
            return url

        encoded = qs["u"][0]
        encoded = unquote(encoded)

        if encoded.startswith("a1"):
            encoded = encoded[2:]

        encoded += "=" * (-len(encoded) % 4)

        decoded = base64.b64decode(encoded).decode("utf-8", errors="ignore")

        if decoded.startswith("http"):
            return decoded

    except:
        pass

    return url

# ---------- SEARCH ----------
def search_urls(query, max_results=5):
    urls = []
    try:
        r = requests.get(
            "https://www.bing.com/search",
            headers=HEADERS,
            params={"q": query},
            timeout=10
        )

        soup = BeautifulSoup(r.text, "lxml")

        for a in soup.select("li.b_algo h2 a"):
            link = a.get("href")
            if not link:
                continue
            link = decode_bing_url(link)
            if link and link.startswith("http"):
                urls.append(link)
            if len(urls) >= max_results:
                break

    except Exception as e:
        print("Search error:", e)

    return urls

# ---------- FETCH ----------
def fetch_html(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.text
    except:
        pass
    return None

# ---------- EXTRACT ----------
def extract_main_text(html):
    try:
        doc = Document(html)
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        return " ".join(text.split())
    except:
        return ""

# ---------- VALIDATION ----------
def is_english(text):
    return not any(ord(c) > 127 for c in text[:300])

def keyword_match(text, keywords):
    text = text.lower()
    return any(k.lower() in text for k in keywords)

def score_page(url, text, keywords):
    score = 0

    if is_trusted_domain(url):
        score += 5
    if keyword_match(text, keywords):
        score += 3
    if len(text) > 1000:
        score += 2
    if is_english(text):
        score += 2
    if len(text) < 300:
        score -= 5

    return score

# ---------- MAIN ENGINE ----------
def retrieve_content_json(queries, keywords, results_per_query=3, max_chars=5000):

    seen_urls = set()
    documents = []

    for query in queries:
        print(f"\nSearching: {query}")
        urls = search_urls(query, results_per_query)

        for url in urls:
            nurl = normalize_url(url)
            if nurl in seen_urls:
                continue

            seen_urls.add(nurl)

            html = fetch_html(nurl)
            if not html:
                continue

            text = extract_main_text(html)
            if not text:
                continue

            page_score = score_page(nurl, text, keywords)
            print("SCORE:", page_score, "URL:", url)

            if page_score < 3:
                continue

            documents.append({
                "query": query,
                "url": nurl,
                "score": page_score,
                "content": text[:max_chars]
            })

            time.sleep(1)

    return json.dumps(documents, indent=2)
query = ""
queries, entity = query_generation(query)
keywords = []
for ent in entity:
    keywords.append(ent[0])
if not keywords:
    keywords = query.lower().split()
result_json = retrieve_content_json(queries=queries, keywords=keywords)

def json_to_chunks(
    json_data,
    chunk_size=200,
    min_chunk_words=30
):
    """
    Converts scraped JSON documents into cleaned word chunks.

    Args:
        json_data (str | list): JSON string or parsed list of dicts
        chunk_size (int): words per chunk
        min_chunk_words (int): discard chunks smaller than this

    Returns:
        List[str]: cleaned chunks
    """

    # ---------------------------
    # LOAD JSON
    # ---------------------------
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    all_chunks = []

    # ---------------------------
    # CLEAN FUNCTION
    # ---------------------------
    def clean_text(text):
        text = text.lower()

        # remove weird unicode / emojis
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # keep periods but remove other punctuation
        text = re.sub(r'[^\w\s\.]', ' ', text)

        # collapse spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    # ---------------------------
    # PROCESS EACH DOCUMENT
    # ---------------------------
    for doc in data:
        content = doc.get("content", "")
        if not content:
            continue

        cleaned = clean_text(content)
        words = cleaned.split()

        # ---------------------------
        # CHUNKING
        # ---------------------------
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < min_chunk_words:
                continue

            chunk = " ".join(chunk_words).strip()
            all_chunks.append(chunk)

    return all_chunks

chunks = json_to_chunks(result_json)
print(f"Length of chunks : {len(chunks)}")

### EMBEDDING MODEL
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 384 dimensions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": False}
)

print("Embedding model loaded successfully!")

FAISS_PATH = r"D:\Coding\Learning\LLMs\RAG\faiss_db_web"

print("\nBuilding FAISS Vector Database...")

# Create FAISS DB from documents
vector_db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save DB locally
vector_db.save_local(FAISS_PATH)

print("FAISS DB Built Successfully!")
print(f"Saved at: {FAISS_PATH}")
print(f"Total Chunks Stored: {len(chunks)}")
print("\nLoading FAISS Database...")

vector_db = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True   # required
)

print("FAISS Loaded Successfully!")
results = vector_db.similarity_search(query, k=3)

for i, text in enumerate(results, 1):
    print(f"\nResult {i}")
    print(text)
    