# =====================================================
# LIBRARIES
# =====================================================
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
from readability import Document as ReadabilityDoc
from urllib.parse import urlparse, parse_qs, unquote
import base64
import time
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LCDocument

# =====================================================
# ENV + CLIENT
# =====================================================
load_dotenv(r"D:\Coding\Learning\LLMs\RAG\.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =====================================================
# QUERY GENERATION
# =====================================================
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
    print("RAW LLM OUTPUT:\n", text)

    try:
        return extract_json_array(text)
    except:
        return ["LLM_OUTPUT_PARSE_ERROR", text]

def query_generation(query):

    contractions_dict = contractions.contractions_dict
    model = spacy.load("en_core_web_sm")
    query_doc = model(query)
    doc = semantic_data(query_doc)

    tokenizer = TreebankWordTokenizer()
    entities = [e[0] for e in doc["entities"]]
    noun_chunks = [c for c in doc["noun_chunks"] if len(c.split()) > 1]
    protected_phrases = set(entities + noun_chunks)

    phrase_map = {}
    masked_text = query

    for i, phrase in enumerate(protected_phrases):
        key = f"__ENT{i}__"
        phrase_map[key] = phrase.replace(" ", "_")
        masked_text = re.sub(re.escape(phrase), key, masked_text)

    sentences = sent_tokenize(masked_text)
    tokenized_query = []

    for sentence in sentences:
        temp = []
        for word in sentence.split():
            lw = word.lower()
            if lw in contractions_dict:
                temp.append(contractions_dict[lw])
            else:
                temp.append(word)

        tokens = tokenizer.tokenize(" ".join(temp))
        restored = [phrase_map.get(tok, tok) for tok in tokens]
        tokenized_query.append(restored)

    variations = generate_query_variations(query, tokenized_query, doc)
    return variations, doc["entities"]

# =====================================================
# TRUSTED DOMAINS
# =====================================================
TRUSTED_DOMAINS = [
    # ---------- GENERAL KNOWLEDGE ----------
    "wikipedia.org",
    "britannica.com",
    "scholarpedia.org",
    "worldhistory.org",
    "history.com",

    # ---------- TECH / CS / AI ----------
    "geeksforgeeks.org",
    "stackoverflow.com",
    "github.com",
    "medium.com",
    "towardsdatascience.com",
    "kaggle.com",
    "arxiv.org",
    "paperswithcode.com",
    "developer.mozilla.org",
    "w3schools.com",
    "realpython.com",
    "freecodecamp.org",
    "ibm.com",
    "microsoft.com",
    "openai.com",
    "nvidia.com",
    "huggingface.co",

    # ---------- SCIENCE / RESEARCH ----------
    "nature.com",
    "sciencedirect.com",
    "springer.com",
    "ieee.org",
    "acm.org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "who.int",
    "un.org",
    "stanford.edu",
    "mit.edu",
    "harvard.edu",
    "ox.ac.uk",
    "cam.ac.uk",

    # ---------- MEDICAL ----------
    "mayoclinic.org",
    "healthline.com",
    "medicalnewstoday.com",
    "webmd.com",
    "cdc.gov",

    # ---------- FINANCE / BUSINESS ----------
    "investopedia.com",
    "forbes.com",
    "bloomberg.com",
    "reuters.com",
    "economist.com",
    "ycombinator.com",
    "crunchbase.com",

    # ---------- EDUCATION / COURSES ----------
    "byjus.com",
    "coursera.org",
    "edx.org",
    "khanacademy.org",
    "udemy.com",
    "brilliant.org",

    # ---------- NEWS ----------
    "bbc.com",
    "cnn.com",
    "nytimes.com",
    "theguardian.com",
    "aljazeera.com",

    # ---------- GOVERNMENT / LEGAL ----------
    ".gov",
    ".edu",
    "india.gov.in",
    "europa.eu",
    "whitehouse.gov",

    # ---------- MATH / ENGINEERING ----------
    "wolfram.com",
    "mathworld.wolfram.com",
    "desmos.com",
    "engineeringtoolbox.com"
]

HEADERS = {"User-Agent": "Mozilla/5.0"}

# =====================================================
# HELPERS
# =====================================================
def normalize_url(url):
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except:
        return url

def is_trusted_domain(url):
    try:
        domain = urlparse(url).netloc.lower()
        return any(td in domain for td in TRUSTED_DOMAINS)
    except:
        return False

def decode_bing_url(url):
    try:
        parsed = urlparse(url)
        if "bing.com" not in parsed.netloc:
            return url
        qs = parse_qs(parsed.query)
        if "u" not in qs:
            return url
        encoded = unquote(qs["u"][0])
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        encoded += "=" * (-len(encoded) % 4)
        decoded = base64.b64decode(encoded).decode("utf-8", errors="ignore")
        if decoded.startswith("http"):
            return decoded
    except:
        pass
    return url

# =====================================================
# SCORING
# =====================================================
def is_english(text):
    ascii_ratio = sum(c.isascii() for c in text[:500]) / max(len(text[:500]),1)
    return ascii_ratio > 0.85

def keyword_match(text, keywords):
    text = text.lower()
    return any(k.lower() in text for k in keywords)

def score_page(url, text, keywords):
    score = 0
    if is_trusted_domain(url): score += 5
    if keyword_match(text, keywords): score += 3
    if len(text) > 1000: score += 2
    if is_english(text): score += 2
    if len(text) < 100: score -= 3
    return score

# =====================================================
# SEARCH
# =====================================================
def search_urls(query, max_results=5):
    urls = []
    r = requests.get("https://www.bing.com/search",
                     headers=HEADERS,
                     params={"q": f"{query}"},
                     timeout=10)
    soup = BeautifulSoup(r.text, "lxml")
    for a in soup.select("li.b_algo h2 a"):
        url = decode_bing_url(a.get("href"))
        print(f"URL --> {url}")
        if url.startswith("http") and is_trusted_domain(url):
            urls.append(url)
        if len(urls) >= max_results:
            break
    return urls

# =====================================================
# FETCH + EXTRACT
# =====================================================
def fetch_html(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.text
    except:
        return None

def extract_main_text(html):
    try:
        doc = ReadabilityDoc(html)
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "lxml")
        text = soup.get_text(separator=" ")

        # FALLBACK IF READABILITY FAILS
        if len(text.strip()) < 200:
            soup = BeautifulSoup(html, "lxml")

            # remove junk
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()

            text = soup.get_text(separator=" ")

        return " ".join(text.split())

    except Exception as e:
        print("Extract Error:", e)
        return ""


# =====================================================
# RETRIEVAL ENGINE
# =====================================================
def retrieve_content_json(queries, keywords, score_limit=3, required_docs=5):

    seen_urls = set()
    documents = []

    for query in queries:
        if len(documents) >= required_docs:
            break

        urls = search_urls(query)

        for url in urls:
            if len(documents) >= required_docs:
                break

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
            if page_score < score_limit:
                continue

            documents.append({
                "query": query,
                "url": nurl,
                "score": page_score,
                "content": text[:5000]
            })
            print("QUERY:", query)
            print("URL:", url)
            print("TEXT LEN:", len(text))
            print("SCORE:", page_score)
            time.sleep(1)

    return documents

# =====================================================
# CHUNKING
# =====================================================
def json_to_chunks(data, chunk_size=120, min_chunk_words=15):
    all_chunks = []
    for doc in data:
        words = doc["content"].split()
        for i in range(0, len(words), chunk_size):
            chunk = words[i:i+chunk_size]
            if len(chunk) >= min_chunk_words:
                all_chunks.append(" ".join(chunk))
    return all_chunks

# =====================================================
# MASTER FUNCTION
# =====================================================
def web_query(query, faiss_path):

    status = {}

    # Query generation
    queries, entities = query_generation(query)
    keywords = [e[0] for e in entities] or query.lower().split()

    docs = retrieve_content_json(queries, keywords)
    chunks = json_to_chunks(docs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    docs_lc = [LCDocument(page_content=c) for c in chunks]
    vector_db = FAISS.from_documents(docs_lc, embeddings)
    vector_db.save_local(faiss_path)

    vector_db = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vector_db.similarity_search(query, k=3)

    status["results"] = [r.page_content for r in results]
    status["chunks"] = len(chunks)
    status["documents"] = len(docs)

    return status

# =====================================================
# TEST RUN
# =====================================================
if __name__ == "__main__":
    FAISS_PATH = r"D:\Coding\Learning\LLMs\RAG\faiss_db_web"
    result = web_query("What is github?", FAISS_PATH)
    print(json.dumps(result, indent=2))
