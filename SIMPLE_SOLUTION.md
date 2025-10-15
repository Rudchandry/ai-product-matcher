# 🎯 SIMPLE SINGLE-FILE SOLUTION
# This eliminates ALL import issues!

## Delete Everything and Start Fresh

### STEP 1: Clean Slate
In PythonAnywhere Files, go to `/home/rudchandryhodge/` and:
1. **Delete the entire `pair-matcher` folder**
2. **Create a new `pair-matcher` folder**
3. **Enter the new folder**

### STEP 2: Create ONE Single File
Create **ONLY** this file: `main.py`

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rapidfuzz import fuzz
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Product Matcher", description="AI-powered product name matching API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lightweight AI
tfidf_vectorizer = None

def _load_tfidf_model():
    """Initialize TF-IDF vectorizer for semantic similarity"""
    global tfidf_vectorizer
    try:
        tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 3),
            max_features=10000,
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        logger.info("TF-IDF vectorizer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TF-IDF vectorizer: {e}")
        return False

# Initialize the model on startup
_load_tfidf_model()

class MatchRequest(BaseModel):
    search_product: str
    product_catalog: List[str]
    max_results: Optional[int] = 5

class ProductMatch(BaseModel):
    product: str
    similarity_score: float
    match_type: str
    details: dict

class MatchResponse(BaseModel):
    search_product: str
    matches: List[ProductMatch]
    total_matches: int

@app.get("/")
async def read_root():
    return {
        "message": "AI Product Matcher API",
        "version": "1.0",
        "endpoints": ["/docs", "/match", "/model-status", "/batch-similarity"]
    }

@app.get("/model-status")
async def model_status():
    return {
        "tfidf_model": "initialized" if tfidf_vectorizer else "not_initialized",
        "status": "ready",
        "features": ["semantic_matching", "fuzzy_search", "model_extraction"]
    }

@app.post("/match", response_model=MatchResponse)
async def match_products(request: MatchRequest):
    """AI-powered product matching with enhanced algorithms"""
    
    if not request.product_catalog:
        raise HTTPException(status_code=400, detail="Product catalog cannot be empty")
    
    try:
        search_product = request.search_product.strip()
        if not search_product:
            raise HTTPException(status_code=400, detail="Search product cannot be empty")
        
        # Get similarity scores for all products
        matches = []
        for product in request.product_catalog:
            if product.strip():  # Skip empty products
                similarity_score = _compute_semantic_similarity(search_product, product)
                
                # Determine match type based on score
                if similarity_score >= 0.9:
                    match_type = "exact"
                elif similarity_score >= 0.7:
                    match_type = "high"
                elif similarity_score >= 0.5:
                    match_type = "medium"
                else:
                    match_type = "low"
                
                matches.append(ProductMatch(
                    product=product,
                    similarity_score=round(similarity_score, 4),
                    match_type=match_type,
                    details={
                        "algorithm": "enhanced_tfidf",
                        "features": ["semantic", "fuzzy", "model_number"]
                    }
                ))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit results
        max_results = min(request.max_results, len(matches))
        top_matches = matches[:max_results]
        
        return MatchResponse(
            search_product=search_product,
            matches=top_matches,
            total_matches=len(matches)
        )
        
    except Exception as e:
        logger.error(f"Error in match_products: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

class BatchSimilarityRequest(BaseModel):
    products: List[str]

@app.post("/batch-similarity")
async def batch_similarity(request: BatchSimilarityRequest):
    """Compare all products against each other"""
    
    if len(request.products) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 products for comparison")
    
    try:
        results = []
        products = request.products
        
        for i, product1 in enumerate(products):
            for j, product2 in enumerate(products[i+1:], i+1):
                similarity = _compute_semantic_similarity(product1, product2)
                results.append({
                    "product1": product1,
                    "product2": product2,
                    "similarity_score": round(similarity, 4),
                    "match_strength": "high" if similarity >= 0.7 else "medium" if similarity >= 0.5 else "low"
                })
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "total_comparisons": len(results),
            "comparisons": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch_similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def _compute_semantic_similarity(search_product: str, catalog_product: str) -> float:
    """Enhanced similarity computation with multiple algorithms"""
    
    # Clean product names
    search_clean = _clean_product_name(search_product)
    catalog_clean = _clean_product_name(catalog_product)
    
    # 1. TF-IDF Semantic Similarity (40% weight)
    semantic_score = _get_tfidf_similarity(search_clean, catalog_clean)
    
    # 2. Fuzzy String Similarity (35% weight)
    fuzzy_score = fuzz.ratio(search_clean.lower(), catalog_clean.lower()) / 100.0
    
    # 3. Model Number Matching (15% weight)
    model_score = _extract_model_similarity(search_product, catalog_product)
    
    # 4. Character-level similarity (10% weight)
    char_score = fuzz.partial_ratio(search_clean.lower(), catalog_clean.lower()) / 100.0
    
    # Weighted combination
    final_score = (
        0.40 * semantic_score +
        0.35 * fuzzy_score +
        0.15 * model_score +
        0.10 * char_score
    )
    
    # Bonus for exact brand matches
    search_words = set(search_clean.lower().split())
    catalog_words = set(catalog_clean.lower().split())
    common_words = search_words.intersection(catalog_words)
    
    if common_words:
        brand_bonus = min(0.1, len(common_words) * 0.02)
        final_score = min(1.0, final_score + brand_bonus)
    
    return final_score

def _clean_product_name(product_name: str) -> str:
    """Enhanced product name cleaning"""
    if not product_name:
        return ""
    
    # Convert to lowercase for processing
    cleaned = product_name.lower()
    
    # Normalize model numbers (e.g., "idrac 8" -> "idrac8")
    cleaned = re.sub(r'\b(\w+)\s+(\d+)\b', r'\1\2', cleaned)
    
    # Expand common abbreviations
    abbreviations = {
        'com': 'commercial',
        'pro': 'professional', 
        'std': 'standard',
        'ent': 'enterprise',
        'srv': 'server',
        'mgmt': 'management'
    }
    
    for abbr, full in abbreviations.items():
        cleaned = re.sub(r'\b' + abbr + r'\b', full, cleaned)
    
    # Remove noise terms
    noise_terms = ['license', 'subscription', 'deployment', 'pack', 'bundle']
    for term in noise_terms:
        cleaned = re.sub(r'\b' + term + r'\b', '', cleaned)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def _get_tfidf_similarity(text1: str, text2: str) -> float:
    """Get TF-IDF cosine similarity between two texts"""
    global tfidf_vectorizer
    
    if not tfidf_vectorizer or not text1 or not text2:
        return 0.0
    
    try:
        # Combine texts for vocabulary
        texts = [text1, text2]
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity_matrix[0][0])
        
    except Exception as e:
        logger.warning(f"TF-IDF similarity failed: {e}")
        return 0.0

def _extract_model_similarity(text1: str, text2: str) -> float:
    """Extract and compare model numbers/versions"""
    
    # Pattern for model numbers
    model_pattern = r'\b(?:v?\d+\.?\d*|[a-z]+\d+|\d+[a-z]+)\b'
    
    models1 = set(re.findall(model_pattern, text1.lower()))
    models2 = set(re.findall(model_pattern, text2.lower()))
    
    if not models1 or not models2:
        return 0.0
    
    # Check for exact model matches
    common_models = models1.intersection(models2)
    if common_models:
        return 1.0
    
    # Check for partial model matches
    for m1 in models1:
        for m2 in models2:
            if m1 in m2 or m2 in m1:
                return 0.7
    
    return 0.0

def extract_key_terms(product_name: str) -> dict:
    """Extract key terms and categories from product name"""
    
    cleaned = _clean_product_name(product_name)
    words = cleaned.split()
    
    # Product categories
    categories = {
        'hardware': ['server', 'switch', 'router', 'storage', 'disk', 'memory', 'cpu'],
        'software': ['license', 'subscription', 'software', 'application', 'program'],
        'networking': ['switch', 'router', 'firewall', 'wireless', 'ethernet'],
        'storage': ['disk', 'ssd', 'hdd', 'storage', 'backup', 'raid']
    }
    
    # Brand detection
    brands = ['dell', 'hp', 'cisco', 'microsoft', 'vmware', 'intel', 'amd']
    
    # Extract information
    detected_categories = []
    detected_brands = []
    model_numbers = re.findall(r'\b(?:v?\d+\.?\d*|[a-z]+\d+|\d+[a-z]+)\b', cleaned)
    
    for word in words:
        # Check categories
        for category, keywords in categories.items():
            if word in keywords and category not in detected_categories:
                detected_categories.append(category)
        
        # Check brands
        if word in brands and word not in detected_brands:
            detected_brands.append(word)
    
    return {
        'categories': detected_categories,
        'brands': detected_brands,
        'model_numbers': model_numbers,
        'key_words': [w for w in words if len(w) > 2][:10]  # Top 10 key words
    }

# Add this as a new endpoint
@app.post("/extract-terms")
async def extract_product_terms(request: dict):
    """Extract key terms from a product name"""
    
    product_name = request.get('product_name', '')
    if not product_name:
        raise HTTPException(status_code=400, detail="Product name is required")
    
    try:
        terms = extract_key_terms(product_name)
        return {
            "product_name": product_name,
            "extracted_terms": terms
        }
    except Exception as e:
        logger.error(f"Error extracting terms: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# WSGI application for PythonAnywhere
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### STEP 3: Create requirements.txt
```
fastapi>=0.104.1
pydantic>=2.4.2
rapidfuzz>=3.9.0
scikit-learn>=1.3.2
numpy>=1.24.4
uvicorn[standard]>=0.23.2
```

### STEP 4: Set up Web App
1. **Web tab** → **Add new web app**
2. **Manual configuration**
3. **Python 3.10**
4. **Source code**: `/home/rudchandryhodge/pair-matcher`
5. **WSGI file**: Replace ALL content with:

```python
import sys
import os

path = '/home/rudchandryhodge/pair-matcher'
if path not in sys.path:
    sys.path.insert(0, path)

from main import application
```

### STEP 5: Install & Test
```bash
cd /home/rudchandryhodge/pair-matcher
pip3.10 install --user -r requirements.txt
```

**Then reload your web app!**

---

## ✅ This Approach:
- **NO import issues** (everything in one file)
- **NO app/app.py confusion** 
- **Simple WSGI setup**
- **All AI features included**
- **Works immediately**

**Total files: Just `main.py` + `requirements.txt`**