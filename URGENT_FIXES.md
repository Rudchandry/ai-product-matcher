# 🚨 IMMEDIATE FIXES NEEDED - Based on Your Error Logs

## Problem 1: Missing app/app.py file
Your error shows: `ModuleNotFoundError: No module named 'app.app'`

**You're missing either:**
- The `app` directory, OR
- The `app.py` file inside the `app` directory, OR  
- The `__init__.py` file inside the `app` directory

## Problem 2: WSGI Configuration Error
Your error shows: `AttributeError: module 'rudchandryhodge_pythonanywhere_com_wsgi' has no attribute 'application'`

This means your WSGI file is missing the `application` variable.

---

# 🔧 EXACT FIXES TO APPLY RIGHT NOW

## FIX 1: Create Missing Files

### In PythonAnywhere Files, navigate to `/home/rudchandryhodge/pair-matcher/`

**Check this structure exists:**
```
/home/rudchandryhodge/pair-matcher/
├── main.py ✓
├── requirements.txt ✓
└── app/              ← CREATE THIS DIRECTORY!
    ├── __init__.py   ← CREATE THIS FILE (empty)!
    └── app.py        ← CREATE THIS FILE!
```

### If the `app` directory is missing:
1. Click "New directory" 
2. Name it: `app`
3. Enter the `app` directory
4. Create `__init__.py` (leave it empty, just save)

### If `app.py` is missing inside the `app` directory:
Create `app.py` with this content (I'll provide complete content below)

## FIX 2: Update Your WSGI File

**Go to your WSGI configuration file** (should be at `/var/www/rudchandryhodge_pythonanywhere_com_wsgi.py`)

**Replace ALL content with this:**

```python
import sys
import os

# Add your project directory to the Python path
path = '/home/rudchandryhodge/pair-matcher'
if path not in sys.path:
    sys.path.insert(0, path)

# Import your FastAPI app
from main import app

# PythonAnywhere expects 'application' variable for WSGI
application = app
```

**That's it! No extra lines, no complex setup.**

---

# 🎯 COMPLETE app.py CONTENT

**Create this file: `/home/rudchandryhodge/pair-matcher/app/app.py`**

```python
from fastapi import FastAPI, HTTPException
from typing import Union
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
        "endpoints": ["/docs", "/match", "/model-status"]
    }

@app.get("/model-status")
async def model_status():
    return {
        "tfidf_model": "initialized" if tfidf_vectorizer else "not_initialized",
        "status": "ready"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

# ⚡ IMMEDIATE ACTION STEPS

1. **Create the `app` directory and files** as shown above
2. **Update your WSGI file** with the simple configuration
3. **Go to Web tab → Click "Reload rudchandryhodge.pythonanywhere.com"**
4. **Wait 30 seconds, then test**: https://rudchandryhodge.pythonanywhere.com/docs

**The errors will disappear once you have the correct file structure!**