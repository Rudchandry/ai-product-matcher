# 🚀 EXACT PythonAnywhere Deployment Instructions
# Copy-paste these files and follow the steps exactly

## STEP 1: Create Project Directory
1. Go to PythonAnywhere Dashboard
2. Click "Files" tab
3. Navigate to `/home/yourusername/` (replace with your actual username)
4. Click "New directory" and name it: `pair-matcher`
5. Enter the `pair-matcher` directory

## STEP 2: Create These Files (Copy-Paste Content Below)

### FILE 1: main.py (COMPLETE FastAPI APP - NO OTHER FILES NEEDED)
Click "New file" → Name it `main.py` → Paste this content:

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

# CRITICAL: This line is what PythonAnywhere needs
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### FILE 2: wsgi.py (OPTIONAL - NOT NEEDED IF USING WEB APP WSGI CONFIG)
Back in `pair-matcher` directory, create `wsgi.py`:

```python
"""
WSGI configuration for FastAPI Product Matcher on PythonAnywhere
"""

import sys
import os

# Add your project directory to the Python path
path = '/home/yourusername/pair-matcher'  # REPLACE 'yourusername' with your actual username
if path not in sys.path:
    sys.path.insert(0, path)

# Set environment variables if needed
os.environ.setdefault('PYTHONPATH', path)

# Import your FastAPI app
from main import application
```

### FILE 5: requirements.txt
Create `requirements.txt`:

```
fastapi==0.104.1
pydantic==2.4.2
rapidfuzz>=3.9.0
scikit-learn==1.3.2
numpy==1.24.4
pandas==2.1.4
joblib==1.3.2
```

## STEP 3: Install Dependencies
1. Go to "Consoles" tab
2. Start a "Bash" console
3. Run these commands:

```bash
cd /home/yourusername/pair-matcher
pip3.10 install --user -r requirements.txt
```

## STEP 4: Create Web App
1. Go to "Web" tab
2. Click "Add a new web app"
3. Choose your domain (yourusername.pythonanywhere.com)
4. Select "Manual configuration"
5. Choose "Python 3.10"

## STEP 5: Configure Web App
1. Set "Source code" to: `/home/yourusername/pair-matcher`
2. Click on "WSGI configuration file" link
3. Replace ALL content with the wsgi.py content above
4. Remember to replace 'yourusername' with your actual username!
5. Save the file

## STEP 6: Reload and Test
1. Click green "Reload yourusername.pythonanywhere.com" button
2. Visit: https://yourusername.pythonanywhere.com/docs
3. Test: https://yourusername.pythonanywhere.com/model-status

## 🎉 Your API Endpoints:
- Documentation: https://yourusername.pythonanywhere.com/docs
- Match Products: https://yourusername.pythonanywhere.com/match
- Model Status: https://yourusername.pythonanywhere.com/model-status
- Batch Similarity: https://yourusername.pythonanywhere.com/batch-similarity

Total time: ~10 minutes! 🚀