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
    allow_origins=["*"],  # In production, replace with specific domains
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
        # Create a TF-IDF vectorizer with optimized parameters
        tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            max_features=10000,  # Limit features for memory efficiency
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        logger.info("TF-IDF vectorizer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TF-IDF vectorizer: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the TF-IDF model on startup"""
    logger.info("Starting up AI Product Matcher...")
    success = _load_tfidf_model()
    if success:
        logger.info("TF-IDF model loaded successfully")
    else:
        logger.warning("TF-IDF model failed to load, using string matching only")

def _clean_product_name(name: str) -> str:
    """Clean and normalize product names for better matching"""
    if not name:
        return ""
    
    # Convert to lowercase and strip whitespace
    cleaned = name.lower().strip()
    
    # Remove common business suffixes and prefixes
    patterns = [
        r'\b(inc|llc|ltd|corp|corporation|company|co\.?)\b',
        r'\b(the|a|an)\b',
        r'[^\w\s]',  # Remove special characters except spaces
    ]
    
    for pattern in patterns:
        cleaned = re.sub(pattern, ' ', cleaned)
    
    # Normalize spacing in model numbers (e.g., "iDRAC 8" -> "idrac8")
    model_spacing_patterns = [
        (r'\bidrac\s*(\d+)\b', r'idrac\1'),
        (r'\bgen\s*(\d+)\b', r'gen\1'),
        (r'\bv\s*(\d+)\b', r'v\1'),
        (r'\bversion\s*(\d+)\b', r'version\1'),
    ]
    
    for pattern, replacement in model_spacing_patterns:
        cleaned = re.sub(pattern, replacement, cleaned)
    
    # Normalize common abbreviations and expansions
    abbreviation_patterns = [
        (r'\bcom\b', 'commercial'),
        (r'\bcomm\b', 'commercial'),
        (r'\bent\b', 'enterprise'),  
        (r'\bstd\b', 'standard'),
        (r'\bpro\b', 'professional'),
        (r'\bbiz\b', 'business'),
        (r'\bedu\b', 'education'),
        (r'\bgov\b', 'government'),
        (r'\bsub\b', 'subscription'),
        (r'\blic\b', 'license'),
        (r'\bgen2\b', 'generation2'),
        (r'\bg2\b', 'generation2'),
        (r'\boem\b', 'original equipment manufacturer'),
    ]
    
    for abbrev, full in abbreviation_patterns:
        cleaned = re.sub(abbrev, full, cleaned)
    
    # Remove noise terms that don't affect product identity
    noise_patterns = [
        r'\b(subscription|license|licensing|level|year|years|1|annual|monthly)\b',
        r'\b\d+\s*(year|month|day)s?\b',  # Remove time periods
        r'\b(one|two|three|four|five)\s*(year|month)\b',
        r'\b(standalone|server|original|equipment|manufacturer)\b',  # Remove deployment descriptors
    ]
    
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, ' ', cleaned)
    
    # Normalize plural forms to singular for better matching
    plural_patterns = [
        (r'\bdesktops\b', 'desktop'),
        (r'\blaptops\b', 'laptop'),
        (r'\bmonitors\b', 'monitor'),
        (r'\bkeyboards\b', 'keyboard'),
        (r'\bmouses\b', 'mouse'),
        (r'\btablets\b', 'tablet'),
        (r'\bphones\b', 'phone'),
        (r'\bservers\b', 'server'),
        (r'\bteams\b', 'team'),
    ]
    
    for plural, singular in plural_patterns:
        cleaned = re.sub(plural, singular, cleaned)
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def _compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity using enhanced product matching logic"""
    global tfidf_vectorizer
    
    if not tfidf_vectorizer:
        logger.warning("TF-IDF vectorizer not available, falling back to string similarity")
        return fuzz.ratio(text1, text2) / 100.0
    
    try:
        # Clean the input texts
        clean_text1 = _clean_product_name(text1)
        clean_text2 = _clean_product_name(text2)
        
        if not clean_text1 or not clean_text2:
            return 0.0
        
        # Extract key product identifiers and important words
        def extract_key_terms(text):
            words = text.split()
            # Product type keywords that are very important for matching
            product_types = {'desktop', 'laptop', 'monitor', 'keyboard', 'mouse', 'tablet', 'phone', 'server', 'switch', 'router', 'gateway', 'controller', 'access', 'point', 'cloud', 'key', 'acrobat', 'photoshop', 'illustrator', 'office', 'word', 'excel', 'powerpoint', 'idrac', 'drac', 'ilo', 'bmc', 'ipmi'}
            brand_names = {'dell', 'hp', 'lenovo', 'apple', 'microsoft', 'adobe', 'google', 'amazon', 'ubiquiti', 'cisco', 'netgear', 'linksys', 'unifi'}
            model_indicators = {'professional', 'business', 'premium', 'standard', 'basic', 'enterprise', 'plus', 'lite', 'max', 'mini', 'generation2', 'team', 'commercial'}
            
            key_terms = set()
            regular_terms = set()
            model_numbers = set()
            
            for word in words:
                word_lower = word.lower()
                
                # Check for model numbers (contain both letters and numbers, or start with specific patterns)
                if (re.match(r'^[a-z]+-[a-z0-9]+-[a-z0-9]+$', word_lower) or  # UCK-G2-PLUS pattern
                    re.match(r'^[a-z]+[0-9]+[a-z]*$', word_lower) or           # Gen2 pattern
                    any(char.isdigit() for char in word) and any(char.isalpha() for char in word)):
                    model_numbers.add(word_lower)
                elif (word_lower in product_types or 
                      word_lower in brand_names or 
                      word_lower in model_indicators or
                      any(char.isdigit() for char in word)):  # Include pure numbers
                    key_terms.add(word_lower)
                else:
                    regular_terms.add(word_lower)
            
            return key_terms, regular_terms, model_numbers
        
        key_terms1, regular_terms1, model_numbers1 = extract_key_terms(clean_text1)
        key_terms2, regular_terms2, model_numbers2 = extract_key_terms(clean_text2)
        
        # Calculate model number similarity (highest priority)
        model_intersection = len(model_numbers1.intersection(model_numbers2))
        model_union = len(model_numbers1.union(model_numbers2))
        model_similarity = model_intersection / model_union if model_union > 0 else 0.0
        
        # Calculate key terms similarity (brand, product type, model indicators)
        key_intersection = len(key_terms1.intersection(key_terms2))
        key_union = len(key_terms1.union(key_terms2))
        key_similarity = key_intersection / key_union if key_union > 0 else 0.0
        
        # Calculate regular terms similarity
        regular_intersection = len(regular_terms1.intersection(regular_terms2))
        regular_union = len(regular_terms1.union(regular_terms2))
        regular_similarity = regular_intersection / regular_union if regular_union > 0 else 0.0
        
        # Character-level similarity for partial matches
        char_score = fuzz.ratio(clean_text1, clean_text2) / 100.0
        
        # Weighted scoring: Model numbers are most important, then key terms
        # Model numbers: 40% (highest priority for exact product matching)
        # Key terms (brand, product type): 35%
        # Regular terms: 15%
        # Character similarity: 10%
        combined_score = (model_similarity * 0.4) + (key_similarity * 0.35) + (regular_similarity * 0.15) + (char_score * 0.1)
        
        # Bonus system
        all_brands = {'dell', 'hp', 'lenovo', 'apple', 'microsoft', 'adobe', 'google', 'amazon', 'ubiquiti', 'cisco', 'netgear', 'linksys'}
        all_product_types = {'desktop', 'laptop', 'monitor', 'keyboard', 'mouse', 'tablet', 'phone', 'server', 'switch', 'router', 'gateway', 'controller', 'access', 'point', 'cloud', 'key'}
        
        brands_match = bool(key_terms1.intersection(key_terms2).intersection(all_brands))
        product_types_match = bool(key_terms1.intersection(key_terms2).intersection(all_product_types))
        
        # Big bonus for model number matches (indicates same exact product)
        if model_similarity > 0:
            combined_score = min(1.0, combined_score + 0.3)  # 30% bonus for model number match
        elif brands_match and product_types_match:
            combined_score = min(1.0, combined_score + 0.2)  # 20% bonus for brand + product type match
        elif brands_match or product_types_match:
            combined_score = min(1.0, combined_score + 0.1)  # 10% bonus for either match
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, combined_score))
        
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        # Fallback to string matching
        return fuzz.ratio(text1, text2) / 100.0

# Pydantic models
class MatchRequest(BaseModel):
    left: List[str]
    right: List[str]
    use_ai: Optional[bool] = True
    threshold: Optional[float] = 0.6

class MatchResult(BaseModel):
    left: str
    right: str
    score: float
    method: str
    is_ai_powered: bool

class ModelStatus(BaseModel):
    ai_available: bool
    model_type: str
    version: str

class BatchSimilarityRequest(BaseModel):
    products: List[str]
    use_ai: Optional[bool] = True

class SimilarityMatrix(BaseModel):
    products: List[str]
    similarity_matrix: List[List[float]]
    method: str

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "AI Product Matcher API"}

@app.get("/model-status", response_model=ModelStatus)
def get_model_status():
    """Get the status of the AI model"""
    return ModelStatus(
        ai_available=tfidf_vectorizer is not None,
        model_type="TF-IDF with Scikit-learn" if tfidf_vectorizer else "String matching only",
        version="1.0.0"
    )

@app.post("/match", response_model=List[MatchResult])
def match_products(request: MatchRequest):
    """
    Match products between two lists using AI semantic similarity or string matching
    """
    if not request.left or not request.right:
        raise HTTPException(status_code=400, detail="Both left and right product lists must be non-empty")
    
    results = []
    use_ai = request.use_ai and tfidf_vectorizer is not None
    
    logger.info(f"Processing {len(request.left)} x {len(request.right)} comparisons, AI: {use_ai}")
    
    for left_product in request.left:
        best_match = None
        best_score = 0.0
        
        for right_product in request.right:
            if use_ai:
                # Use TF-IDF semantic similarity
                similarity = _compute_semantic_similarity(left_product, right_product)
                method = "tfidf_semantic"
            else:
                # Use string matching
                similarity = fuzz.ratio(left_product, right_product) / 100.0
                method = "string_fuzzy"
            
            if similarity > best_score:
                best_score = similarity
                best_match = right_product
        
        # Only include matches above threshold
        if best_match and best_score >= request.threshold:
            results.append(MatchResult(
                left=left_product,
                right=best_match,
                score=round(best_score, 3),
                method=method,
                is_ai_powered=use_ai
            ))
    
    logger.info(f"Found {len(results)} matches above threshold {request.threshold}")
    return results

@app.post("/batch-similarity", response_model=SimilarityMatrix)
def compute_batch_similarity(request: BatchSimilarityRequest):
    """
    Compute similarity matrix for a list of products
    """
    if not request.products:
        raise HTTPException(status_code=400, detail="Product list must be non-empty")
    
    if len(request.products) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 products allowed for batch processing")
    
    use_ai = request.use_ai and tfidf_vectorizer is not None
    n_products = len(request.products)
    
    # Initialize similarity matrix
    similarity_matrix = [[0.0 for _ in range(n_products)] for _ in range(n_products)]
    
    logger.info(f"Computing {n_products}x{n_products} similarity matrix, AI: {use_ai}")
    
    for i in range(n_products):
        for j in range(n_products):
            if i == j:
                similarity_matrix[i][j] = 1.0
            elif i < j:  # Only compute upper triangle, then mirror
                if use_ai:
                    similarity = _compute_semantic_similarity(
                        request.products[i], 
                        request.products[j]
                    )
                    method = "tfidf_semantic"
                else:
                    similarity = fuzz.ratio(request.products[i], request.products[j]) / 100.0
                    method = "string_fuzzy"
                
                # Mirror the result
                similarity_matrix[i][j] = round(similarity, 3)
                similarity_matrix[j][i] = round(similarity, 3)
    
    return SimilarityMatrix(
        products=request.products,
        similarity_matrix=similarity_matrix,
        method=method
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
