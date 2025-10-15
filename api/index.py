from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rapidfuzz import fuzz
import re

# Create FastAPI app directly in the serverless function
app = FastAPI(title="AI Product Matcher", description="AI-powered product matching API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchRequest(BaseModel):
    search_product: str
    product_catalog: List[str]
    max_results: Optional[int] = 5

class ProductMatch(BaseModel):
    product: str
    similarity_score: float
    match_type: str

class MatchResponse(BaseModel):
    search_product: str
    matches: List[ProductMatch]
    total_matches: int

@app.get("/")
async def read_root():
    return {
        "message": "AI Product Matcher API",
        "version": "2.0",
        "deployment": "vercel",
        "status": "running",
        "endpoints": ["/docs", "/match", "/health"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-product-matcher"}

@app.post("/match", response_model=MatchResponse)
async def match_products(request: MatchRequest):
    """AI-powered product matching with fuzzy search"""
    
    if not request.product_catalog:
        raise HTTPException(status_code=400, detail="Product catalog cannot be empty")
    
    search_product = request.search_product.strip()
    if not search_product:
        raise HTTPException(status_code=400, detail="Search product cannot be empty")
    
    matches = []
    for product in request.product_catalog:
        if product.strip():
            # Enhanced fuzzy matching
            similarity_score = calculate_similarity(search_product.lower(), product.lower())
            
            if similarity_score >= 0.8:
                match_type = "high"
            elif similarity_score >= 0.6:
                match_type = "medium"
            else:
                match_type = "low"
            
            matches.append(ProductMatch(
                product=product,
                similarity_score=round(similarity_score, 4),
                match_type=match_type
            ))
    
    # Sort by similarity score (descending)
    matches.sort(key=lambda x: x.similarity_score, reverse=True)
    
    # Limit results
    max_results = min(request.max_results, len(matches))
    top_matches = matches[:max_results]
    
    return MatchResponse(
        search_product=request.search_product,
        matches=top_matches,
        total_matches=len(matches)
    )

def calculate_similarity(search: str, product: str) -> float:
    """Calculate similarity between search term and product name"""
    
    # Exact match
    if search == product:
        return 1.0
    
    # Substring match
    if search in product or product in search:
        return 0.9
    
    # Fuzzy string similarity
    fuzzy_score = fuzz.ratio(search, product) / 100.0
    
    # Word overlap bonus
    search_words = set(search.split())
    product_words = set(product.split())
    common_words = search_words.intersection(product_words)
    
    if common_words:
        word_bonus = len(common_words) / max(len(search_words), len(product_words)) * 0.3
        fuzzy_score = min(1.0, fuzzy_score + word_bonus)
    
    return fuzzy_score

@app.post("/extract-terms")
async def extract_product_terms(request: dict):
    """Extract key terms from a product name"""
    product_name = request.get('product_name', '')
    if not product_name:
        raise HTTPException(status_code=400, detail="Product name is required")
    
    # Model number patterns
    model_pattern = r'\b(?:v?\d+\.?\d*|[a-z]+\d+|\d+[a-z]+)\b'
    models = re.findall(model_pattern, product_name.lower())
    
    # Common tech brands
    brands = ['dell', 'hp', 'cisco', 'microsoft', 'vmware', 'intel', 'amd', 'nvidia']
    detected_brands = [brand for brand in brands if brand in product_name.lower()]
    
    return {
        "product_name": product_name,
        "extracted_terms": {
            'models': models,
            'brands': detected_brands,
            'words': product_name.lower().split()
        }
    }

# For Vercel, we need to use Mangum to wrap the ASGI app
from mangum import Mangum
handler = Mangum(app, lifespan="off")