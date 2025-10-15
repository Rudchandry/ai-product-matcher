# 🚀 VERCEL DEPLOYMENT - MUCH BETTER THAN PYTHONANYWHERE!

Vercel is perfect for FastAPI - no WSGI configuration headaches, no import issues, and it works immediately!

## STEP 1: Prepare Your Project

### Create these files in your local project directory:

**1. `main.py` (Your FastAPI app):**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rapidfuzz import fuzz
import re

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
    
    search_product = request.search_product.strip().lower()
    if not search_product:
        raise HTTPException(status_code=400, detail="Search product cannot be empty")
    
    matches = []
    for product in request.product_catalog:
        if product.strip():
            # Enhanced fuzzy matching
            similarity_score = calculate_similarity(search_product, product.lower())
            
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

# For Vercel deployment
app.mount = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**2. `vercel.json` (Deployment configuration):**
```json
{
  "functions": {
    "main.py": {
      "runtime": "python3.9"
    }
  },
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/main.py"
    }
  ]
}
```

**3. `requirements.txt`:**
```
fastapi>=0.104.1
pydantic>=2.4.2
rapidfuzz>=3.9.0
uvicorn>=0.23.2
```

**4. `api/index.py` (Vercel entry point):**
```python
from main import app

# Vercel expects this structure
handler = app
```

## STEP 2: Deploy to Vercel

### Option A: Using Vercel CLI (Recommended)
```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy (run from your project directory)
vercel

# Follow the prompts:
# - Link to existing project? No
# - Project name: ai-product-matcher (or whatever you want)
# - Directory: ./ (current directory)
```

### Option B: Using GitHub (Easier for beginners)
1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial FastAPI deployment"
   git remote add origin https://github.com/yourusername/ai-product-matcher.git
   git push -u origin main
   ```

2. **Connect to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Sign up/Login with GitHub
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will automatically detect it's a Python project
   - Click "Deploy"

## STEP 3: Test Your Deployment

Your API will be live at: `https://your-project-name.vercel.app`

**Test endpoints:**
- Documentation: `https://your-project-name.vercel.app/docs`
- Health check: `https://your-project-name.vercel.app/health`
- Match products: `https://your-project-name.vercel.app/match`

## ✅ VERCEL ADVANTAGES OVER PYTHONANYWHERE:

✅ **No WSGI configuration headaches**
✅ **Automatic HTTPS**
✅ **Global CDN**
✅ **Instant deployments**
✅ **Git-based deployments**
✅ **Better free tier**
✅ **Automatic scaling**
✅ **No import issues**
✅ **Works immediately**

## 🎉 TOTAL TIME: ~5 minutes instead of hours!

Would you like me to help you set this up? It's infinitely easier than PythonAnywhere!