#!/bin/bash
# FOOLPROOF DEPLOYMENT SCRIPT FOR PYTHONANYWHERE
# Copy and paste this ENTIRE script into a PythonAnywhere Bash console

echo "🚀 Starting foolproof deployment..."

# Navigate to home directory
cd /home/rudchandryhodge

# Remove any existing deployment
echo "🧹 Cleaning up existing files..."
rm -rf pair-matcher

# Create fresh project directory
echo "📁 Creating fresh project directory..."
mkdir pair-matcher
cd pair-matcher

# Create the complete FastAPI application in ONE file
echo "📝 Creating main.py..."
cat > main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rapidfuzz import fuzz
import re

app = FastAPI(title="AI Product Matcher", description="AI-powered product name matching API")

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
    return {"message": "AI Product Matcher API", "status": "running", "endpoints": ["/docs", "/match", "/model-status"]}

@app.get("/model-status")
async def model_status():
    return {"status": "ready", "algorithm": "fuzzy_matching"}

@app.post("/match", response_model=MatchResponse)
async def match_products(request: MatchRequest):
    if not request.product_catalog:
        raise HTTPException(status_code=400, detail="Product catalog cannot be empty")
    
    search_product = request.search_product.strip()
    if not search_product:
        raise HTTPException(status_code=400, detail="Search product cannot be empty")
    
    matches = []
    for product in request.product_catalog:
        if product.strip():
            # Simple fuzzy matching
            similarity_score = fuzz.ratio(search_product.lower(), product.lower()) / 100.0
            
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
    
    matches.sort(key=lambda x: x.similarity_score, reverse=True)
    top_matches = matches[:request.max_results]
    
    return MatchResponse(
        search_product=search_product,
        matches=top_matches,
        total_matches=len(matches)
    )

# This is what PythonAnywhere needs
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Create requirements.txt
echo "📦 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
fastapi>=0.104.1
pydantic>=2.4.2
rapidfuzz>=3.9.0
uvicorn>=0.23.2
EOF

# Install packages
echo "⬇️ Installing packages..."
pip3.10 install --user -r requirements.txt

# Test the import
echo "🧪 Testing import..."
python3.10 -c "from main import application; print('✅ SUCCESS: Application imported correctly!')"

echo "🎉 Deployment complete!"
echo ""
echo "🔧 Now configure your web app:"
echo "1. Go to Web tab in PythonAnywhere"
echo "2. Set Source code to: /home/rudchandryhodge/pair-matcher"
echo "3. Set WSGI configuration file content to:"
echo ""
echo "import sys"
echo "import os"
echo "path = '/home/rudchandryhodge/pair-matcher'"
echo "if path not in sys.path:"
echo "    sys.path.insert(0, path)"
echo "from main import application"
echo ""
echo "4. Click 'Reload rudchandryhodge.pythonanywhere.com'"
echo "5. Visit: https://rudchandryhodge.pythonanywhere.com/docs"