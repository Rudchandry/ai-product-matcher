# 🎯 INSTANT WORKING SOLUTION
# Copy this ENTIRE script into PythonAnywhere Bash console

echo "🚀 Creating instant working API..."

# Clean slate
cd /home/rudchandryhodge
rm -rf pair-matcher
mkdir pair-matcher
cd pair-matcher

# Create working FastAPI app in ONE command
python3.10 << 'PYTHON_SCRIPT'
content = '''from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Product Matcher API")

class MatchRequest(BaseModel):
    search_product: str
    product_catalog: List[str]

@app.get("/")
def root():
    return {"message": "Product Matcher API is running!", "status": "success"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "product-matcher"}

@app.post("/match")
def match_products(request: MatchRequest):
    search = request.search_product.lower()
    matches = []
    
    for product in request.product_catalog:
        # Simple string matching
        if search in product.lower() or product.lower() in search:
            score = 0.9
        elif any(word in product.lower() for word in search.split()):
            score = 0.7
        else:
            score = 0.3
            
        matches.append({
            "product": product,
            "score": score,
            "match": "high" if score >= 0.8 else "medium" if score >= 0.6 else "low"
        })
    
    matches.sort(key=lambda x: x["score"], reverse=True)
    return {
        "search_product": request.search_product,
        "matches": matches[:5],
        "total": len(matches)
    }

application = app
'''

with open('main.py', 'w') as f:
    f.write(content)
print("✅ main.py created")
PYTHON_SCRIPT

# Create requirements
echo "fastapi" > requirements.txt
echo "pydantic" >> requirements.txt

# Install
pip3.10 install --user fastapi pydantic

# Test
python3.10 -c "from main import application; print('SUCCESS: Ready to deploy')"

echo ""
echo "🎉 DEPLOYMENT READY!"
echo ""
echo "Now do these 3 steps in PythonAnywhere:"
echo "1. Web tab → Set source code: /home/rudchandryhodge/pair-matcher"
echo "2. WSGI file → Replace with:"
echo "   import sys"
echo "   sys.path.insert(0, '/home/rudchandryhodge/pair-matcher')"
echo "   from main import application"
echo "3. Reload web app"
echo ""
echo "Then visit: https://rudchandryhodge.pythonanywhere.com/"