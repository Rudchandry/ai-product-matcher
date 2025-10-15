from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json

# Create a very simple FastAPI app for testing
app = FastAPI(title="AI Product Matcher")

@app.get("/")
async def root():
    return {
        "message": "AI Product Matcher API - Simple Version",
        "version": "2.1",
        "deployment": "vercel", 
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ai-product-matcher-simple"}

@app.get("/test")
async def test():
    return {"test": "success", "dependencies": "minimal"}

# Simple product matching without external dependencies
@app.post("/match-simple")
async def simple_match(data: dict):
    search = data.get("search", "").lower()
    products = data.get("products", [])
    
    matches = []
    for product in products:
        # Simple string matching
        if search in product.lower():
            score = 0.9 if search == product.lower() else 0.7
            matches.append({
                "product": product,
                "score": score,
                "type": "simple_match"
            })
    
    return {
        "search": search,
        "matches": matches[:5],  # Top 5
        "total": len(matches)
    }

# Create the handler function for Vercel
def handler(request):
    import asyncio
    from fastapi.testclient import TestClient
    
    try:
        # Use TestClient for synchronous handling
        client = TestClient(app)
        
        # Extract request details
        path = request.get('rawPath', '/')
        method = request.get('httpMethod', 'GET')
        
        if method == 'GET':
            response = client.get(path)
        elif method == 'POST':
            body = request.get('body', '{}')
            if isinstance(body, str):
                try:
                    json_body = json.loads(body)
                except:
                    json_body = {}
            else:
                json_body = body or {}
            response = client.post(path, json=json_body)
        else:
            response = client.get('/')  # Fallback
        
        return {
            "statusCode": response.status_code,
            "headers": {"Content-Type": "application/json"},
            "body": response.text
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e), "type": "handler_error"})
        }