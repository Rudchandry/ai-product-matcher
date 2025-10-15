import sys
import os

# Add the parent directory to Python path to import main
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from main import app
    print("Successfully imported main app")
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback: create a simple FastAPI app
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="Product Matcher API - Fallback")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "Product Matcher API - Import Error", "error": str(e)}
    
    @app.get("/health")
    async def health():
        return {"status": "unhealthy", "error": "main app import failed"}

# For Vercel, we need to use Mangum to wrap the ASGI app
from mangum import Mangum
handler = Mangum(app, lifespan="off")