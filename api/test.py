from fastapi import FastAPI

# Create a minimal FastAPI app for testing
app = FastAPI(title="Test API")

@app.get("/")
async def root():
    return {"message": "Minimal test API", "status": "working"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Try without Mangum first to see if that's the issue
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    # If Mangum fails, create a simple handler
    def handler(event, context):
        return {
            "statusCode": 200,
            "body": '{"message": "Mangum import failed, using fallback"}',
            "headers": {"Content-Type": "application/json"}
        }