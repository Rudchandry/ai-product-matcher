import json

def handler(request, context):
    """
    Simple Vercel serverless function handler
    """
    try:
        # Get request method and path
        method = request.get('httpMethod', 'GET')
        path = request.get('path', '/')
        
        if method == 'GET':
            if path == '/' or path == '':
                response_data = {
                    "message": "AI Product Matcher API - Simple Function",
                    "version": "2.4",
                    "deployment": "vercel",
                    "status": "working",
                    "handler": "simple_function",
                    "method": method,
                    "path": path
                }
            elif path == '/health':
                response_data = {
                    "status": "healthy",
                    "service": "ai-product-matcher-simple"
                }
            elif path == '/test':
                response_data = {
                    "test": "success",
                    "handler": "simple_function",
                    "request_keys": list(request.keys())
                }
            else:
                response_data = {
                    "error": "Not found",
                    "path": path,
                    "available": ["/", "/health", "/test"]
                }
        
        elif method == 'POST':
            # Handle POST requests
            body = request.get('body', '{}')
            try:
                if isinstance(body, str):
                    data = json.loads(body)
                else:
                    data = body or {}
            except:
                data = {}
            
            search = data.get("search", "").lower()
            products = data.get("products", ["Dell Laptop", "HP Printer", "Cisco Router"])
            
            matches = []
            for product in products:
                if search and search in product.lower():
                    score = 0.9 if search == product.lower() else 0.7
                    matches.append({
                        "product": product,
                        "score": score,
                        "type": "simple_match"
                    })
            
            response_data = {
                "search": search,
                "matches": matches[:5],
                "total": len(matches),
                "handler": "simple_function"
            }
        
        else:
            response_data = {
                "error": "Method not allowed",
                "method": method,
                "allowed": ["GET", "POST"]
            }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                "error": str(e),
                "type": "function_exception",
                "handler": "simple_function"
            })
        }