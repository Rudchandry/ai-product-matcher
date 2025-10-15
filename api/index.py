from http.server import BaseHTTPRequestHandler
import json
import urllib.parse

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the path
        path = self.path
        
        try:
            if path == "/" or path == "":
                response_data = {
                    "message": "AI Product Matcher API - Basic Version",
                    "version": "2.3",
                    "deployment": "vercel",
                    "status": "running",
                    "method": "BaseHTTPRequestHandler",
                    "commit": "691035a",
                    "config": "no_vercel_json"
                }
            elif path == "/health":
                response_data = {
                    "status": "healthy", 
                    "service": "ai-product-matcher-basic"
                }
            elif path == "/test":
                response_data = {
                    "test": "success", 
                    "handler": "BaseHTTPRequestHandler",
                    "path": path
                }
            else:
                response_data = {
                    "error": "Not found",
                    "path": path,
                    "available_paths": ["/", "/health", "/test"]
                }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response_json = json.dumps(response_data)
            self.wfile.write(response_json.encode())
            
        except Exception as e:
            # Error handling
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            error_response = {
                "error": str(e),
                "type": "handler_exception",
                "path": path
            }
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            if post_data:
                try:
                    data = json.loads(post_data.decode())
                except:
                    data = {}
            else:
                data = {}
            
            # Simple matching logic
            search = data.get("search", "").lower()
            products = data.get("products", ["Dell Laptop", "HP Printer", "Cisco Router"])
            
            matches = []
            for product in products:
                if search in product.lower():
                    score = 0.9 if search == product.lower() else 0.7
                    matches.append({
                        "product": product,
                        "score": score,
                        "type": "basic_match"
                    })
            
            response_data = {
                "search": search,
                "matches": matches[:5],
                "total": len(matches),
                "handler": "BaseHTTPRequestHandler"
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            error_response = {
                "error": str(e),
                "type": "post_handler_exception"
            }
            self.wfile.write(json.dumps(error_response).encode())