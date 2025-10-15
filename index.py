from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            response_data = {
                "message": "AI Product Matcher API - Working!",
                "version": "3.0",
                "status": "success",
                "deployment": "vercel",
                "handler": "BaseHTTPRequestHandler",
                "path": self.path
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            error_response = {
                "error": str(e),
                "type": "handler_error"
            }
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            
            try:
                data = json.loads(post_data.decode())
            except:
                data = {}
            
            response_data = {
                "message": "POST request received",
                "data_received": data,
                "handler": "BaseHTTPRequestHandler"
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            error_response = {
                "error": str(e),
                "type": "post_handler_error"
            }
            self.wfile.write(json.dumps(error_response).encode())