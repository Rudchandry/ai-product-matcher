from http.server import BaseHTTPRequestHandler
import json
import re
import urllib.parse

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            path = self.path
            
            if path == "/" or path == "":
                response_data = {
                    "message": "AI Product Matcher API - Live & Working!",
                    "version": "3.1",
                    "status": "success",
                    "deployment": "vercel",
                    "endpoints": {
                        "/": "API info",
                        "/health": "Health check", 
                        "/match": "POST - AI product matching",
                        "/extract-terms": "POST - Extract product terms"
                    },
                    "example_usage": {
                        "match": {
                            "method": "POST",
                            "url": "https://ai-product-matcher-38sp.vercel.app/match",
                            "body": {
                                "search_product": "Dell Laptop",
                                "product_catalog": ["Dell XPS 13 Laptop", "HP Pavilion", "Cisco Router"],
                                "max_results": 5
                            }
                        }
                    }
                }
            elif path == "/health":
                response_data = {
                    "status": "healthy",
                    "service": "ai-product-matcher",
                    "deployment": "vercel",
                    "timestamp": "2025-10-15"
                }
            else:
                response_data = {
                    "error": "Endpoint not found",
                    "path": path,
                    "available_endpoints": ["/", "/health", "/match", "/extract-terms"]
                }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(response_data, indent=2).encode())
            
        except Exception as e:
            self.send_error_response(str(e), "get_handler_error")
    
    def do_POST(self):
        try:
            path = self.path
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            
            try:
                data = json.loads(post_data.decode())
            except:
                data = {}
            
            if path == "/match":
                response_data = self.handle_match_request(data)
            elif path == "/extract-terms":
                response_data = self.handle_extract_terms(data)
            else:
                response_data = {
                    "error": "POST endpoint not found",
                    "path": path,
                    "available_post_endpoints": ["/match", "/extract-terms"]
                }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(response_data, indent=2).encode())
            
        except Exception as e:
            self.send_error_response(str(e), "post_handler_error")
    
    def handle_match_request(self, data):
        """AI-powered product matching"""
        search_product = data.get("search_product", "").strip()
        product_catalog = data.get("product_catalog", [])
        max_results = data.get("max_results", 5)
        
        if not search_product:
            return {"error": "search_product is required", "code": 400}
        
        if not product_catalog:
            return {"error": "product_catalog cannot be empty", "code": 400}
        
        matches = []
        for product in product_catalog:
            if product and product.strip():
                similarity_score = self.calculate_similarity(search_product.lower(), product.lower())
                
                if similarity_score >= 0.8:
                    match_type = "high"
                elif similarity_score >= 0.6:
                    match_type = "medium"  
                else:
                    match_type = "low"
                
                matches.append({
                    "product": product,
                    "similarity_score": round(similarity_score, 4),
                    "match_type": match_type
                })
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Limit results
        top_matches = matches[:min(max_results, len(matches))]
        
        return {
            "search_product": search_product,
            "matches": top_matches,
            "total_matches": len(matches),
            "api": "ai-product-matcher",
            "deployment": "vercel"
        }
    
    def calculate_similarity(self, search, product):
        """Calculate similarity between search term and product name"""
        # Exact match
        if search == product:
            return 1.0
        
        # Substring match
        if search in product or product in search:
            return 0.9
        
        # Simple fuzzy similarity (since we can't use rapidfuzz)
        search_words = set(search.split())
        product_words = set(product.split())
        
        # Word overlap scoring
        common_words = search_words.intersection(product_words)
        total_words = search_words.union(product_words)
        
        if total_words:
            word_similarity = len(common_words) / len(total_words)
        else:
            word_similarity = 0
        
        # Character overlap bonus
        search_chars = set(search.replace(" ", ""))
        product_chars = set(product.replace(" ", ""))
        common_chars = search_chars.intersection(product_chars)
        total_chars = search_chars.union(product_chars)
        
        if total_chars:
            char_similarity = len(common_chars) / len(total_chars)
        else:
            char_similarity = 0
        
        # Combined similarity (weighted toward word matching)
        final_similarity = (word_similarity * 0.7) + (char_similarity * 0.3)
        
        return final_similarity
    
    def handle_extract_terms(self, data):
        """Extract key terms from a product name"""
        product_name = data.get("product_name", "").strip()
        
        if not product_name:
            return {"error": "product_name is required", "code": 400}
        
        # Model number patterns
        model_pattern = r'\b(?:v?\d+\.?\d*|[a-z]+\d+|\d+[a-z]+)\b'
        models = re.findall(model_pattern, product_name.lower())
        
        # Common tech brands
        brands = ['dell', 'hp', 'cisco', 'microsoft', 'vmware', 'intel', 'amd', 'nvidia', 'apple', 'lenovo']
        detected_brands = [brand for brand in brands if brand in product_name.lower()]
        
        return {
            "product_name": product_name,
            "extracted_terms": {
                "models": models,
                "brands": detected_brands,
                "words": product_name.lower().split()
            },
            "api": "ai-product-matcher"
        }
    
    def send_error_response(self, error_msg, error_type):
        """Send standardized error response"""
        self.send_response(500)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = {
            "error": error_msg,
            "type": error_type,
            "api": "ai-product-matcher"
        }
        self.wfile.write(json.dumps(error_response).encode())