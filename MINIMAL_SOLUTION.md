# 🎯 ULTRA-MINIMAL WORKING SOLUTION
# Copy-paste this EXACT script in PythonAnywhere Bash console

# Clean everything and start fresh
cd /home/rudchandryhodge
rm -rf pair-matcher
mkdir pair-matcher
cd pair-matcher

# Create the absolute minimal FastAPI app
cat > main.py << 'EOF'
from fastapi import FastAPI

app = FastAPI(title="Simple API Test")

@app.get("/")
def hello():
    return {"message": "Hello! API is working!", "status": "success"}

@app.get("/test")
def test():
    return {"test": "This endpoint works too!"}

# PythonAnywhere needs this
application = app
EOF

# Create minimal requirements
echo "fastapi" > requirements.txt

# Install just FastAPI
pip3.10 install --user fastapi

# Test it works
echo "Testing import..."
python3.10 -c "from main import application; print('Import works!')"

echo "Done! Now update your WSGI file."