# 🔍 DEBUGGING CHECKLIST - Find the Exact Problem

## Step 1: Check if Files Were Created Successfully
In PythonAnywhere Bash console, run:

```bash
cd /home/rudchandryhodge/pair-matcher
ls -la
```

**Expected output:**
```
main.py
requirements.txt
```

## Step 2: Check if main.py Has Correct Content
```bash
head -20 main.py
```

**Should start with:**
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
...
```

## Step 3: Test Import Manually
```bash
python3.10 -c "import sys; print('Python version:', sys.version)"
python3.10 -c "try:
    from main import application
    print('SUCCESS: Import works')
except Exception as e:
    print('ERROR:', e)"
```

**Or use this single-line version:**
```bash
python3.10 -c "try: from main import application; print('SUCCESS')" 2>&1 || echo "FAILED"
```

## Step 4: Check Dependencies
```bash
pip3.10 list | grep -E "(fastapi|pydantic|rapidfuzz)"
```

## Step 5: Check Your WSGI File
1. Go to **Web tab** in PythonAnywhere
2. Click on your **WSGI configuration file** link
3. Make sure it contains EXACTLY this:

```python
import sys
import os

path = '/home/rudchandryhodge/pair-matcher'
if path not in sys.path:
    sys.path.insert(0, path)

from main import application
```

## Step 6: Check Web App Settings
In **Web tab**, verify:
- **Source code**: `/home/rudchandryhodge/pair-matcher`
- **Python version**: 3.10

## Step 7: Check Error Logs
In **Web tab**, scroll down to **Log files** and click **error.log**

## Step 8: If Still Not Working - Ultra Simple Test
Create a minimal test file:

```bash
cat > test_simple.py << 'EOF'
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

application = app
EOF
```

Then update WSGI to use:
```python
import sys
import os

path = '/home/rudchandryhodge/pair-matcher'
if path not in sys.path:
    sys.path.insert(0, path)

from test_simple import application
```

---

## Quick Fix Commands (Run These One by One):

```bash
# 1. Verify location
cd /home/rudchandryhodge/pair-matcher
pwd

# 2. Check files exist
ls -la

# 3. Test Python import
python3.10 -c "from main import application; print('SUCCESS')" || echo "Import failed"

# 4. Check FastAPI can start
python3.10 -c "from main import app; print('FastAPI app loaded:', type(app))" || echo "FastAPI failed"

# 5. Reinstall dependencies if needed
pip3.10 install --user fastapi pydantic rapidfuzz uvicorn

# 6. Test again
python3.10 -c "from main import application; print('Final test: SUCCESS')"
```

**Tell me the output of these commands and I'll pinpoint the exact issue!**