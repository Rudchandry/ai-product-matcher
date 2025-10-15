# 🔧 DEBUGGING PYTHONANYWHERE DEPLOYMENT

## Quick Diagnosis Steps

### STEP 1: Check Error Logs
1. Go to PythonAnywhere Dashboard → "Web" tab
2. Click on your app (rudchandryhodge.pythonanywhere.com)
3. Scroll down to "Log files"
4. Click on **"error.log"** - this will show the exact error

### STEP 2: Check WSGI Configuration
Your WSGI file should be at: `/var/www/rudchandryhodge_pythonanywhere_com_wsgi.py`

Replace ALL content with this (update the username):

```python
"""
WSGI configuration for FastAPI Product Matcher on PythonAnywhere
"""

import sys
import os

# Add your project directory to the Python path
path = '/home/rudchandryhodge/pair-matcher'  # Updated with your username
if path not in sys.path:
    sys.path.insert(0, path)

# Import your FastAPI app
from main import app

# PythonAnywhere expects 'application' variable for WSGI
application = app
```

### STEP 3: Verify File Structure
In PythonAnywhere Files, check this structure exists:

```
/home/rudchandryhodge/pair-matcher/
├── main.py
├── wsgi.py
├── requirements.txt
└── app/
    ├── __init__.py
    └── app.py
```

### STEP 4: Check Dependencies
Open a Bash console and run:

```bash
cd /home/rudchandryhodge/pair-matcher
pip3.10 install --user fastapi uvicorn scikit-learn rapidfuzz numpy pandas pydantic
```

### STEP 5: Test Import Manually
In the Bash console, test if your app imports correctly:

```bash
cd /home/rudchandryhodge/pair-matcher
python3.10 -c "from main import app; print('Import successful!')"
```

### STEP 6: Web App Settings
In the "Web" tab, verify:
- **Source code**: `/home/rudchandryhodge/pair-matcher`
- **WSGI configuration file**: The path should point to your WSGI file
- **Python version**: 3.10

## Common Issues & Fixes

### Issue 1: Import Error
**Error log shows**: `ModuleNotFoundError: No module named 'app'`
**Fix**: Make sure `app/__init__.py` exists (can be empty)

### Issue 2: Path Error
**Error log shows**: `No module named 'main'`
**Fix**: Check WSGI file has correct path: `/home/rudchandryhodge/pair-matcher`

### Issue 3: Dependencies Missing
**Error log shows**: `ModuleNotFoundError: No module named 'fastapi'`
**Fix**: Install dependencies with `pip3.10 install --user -r requirements.txt`

### Issue 4: WSGI File Wrong Location
**Error**: "Something went wrong" with no specific error
**Fix**: Make sure WSGI configuration points to the right file

## After Making Changes:
1. Save all files
2. Go to "Web" tab
3. Click **"Reload rudchandryhodge.pythonanywhere.com"**
4. Wait 30 seconds
5. Try accessing the site again

## Need Immediate Help?
Copy the exact error from your error.log and I'll provide a specific fix!