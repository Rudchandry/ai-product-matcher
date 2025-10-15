"""
WSGI configuration for FastAPI Product Matcher on PythonAnywhere

This file configures the FastAPI app to run on PythonAnywhere's WSGI server.
"""

import sys
import os

# Add your project directory to the Python path
path = '/home/yourusername/pair-matcher'  # Replace 'yourusername' with your actual username
if path not in sys.path:
    sys.path.insert(0, path)

# Set environment variables if needed
os.environ.setdefault('PYTHONPATH', path)

# Import your FastAPI app
from main import app

# PythonAnywhere expects 'application' variable for WSGI
application = app

# Optional: Add some debugging
if __name__ == "__main__":
    print("WSGI app loaded successfully")
    print(f"App type: {type(application)}")
    print(f"Python path: {sys.path}")