import sys
import os

# Add the parent directory to Python path to import main
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import app

# Export the app for Vercel
handler = app