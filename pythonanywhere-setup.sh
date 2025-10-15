#!/bin/bash
# PythonAnywhere Setup Script for FastAPI Product Matcher
# Run this in the PythonAnywhere Bash console

echo "🚀 Setting up FastAPI Product Matcher on PythonAnywhere"
echo "======================================================"

# Create project directory
echo "📁 Creating project directory..."
mkdir -p /home/$USER/pair-matcher
cd /home/$USER/pair-matcher

echo "📦 Installing Python packages..."
pip3.10 install --user fastapi==0.104.1
pip3.10 install --user uvicorn==0.24.0
pip3.10 install --user pydantic==2.4.2
pip3.10 install --user rapidfuzz==3.5.2
pip3.10 install --user scikit-learn==1.3.2
pip3.10 install --user numpy==1.24.4
pip3.10 install --user pandas==2.1.4
pip3.10 install --user joblib==1.3.2

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your app files to /home/$USER/pair-matcher/"
echo "2. Create a web app in the Web tab"
echo "3. Set source code directory to /home/$USER/pair-matcher/"
echo "4. Upload and configure wsgi.py"
echo ""
echo "Your API will be available at: https://$USER.pythonanywhere.com"