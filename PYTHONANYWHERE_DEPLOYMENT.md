# FastAPI Product Matcher - PythonAnywhere Deployment Guide

## 🌟 **Your AI-Powered Product Matching API**

This guide will help you deploy your FastAPI application with AI product matching to PythonAnywhere.

## 📋 **Prerequisites**
- PythonAnywhere account (Free or Paid)
- Your app files ready for upload

## 🚀 **Deployment Steps**

### **Step 1: Create PythonAnywhere Account**
1. Go to [pythonanywhere.com](https://www.pythonanywhere.com)
2. Sign up for an account
3. Your app will be available at: `https://yourusername.pythonanywhere.com`

### **Step 2: Upload Your Files**
Upload these files to `/home/yourusername/pair-matcher/`:
- `app/app.py` (your main FastAPI application)
- `main.py` (entry point)
- `wsgi.py` (WSGI configuration)
- `requirements-pythonanywhere.txt`

### **Step 3: Install Dependencies**
In PythonAnywhere Bash console, run:
```bash
cd /home/yourusername/pair-matcher
pip3.10 install --user -r requirements-pythonanywhere.txt
```

### **Step 4: Create Web App**
1. Go to **Web** tab in PythonAnywhere dashboard
2. Click **Add a new web app**
3. Choose **Manual configuration**
4. Select **Python 3.10**
5. Set **Source code directory** to: `/home/yourusername/pair-matcher/`

### **Step 5: Configure WSGI**
1. In Web tab, click **WSGI configuration file**
2. Replace contents with your `wsgi.py` file content
3. Update the username in the path: `/home/yourusername/pair-matcher`

### **Step 6: Reload and Test**
1. Click **Reload yourusername.pythonanywhere.com**
2. Visit `https://yourusername.pythonanywhere.com/docs`

## 🔗 **Your API Endpoints**

Once deployed, your API will have these endpoints:
- **API Documentation**: `https://yourusername.pythonanywhere.com/docs`
- **Product Matching**: `https://yourusername.pythonanywhere.com/match`
- **Model Status**: `https://yourusername.pythonanywhere.com/model-status`
- **Batch Similarity**: `https://yourusername.pythonanywhere.com/batch-similarity`

## 🧪 **Test Your Deployed API**

```bash
curl -X POST "https://yourusername.pythonanywhere.com/match" \
-H "Content-Type: application/json" \
-d '{
  "left": ["Dell Desktop Pro"],
  "right": ["Dell Pro Desktop Computer", "HP Laptop"],
  "use_ai": true,
  "threshold": 0.3
}'
```

## ✨ **Your AI Features**
- 🤖 **Enhanced AI matching** (0.7-0.9 scores for identical products)
- 🔧 **Smart model recognition** (iDRAC8, UCK-G2-PLUS patterns)
- 📝 **Abbreviation expansion** (COM→Commercial, Pro→Professional)
- 🧹 **Noise filtering** (removes license, subscription terms)
- 🎯 **High accuracy** for Dell, Ubiquiti, Adobe, Microsoft products

## 🆓 **Free vs Paid Accounts**
- **Free**: 1 web app, limited CPU seconds, `yourusername.pythonanywhere.com`
- **Hacker ($5/month)**: Multiple apps, more CPU, custom domains

## 🔧 **Troubleshooting**
- Check **Error logs** in Web tab
- Ensure all files are in correct directory
- Verify Python version (3.10 recommended)
- Check WSGI file path matches your username

Your FastAPI AI Product Matcher is ready for PythonAnywhere! 🎉