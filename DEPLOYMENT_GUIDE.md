# FastAPI Product Matcher - Quick Start Guide

## 🚀 Get Your API Online in 3 Steps

### Step 1: Update AWS Credentials
```bash
aws configure
```
Enter your new:
- AWS Access Key ID
- AWS Secret Access Key  
- Default region: `us-west-2`
- Output format: `json`

### Step 2: Deploy Your API
```bash
# Check if old environment exists
eb status

# If it exists but is terminated, create new one:
eb create fastapi-env3 --platform "Python 3.9 running on 64bit Amazon Linux 2023"

# If it exists and is healthy, just deploy:
eb deploy
```

### Step 3: Get Your URL
```bash
eb open
```

## 🔗 Your API Endpoints
Once deployed, your API will be available at:
- **Main matching**: `https://your-app-url.elasticbeanstalk.com/match`
- **Model status**: `https://your-app-url.elasticbeanstalk.com/model-status` 
- **Batch similarity**: `https://your-app-url.elasticbeanstalk.com/batch-similarity`

## 🧪 Test Your API
```json
{
  "left": ["Dell Desktop Pro"],
  "right": ["Dell Pro Desktop Computer", "HP Laptop"],
  "use_ai": true,
  "threshold": 0.3
}
```

## ✅ Your App Features
- ✅ AI-powered product matching
- ✅ Fuzzy string matching fallback
- ✅ Model number recognition (iDRAC8, UCK-G2-PLUS)
- ✅ Abbreviation expansion (COM → Commercial)
- ✅ License term filtering
- ✅ CORS enabled for browser clients

Your FastAPI app is ready to deploy! 🎉