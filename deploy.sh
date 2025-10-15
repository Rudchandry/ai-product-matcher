#!/bin/bash
# FastAPI Deployment Script for AWS Elastic Beanstalk

echo "🚀 FastAPI Product Matcher Deployment Script"
echo "=============================================="

echo "Step 1: Checking AWS credentials..."
aws sts get-caller-identity
if [ $? -ne 0 ]; then
    echo "❌ AWS credentials invalid. Please run 'aws configure' first"
    exit 1
fi

echo "Step 2: Checking EB environment status..."
eb status

echo "Step 3: Deploying application..."
eb deploy

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo "🌐 Getting application URL..."
    eb open
else
    echo "❌ Deployment failed. Trying to create new environment..."
    eb create fastapi-env3 --platform "Python 3.9 running on 64bit Amazon Linux 2023"
fi

echo "🎉 Done! Your API should now be available online."