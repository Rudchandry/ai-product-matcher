@echo off
echo ==================================================
echo  FastAPI Product Matcher - AWS Deployment Script
echo ==================================================
echo.

cd /d "C:\Users\RudchandryHodge(ITSo\OneDrive - IT Solutions NV\Documents\pair-matcher"

echo Step 1: Activating virtual environment...
call ".\.venv\Scripts\Activate.ps1"

echo Step 2: Checking AWS credentials...
aws sts get-caller-identity
if %errorlevel% neq 0 (
    echo ERROR: AWS credentials invalid. Please run 'aws configure' first.
    pause
    exit /b 1
)

echo Step 3: Checking environment status...
".\.venv\Scripts\eb.exe" status

echo Step 4: Deploying to fastapi-env2...
".\.venv\Scripts\eb.exe" deploy

if %errorlevel% equ 0 (
    echo.
    echo ✅ Deployment successful!
    echo 🌐 Your API should be available at:
    echo    http://fastapi-env2.eba-hpgdz7xf.us-west-2.elasticbeanstalk.com
    echo.
) else (
    echo.
    echo ❌ Deployment failed. Environment might be terminated.
    echo 🔧 Trying to recreate environment...
    ".\.venv\Scripts\eb.exe" create fastapi-env2 --platform "Python 3.9 running on 64bit Amazon Linux 2023"
)

echo.
echo 🎉 Deployment process complete!
pause