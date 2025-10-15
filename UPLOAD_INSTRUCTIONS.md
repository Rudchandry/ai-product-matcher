# Files to Upload to PythonAnywhere

## Upload these files to /home/yourusername/pair-matcher/

### 1. Core Application Files
- `main.py` (FastAPI entry point)
- `app/app.py` (your main FastAPI application)
- `wsgi.py` (WSGI configuration for PythonAnywhere)

### 2. Dependencies
- `requirements-pythonanywhere.txt` (Python packages)

### 3. Optional
- `README.md` or `PYTHONANYWHERE_DEPLOYMENT.md` (documentation)

## Step-by-Step Upload Process

### Method 1: File Manager (Easiest)
1. Go to PythonAnywhere Dashboard
2. Click "Files" tab
3. Navigate to `/home/yourusername/`
4. Create folder: `pair-matcher`
5. Upload files using "Upload a file" button

### Method 2: Git (If you have GitHub)
```bash
cd /home/yourusername
git clone your-repository-url pair-matcher
```

### Method 3: Bash Console Upload
1. Go to "Consoles" tab
2. Start a Bash console
3. Use wget or curl to download files from a URL