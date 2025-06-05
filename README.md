# Simple Flask Backend

A minimal Flask backend application designed for Vercel deployment.

## Endpoints

- `/`: Welcome message
- `/api/health`: Health check endpoint

## Local Development

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
# or
.\venv\Scripts\activate  
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python app.py
```