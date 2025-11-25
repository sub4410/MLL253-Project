# Stress-Strain Analysis Backend

Flask backend API for processing stress-strain data and performing mechanical testing analysis.

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the backend server:**
   ```bash
   python app.py
   ```

   The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/api/health`
- Returns server status

### Analyze Stress-Strain Data
- **POST** `/api/analyze`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file`: CSV file with strain (%) and stress (MPa) columns
  - `testType`: "Compression Test" or "Tensile Test" (default: "Compression Test")
  - `outputDecimalPlaces`: Number of decimal places (default: 3)
  - `stressAtCertainStrain`: Strain value for stress calculation (default: 5)
  - Additional parameters for elastic modulus, yield stress, and plateau analysis

- **Returns:** JSON with analysis results including:
  - Elastic Modulus
  - Yield Stress (0.2%)
  - UTS (Ultimate Tensile Strength)
  - Area Under Curve (Resilience)
  - Stress-Strain graph (base64 encoded PNG)
  - Test-specific results

## CSV File Format

The CSV file should have two columns:
1. Strain (in %)
2. Stress (in MPa)

Example:
```
0.0,0.0
0.5,10.2
1.0,20.5
...
```

## Deployment

For production deployment, use a production-ready WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
