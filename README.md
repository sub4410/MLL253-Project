# 🔬 Stress-Strain Analyzer - Mathematical/Computational Approach

## Overview

A web-based tool for **automatic mechanical property extraction from stress-strain curves** using **pure mathematical and computational algorithms** (NO Machine Learning).

### Key Features

✅ **Automatic Detection:**
- Elastic region with R² threshold analysis
- 0.2% offset yield point calculation
- Ultimate Tensile Strength (UTS) identification
- Necking region detection

✅ **Smooth Curves:**
- Savitzky-Golay filter for noise reduction
- Preserves peaks and critical features
- Adjustable smoothing parameters

✅ **Comprehensive Analysis:**
1. **Young's Modulus (E)** - Elastic stiffness
2. **Yield Strength** - 0.2% offset method
3. **Ultimate Tensile Strength (UTS)** - Maximum stress
4. **% Elongation** - Ductility measure
5. **Resilience** - Elastic energy absorption
6. **Toughness** - Total energy absorption
7. **Strain Hardening Exponent (n)** - Plastic behavior

✅ **Visual Output:**
- Color-coded regions (Elastic=Blue, Plastic=Green, Necking=Red)
- Marked yield point and UTS
- Shaded resilience and toughness areas
- Comprehensive properties table
- Region distribution pie chart
- Key metrics bar chart

---

## Quick Start

### 1. **Start Backend** (Python/Flask)
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Backend runs on: `http://localhost:5000`

### 2. **Start Frontend** (React)
```bash
cd frontend
npm install
npm start
```
Frontend runs on: `http://localhost:3000`

### 3. **Upload CSV Data**
- Format: **2 columns minimum** - `Strain`, `Stress`
- Example:
  ```csv
  Strain,Stress
  0.0001,20
  0.0002,40
  ...
  ```
- Drag & drop or click to select file
- Click **Analyze** button

---

## How It Works (Mathematical Approach)

### Step 1: Data Smoothing
- **Algorithm**: Savitzky-Golay filter
- **Purpose**: Remove noise while preserving curve features
- **Parameters**: 
  - Window size (default: 11 points)
  - Polynomial order (default: 3)

### Step 2: Elastic Region Detection
- **Method**: Sliding window with R² threshold
- **Algorithm**:
  1. Fit linear regression to increasing data subsets
  2. Calculate R² for each fit
  3. Stop when R² drops below 0.998
- **Output**: Young's Modulus (E) = slope of elastic region

### Step 3: 0.2% Offset Yield Point
- **Formula**: Offset line = E × (ε - 0.002)
- **Method**: Find intersection of offset line with stress-strain curve
- **Output**: Yield strength (σ_y) and yield strain (ε_y)

### Step 4: Ultimate Tensile Strength (UTS)
- **Method**: Find maximum stress value in curve
- **Output**: UTS and corresponding strain

### Step 5: % Elongation
- **Formula**: % Elongation = (final strain / original length) × 100
- **Output**: Ductility measure

### Step 6: Resilience Calculation
- **Formula**: Resilience = ∫₀^εʸ σ dε
- **Method**: Trapezoidal integration under elastic curve
- **Output**: Elastic energy absorption (MPa)

### Step 7: Toughness Calculation
- **Formula**: Toughness = ∫₀^εᶠ σ dε
- **Method**: Trapezoidal integration under entire curve
- **Output**: Total energy absorption (MPa)

### Step 8: Strain Hardening Analysis
- **Formula**: σ = K × εⁿ (Hollomon equation)
- **Method**: Log-log linear regression on plastic region
- **Output**: Strength coefficient (K) and strain hardening exponent (n)

---

## API Documentation

### Endpoint: `/api/analyze`
**Method**: POST  
**Content-Type**: multipart/form-data

**Request Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | *required* | CSV file with Strain, Stress columns |
| `outputDecimalPlaces` | integer | 3 | Decimal precision for results |
| `smoothingWindow` | integer | 11 | Savitzky-Golay window size (odd number) |
| `smoothingOrder` | integer | 3 | Polynomial order for smoothing (2-5) |

**Response (JSON):**
```json
{
  "youngsModulus": 200000.0,
  "yieldStress": 400.0,
  "yieldStrain": 0.002,
  "UTS": 920.0,
  "percentElongation": 14.0,
  "resilience": 0.4,
  "toughness": 100.0,
  "strainHardeningExponent": 0.25,
  "strengthCoefficient": 1200.0,
  "elasticDataPoints": 20,
  "plasticDataPoints": 40,
  "neckingDataPoints": 15,
  "elasticR2": 0.9995,
  "graphData": "data:image/png;base64,..."
}
```

---

## Sample CSV Data

See `sample_stress_strain.csv` for example format:
- 60+ data points
- Elastic region (0-0.002 strain)
- Plastic region with strain hardening
- Necking region (strain softening after UTS)

---

## Technologies Used

### Backend
- **Flask 3.0.0** - Web framework
- **NumPy 1.26.2** - Numerical computations
- **Pandas 2.1.4** - Data handling
- **SciPy 1.11.4** - Signal processing & integration
- **Matplotlib 3.8.2** - Graph generation

### Frontend
- **React 18.2.0** - UI framework
- **Axios** - HTTP client
- **react-dropzone** - File upload

---

## Algorithm Advantages

### Why Mathematical vs Machine Learning?

✅ **Interpretability**: Every calculation is based on established mechanical engineering principles  
✅ **Accuracy**: No training data needed, works with any material  
✅ **Speed**: Instant analysis (no model training)  
✅ **Transparency**: Clear mathematical formulas for each property  
✅ **Robustness**: Handles noisy data with smoothing filters  
✅ **Standards Compliance**: Uses ASTM standard methods (0.2% offset)

---

## Troubleshooting

### Backend Issues
**Error: "Insufficient data points"**
- Ensure CSV has at least 20 rows
- Check for missing/NaN values

**Error: "CSV must contain Strain and Stress columns"**
- First row should be headers: `Strain,Stress`
- Or ensure exactly 2 columns (auto-named)

### Frontend Issues
**CORS errors**
- Backend running on port 5000?
- Check `flask-cors` is installed

**Graph not displaying**
- Clear browser cache
- Check browser console for errors

---

## Project Structure

```
MLL253 Project/
├── backend/
│   ├── app.py                  # Mathematical analysis API
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.js   # CSV upload component
│   │   │   ├── ParametersForm.js # Smoothing controls
│   │   │   └── Results.js      # Property display
│   │   └── App.js              # Main React app
│   └── package.json
├── sample_stress_strain.csv    # Test data
└── README.md                   # This file
```

---

## References

1. **ASTM E8** - Standard Test Methods for Tension Testing
2. **0.2% Offset Method** - Yield strength determination
3. **Savitzky-Golay Filter** - Smoothing and differentiation of data
4. **Hollomon Equation** - σ = K·εⁿ (plastic deformation)

---

## License

MIT License - Feel free to use and modify

---

## Authors

MLL253 Project Team  
November 2025

---

## Need Help?

- Check `sample_stress_strain.csv` for correct format
- See backend console for detailed error traces
- Frontend errors appear in browser console (F12)
