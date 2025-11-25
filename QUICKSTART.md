# 🚀 QUICK START GUIDE

## Installation (One-Time Setup)

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
cd frontend
npm install
```

---

## Running the Application

### Terminal 1: Start Backend
```bash
cd backend
python app.py
```
✅ Backend running on `http://localhost:5000`

### Terminal 2: Start Frontend
```bash
cd frontend
npm start
```
✅ Frontend opens automatically at `http://localhost:3000`

---

## Usage

### Step 1: Upload CSV
- Drag & drop or click to select
- Format: **Strain, Stress** (2 columns)
- Example: `sample_stress_strain.csv`

### Step 2: Adjust Parameters (Optional)
- **Decimal Places**: Output precision (0-10)
- **Smoothing Window**: 11 (default) - odd number 5-51
- **Smoothing Order**: 3 (default) - polynomial 2-5

### Step 3: Analyze
- Click **"Analyze Data"** button
- Wait 1-2 seconds

### Step 4: View Results
- **Numerical Values**: Young's Modulus, Yield, UTS, % Elongation, etc.
- **Graph**: Color-coded regions with marked key points
- **Charts**: Pie chart (region distribution) + Bar chart (properties)

---

## Expected Output

### Mechanical Properties
| Property | Description | Units |
|----------|-------------|-------|
| **Young's Modulus (E)** | Elastic stiffness | MPa |
| **Yield Stress** | 0.2% offset yield strength | MPa |
| **UTS** | Ultimate tensile strength | MPa |
| **% Elongation** | Ductility measure | % |
| **Resilience** | Elastic energy | MPa |
| **Toughness** | Total energy | MPa |

### Graph Features
- 🔵 **Blue line**: Elastic region
- 🟢 **Green line**: Plastic region (strain hardening)
- 🔴 **Red line**: Necking region (if present)
- 🟡 **Yellow dot**: Yield point (0.2% offset)
- 🟣 **Magenta dot**: UTS (maximum stress)
- 🩵 **Cyan shade**: Resilience area
- 🟩 **Green shade**: Toughness area

---

## Sample Data Format

```csv
Strain,Stress
0.0001,20
0.0002,40
0.0003,60
...
0.1,915
0.12,918
0.14,896
```

**Requirements:**
- First row: Column headers `Strain,Stress`
- Minimum 20 data points
- Sorted by increasing strain
- No missing values

---

## Troubleshooting

### ❌ "Backend not running"
```bash
# Check if backend started:
cd backend
python app.py
# Look for: "Running on http://127.0.0.1:5000"
```

### ❌ "CORS error"
```bash
# Reinstall flask-cors:
pip install flask-cors==4.0.0
```

### ❌ "Insufficient data points"
- Ensure CSV has 20+ rows
- Remove any empty rows
- Check for NaN values

### ❌ "Graph not displaying"
- Refresh browser (Ctrl+R)
- Clear cache (Ctrl+Shift+Del)
- Check browser console (F12)

---

## What Makes This Special?

### ✅ No Machine Learning!
- Pure mathematical/computational approach
- Faster and more accurate
- Transparent formulas

### ✅ Smooth Curves
- Savitzky-Golay filter removes noise
- Preserves important features
- Adjustable smoothing

### ✅ Automatic Detection
- Elastic region (R² threshold)
- 0.2% offset yield point
- UTS (maximum stress)
- Necking detection

### ✅ Standards Compliant
- ASTM E8 method
- Industry-standard calculations
- Professional quality results

---

## File Structure

```
MLL253 Project/
│
├── backend/
│   ├── app.py              ← Mathematical analysis API
│   └── requirements.txt    ← Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/     ← React UI components
│   │   └── App.js          ← Main app
│   └── package.json        ← Node dependencies
│
├── sample_stress_strain.csv ← Test data
├── README.md               ← Full documentation
├── IMPLEMENTATION_SUMMARY.md ← Technical details
└── QUICKSTART.md           ← This file!
```

---

## Need Help?

1. **Check sample CSV**: `sample_stress_strain.csv`
2. **Read full docs**: `README.md`
3. **Technical details**: `IMPLEMENTATION_SUMMARY.md`
4. **Backend errors**: Check terminal running `python app.py`
5. **Frontend errors**: Press F12 in browser

---

## Next Steps

### Test with Sample Data
```bash
# 1. Start backend (Terminal 1)
cd backend && python app.py

# 2. Start frontend (Terminal 2)
cd frontend && npm start

# 3. Upload sample_stress_strain.csv
# 4. Click "Analyze Data"
# 5. View beautiful results! 🎉
```

### Use Your Own Data
- Export stress-strain data from testing machine
- Format as CSV: `Strain,Stress`
- Upload and analyze!

---

## 🎯 That's It!

Your stress-strain analyzer is ready to use with:
- ✅ Smooth curves (Savitzky-Golay filter)
- ✅ Automatic property detection
- ✅ Professional-quality graphs
- ✅ All 7 mechanical properties
- ✅ Zero machine learning needed!

**Happy Analyzing! 🚀**
