# Stress-Strain Analyzer Web Application

A modern web application for analyzing stress-strain data from mechanical testing. Upload CSV files and get comprehensive analysis including UTS, yield point, resilience, and interactive graphs.

## 🚀 Features

- **CSV File Upload**: Drag-and-drop interface for uploading stress-strain data
- **Comprehensive Analysis**: 
  - Elastic Modulus calculation
  - Yield Stress (0.2% offset method)
  - Ultimate Tensile Strength (UTS)
  - Resilience (Area under curve)
  - Stress at specified strain
  - Breaking stress (Tensile tests)
  - Plateau analysis with dip detection (Compression tests)
- **Interactive Graphs**: Visual stress-strain curves
- **Configurable Parameters**: Advanced settings for fine-tuning analysis
- **Support for Both Test Types**: Tensile and Compression tests

## 📋 Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 14+** and npm (for frontend)
- Modern web browser

## 🛠️ Installation & Setup

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server:**
   ```bash
   python app.py
   ```
   
   The backend will start on `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```
   
   The frontend will open at `http://localhost:3000`

## 📊 CSV File Format

Your CSV file should contain two columns (no headers):
- **Column 1**: Strain (in %)
- **Column 2**: Stress (in MPa)

Example:
```csv
0.0,0.0
0.5,10.2
1.0,20.5
1.5,30.8
2.0,40.1
```

You can use the `testData.csv` file included in the project for testing.

## 🎯 Usage

1. **Start both backend and frontend servers** (see Installation & Setup)

2. **Open the application** in your browser at `http://localhost:3000`

3. **Upload your CSV file** by dragging and dropping or clicking the upload area

4. **Configure parameters** (optional):
   - Select test type (Compression or Tensile)
   - Adjust decimal places for results
   - Set strain value for stress calculation
   - Expand "Advanced Parameters" for fine-tuning

5. **Click "Run Analysis"** to process your data

6. **View results**:
   - Key metrics displayed in cards
   - Stress-strain curve visualization
   - Plateau analysis details (for compression tests)

## 🔧 Configuration Parameters

### Basic Parameters
- **Test Type**: Compression Test or Tensile Test
- **Decimal Places**: Number of decimal places in results (0-10)
- **Stress at Strain**: Strain percentage to calculate corresponding stress

### Advanced Parameters

#### Elastic Modulus
- **Starting Strain**: Initial strain for modulus calculation
- **Finding Step**: Step size for regression (higher = faster but less accurate)
- **R² Minimum**: Minimum correlation coefficient threshold
- **Steps After R² Min**: Detection steps after threshold
- **Backtrack Value**: Steps to backtrack for final calculation

#### Yield Stress
- **Accuracy**: Decimal precision for yield stress line
- **Finding Step**: Step size for data points on yield line
- **Low Modulus**: Threshold for low elastic modulus (MPa)
- **Cutting Range**: Strain range for high modulus case (%)

#### Plateau Analysis (Compression Only)
- **Segment Length**: Analysis segment size for dip detection
- **Defining Factor**: Multiplier of yield stress to define plateau end

## 📦 Deployment

### Backend Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Or use Docker:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
COPY analysisFunctions.py .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Frontend Deployment

Build the production version:

```bash
cd frontend
npm run build
```

The `build` folder can be deployed to:
- **Netlify**: Drag and drop the build folder
- **Vercel**: Connect your GitHub repo
- **AWS S3 + CloudFront**: Upload build folder to S3
- **GitHub Pages**: Configure in package.json

Update the backend URL in production by modifying the proxy or using environment variables.

## 🌐 Environment Variables

### Backend
Create a `.env` file in the backend directory:
```
FLASK_ENV=production
PORT=5000
CORS_ORIGINS=https://your-frontend-domain.com
```

### Frontend
Create a `.env` file in the frontend directory:
```
REACT_APP_API_URL=https://your-backend-domain.com
```

Then update axios calls to use `process.env.REACT_APP_API_URL`

## 📝 API Documentation

### Endpoints

#### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Backend is running"
}
```

#### Analyze Stress-Strain Data
```http
POST /api/analyze
Content-Type: multipart/form-data
```

**Parameters:**
- `file`: CSV file (required)
- `testType`: "Compression Test" or "Tensile Test"
- `outputDecimalPlaces`: integer
- `stressAtCertainStrain`: float
- Additional parameters (see Configuration section)

**Response:**
```json
{
  "elasticModulus": 200.5,
  "yieldStress": 250.3,
  "UTS": 450.7,
  "areaUnderCurve": 1250.8,
  "stressAtStrain": {
    "strain": 5,
    "stress": 350.2
  },
  "graphData": "data:image/png;base64,...",
  "breakingStress": 420.5,
  "plateauAnalysis": {
    "numDips": 3,
    "dipDeltaStress": ["10.5", "12.3", "11.8"],
    "dipDeltaStrain": ["0.5", "0.6", "0.5"],
    "description": "..."
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is provided as-is for educational and research purposes.

## 🐛 Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError: No module named 'flask'`
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Problem**: Backend can't find `analysisFunctions.py`
- **Solution**: Ensure `analysisFunctions.py` is in the project root directory

**Problem**: CORS errors
- **Solution**: Verify `flask-cors` is installed and CORS is properly configured

### Frontend Issues

**Problem**: `npm install` fails
- **Solution**: Delete `node_modules` and `package-lock.json`, then run `npm install` again

**Problem**: Can't connect to backend
- **Solution**: Ensure backend is running on port 5000 and proxy is configured in `package.json`

**Problem**: File upload doesn't work
- **Solution**: Check browser console for errors and ensure CSV format is correct

## 📧 Support

For issues or questions:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

## 🔮 Future Enhancements

- [ ] Multiple file upload and batch processing
- [ ] Export results to PDF/Excel
- [ ] Save and load parameter presets
- [ ] Comparison of multiple test results
- [ ] User authentication and result history
- [ ] Real-time graph interaction and zoom
- [ ] Mobile-responsive improvements
- [ ] Dark mode support

---

**Built with ❤️ using React, Flask, and Python scientific libraries**
