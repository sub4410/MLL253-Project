# 🚀 DEPLOYMENT GUIDE

This guide will help you deploy your Stress-Strain Analyzer to the web.

## 📋 Pre-Deployment Checklist

### Backend
- [ ] Remove `debug=True` from `app.py`
- [ ] Set proper CORS origins (not wildcard `*`)
- [ ] Add environment variable support
- [ ] Test with production data
- [ ] Add file size limits

### Frontend
- [ ] Update API URL for production
- [ ] Test build process (`npm run build`)
- [ ] Optimize images if any
- [ ] Check responsive design
- [ ] Test on multiple browsers

## 🌐 Deployment Options

### Option 1: Vercel (Frontend) + Render (Backend)

**Best for**: Quick, free deployment with automatic CI/CD

#### Deploy Backend to Render

1. Create account at [render.com](https://render.com)

2. Push your code to GitHub

3. Create new Web Service on Render:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 4 -b 0.0.0.0:$PORT app:app`
   - **Environment Variables**:
     ```
     PYTHON_VERSION=3.11
     ```

4. Add `gunicorn` to `backend/requirements.txt`:
   ```
   gunicorn==21.2.0
   ```

5. Note your backend URL (e.g., `https://your-app.onrender.com`)

#### Deploy Frontend to Vercel

1. Create account at [vercel.com](https://vercel.com)

2. Import your GitHub repository

3. Configure:
   - **Framework Preset**: Create React App
   - **Root Directory**: `frontend`
   - **Environment Variable**:
     ```
     REACT_APP_API_URL=https://your-app.onrender.com
     ```

4. Update `frontend/src/App.js` to use environment variable:
   ```javascript
   const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
   
   const response = await axios.post(`${API_URL}/api/analyze`, formData, {
     headers: { 'Content-Type': 'multipart/form-data' },
   });
   ```

5. Deploy! Vercel will give you a URL like `https://your-app.vercel.app`

---

### Option 2: Heroku (Full Stack)

**Best for**: Single platform, easy management

#### Setup

1. Install Heroku CLI:
   ```bash
   npm install -g heroku
   ```

2. Login:
   ```bash
   heroku login
   ```

#### Deploy Backend

1. Create `backend/Procfile`:
   ```
   web: gunicorn app:app
   ```

2. Add `gunicorn` to `backend/requirements.txt`

3. Initialize git in backend folder:
   ```bash
   cd backend
   git init
   heroku create your-backend-name
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```

4. Note your backend URL

#### Deploy Frontend

1. Build frontend:
   ```bash
   cd frontend
   npm run build
   ```

2. Install serve:
   ```bash
   npm install -g serve
   ```

3. Create `frontend/Procfile`:
   ```
   web: serve -s build -l $PORT
   ```

4. Deploy:
   ```bash
   heroku create your-frontend-name
   git add .
   git commit -m "Deploy frontend"
   git push heroku main
   ```

---

### Option 3: AWS (Production Ready)

**Best for**: Enterprise, scalability, full control

#### Backend on Elastic Beanstalk

1. Install EB CLI:
   ```bash
   pip install awsebcli
   ```

2. Initialize:
   ```bash
   cd backend
   eb init -p python-3.11 stress-strain-backend
   ```

3. Create environment:
   ```bash
   eb create production
   ```

4. Deploy:
   ```bash
   eb deploy
   ```

#### Frontend on S3 + CloudFront

1. Build frontend:
   ```bash
   cd frontend
   npm run build
   ```

2. Create S3 bucket:
   - Go to AWS S3 console
   - Create bucket (e.g., `stress-strain-analyzer`)
   - Enable static website hosting
   - Upload `build` folder contents

3. Create CloudFront distribution:
   - Origin: Your S3 bucket
   - Default root object: `index.html`
   - Error pages: 403 → `/index.html` (for React Router)

4. Update DNS with CloudFront URL

---

### Option 4: Railway (Modern Alternative)

**Best for**: Simple, affordable, modern DevOps

1. Create account at [railway.app](https://railway.app)

2. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

#### Deploy Backend

1. Navigate to backend:
   ```bash
   cd backend
   railway login
   railway init
   ```

2. Add `railway.json`:
   ```json
   {
     "$schema": "https://railway.app/railway.schema.json",
     "build": {
       "builder": "NIXPACKS"
     },
     "deploy": {
       "startCommand": "gunicorn -w 4 -b 0.0.0.0:$PORT app:app",
       "restartPolicyType": "ON_FAILURE"
     }
   }
   ```

3. Deploy:
   ```bash
   railway up
   ```

#### Deploy Frontend

Same as Vercel (above)

---

## 🔧 Production Configuration

### Update `backend/app.py`

```python
import os

# Production config
DEBUG = os.getenv('FLASK_ENV') != 'production'
ALLOWED_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
```

### Update `frontend/src/App.js`

```javascript
// At the top of App.js
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// In handleAnalyze function
const response = await axios.post(`${API_URL}/api/analyze`, formData, {
  headers: { 'Content-Type': 'multipart/form-data' },
});
```

### Create `frontend/.env.production`

```
REACT_APP_API_URL=https://your-backend-url.com
```

## 🔐 Security Enhancements

### Backend

1. **Add rate limiting**:
   ```bash
   pip install flask-limiter
   ```

   ```python
   from flask_limiter import Limiter
   
   limiter = Limiter(
       app,
       key_func=lambda: request.remote_addr,
       default_limits=["100 per hour"]
   )
   
   @app.route('/api/analyze', methods=['POST'])
   @limiter.limit("10 per minute")
   def analyze_stress_strain():
       # ... existing code
   ```

2. **Add file size limit**:
   ```python
   app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
   ```

3. **Validate CSV format**:
   ```python
   def validate_csv(file):
       # Add validation logic
       pass
   ```

### Frontend

1. **Add error boundaries**
2. **Implement request timeouts**
3. **Add CSRF protection if needed**

## 📊 Monitoring

### Backend Monitoring

- **Sentry** for error tracking
- **New Relic** for performance
- **Datadog** for logs

### Frontend Monitoring

- **Google Analytics** for usage
- **Sentry** for errors
- **LogRocket** for session replay

## 🧪 Testing Before Deployment

```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test

# Build test
npm run build
serve -s build
```

## 📈 Post-Deployment

### Verify Deployment

- [ ] Backend health check: `https://your-backend.com/api/health`
- [ ] Frontend loads: `https://your-frontend.com`
- [ ] File upload works
- [ ] Analysis completes successfully
- [ ] Graph displays correctly
- [ ] Mobile responsive

### Performance Optimization

1. **Enable gzip compression**
2. **Use CDN for static assets**
3. **Implement caching**
4. **Optimize images**
5. **Lazy load components**

### Monitor

- [ ] Set up error alerts
- [ ] Monitor response times
- [ ] Track user analytics
- [ ] Check server resources

## 💰 Cost Estimates

### Free Tier Options
- **Render**: Free (with auto-sleep)
- **Vercel**: Free (hobby plan)
- **Railway**: $5/month credit (trial)
- **Heroku**: Free tier deprecated

### Paid Options
- **Render**: $7/month (always on)
- **Railway**: ~$5-10/month
- **AWS**: $10-50/month (depends on traffic)
- **Heroku**: $7/month per dyno

## 🆘 Troubleshooting Deployment

### Common Issues

**CORS errors in production**
- Update backend CORS origins to include frontend URL

**404 on frontend routes**
- Add redirect rules for SPA routing
- Vercel: `vercel.json` with rewrites
- S3: CloudFront error page configuration

**Backend timeout**
- Increase timeout settings
- Optimize analysis code
- Use async processing for large files

**Environment variables not working**
- Rebuild after adding variables
- Check variable names match exactly
- Verify in deployment platform settings

## 📚 Additional Resources

- [Flask Deployment Options](https://flask.palletsprojects.com/en/2.3.x/deploying/)
- [Create React App Deployment](https://create-react-app.dev/docs/deployment/)
- [Vercel Documentation](https://vercel.com/docs)
- [Render Documentation](https://render.com/docs)
- [Railway Documentation](https://docs.railway.app/)

---

**🎉 Ready to Deploy!**

**Recommended for beginners**: Vercel (frontend) + Render (backend)  
**Recommended for production**: AWS or Railway  
**Easiest**: Railway (both frontend & backend)

Choose your platform and follow the steps above. Good luck! 🚀
