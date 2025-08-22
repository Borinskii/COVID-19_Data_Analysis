# 🦠 COVID-19 Intelligence Dashboard

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

Interactive analytics platform with machine learning-powered forecasting, clustering analysis, and real-time data visualization for COVID-19 epidemiological data.

## 🚀 Features

- **Interactive Dashboards**: Real-time visualization with Plotly & Dash
- **ML Analytics**: Time series forecasting, clustering analysis, correlation studies
- **Multi-Source Data**: COVID cases, vaccinations, mobility, economic indicators
- **RESTful API**: FastAPI backend with Snowflake integration
- **Comment System**: Collaborative annotations with MongoDB

## 🏗️ Architecture

**Technology Stack:**
- **Frontend**: Dash, Plotly, Bootstrap Components
- **Backend**: FastAPI, Uvicorn
- **Databases**: Snowflake (analytics), MongoDB (comments)
- **ML/Analytics**: Scikit-learn, Scipy, NumPy, Pandas

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- **Snowflake Account** (required)
- MongoDB (optional, for comments)

### 1. Clone & Setup
```cmd
git clone https://github.com/Borinskii/COVID-19_Data_Analysis.git
cd COVID-19_Data_Analysis

# Create virtual environment
# Windows:
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
```cmd
# Windows:
copy .env.example .env

# macOS/Linux:
cp .env.example .env

# Then edit .env file with your actual credentials
```

**Edit `.env` with your actual values:**
```bash
# Snowflake (Required)
SNOWFLAKE_ACCOUNT=mycompany.us-east-1.aws
SNOWFLAKE_USER=john.doe
SNOWFLAKE_PASSWORD=SecurePassword123
SNOWFLAKE_ROLE=ACCOUNTADMIN
SNOWFLAKE_WAREHOUSE=COMPUTE_WH

# MongoDB (Optional - for comments)
MONGODB_URI=mongodb+srv://myuser:mypassword@covid-cluster.abc123.mongodb.net/
MONGODB_DB=COVID_COMMENTS

# API
API_HOST=0.0.0.0
API_PORT=8003
```

### 3. Run the Application

**Terminal 1 - Start API:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```
✅ API: `http://localhost:8003` | Docs: `http://localhost:8003/docs`

**Terminal 2 - Start Dashboard:**
```bash
python covid_19_dashboard.py
```
✅ Dashboard: `http://localhost:8050`

## 📊 Data Sources

- **Johns Hopkins University**: COVID-19 cases & deaths
- **Our World in Data**: Vaccination statistics
- **Apple Mobility**: Transportation trends
- **World Bank**: Economic indicators (GDP, HDI)
- **CDC**: Testing & ICU capacity data

## 🎯 Key Features

### Dashboard Tabs
- **Cases**: Confirmed cases, deaths, daily trends with 7-day averages
- **Vaccinations**: Progress tracking and per capita rates
- **Mobility**: Transportation trends (driving, walking, transit)
- **Economy**: GDP per capita and Human Development Index
- **🔮 Forecasting**: 30-day ML predictions with confidence intervals
- **🎯 Clustering**: Country grouping by pandemic patterns (K-means + PCA)
- **Correlations**: Statistical relationships between metrics
- **Comments**: Collaborative data annotations

### ML Analytics
- **Forecasting**: Exponential smoothing with 95% confidence intervals
- **Clustering**: K-means clustering to identify pandemic patterns
- **Correlation**: Pearson correlation analysis with significance testing

## 🔌 API Endpoints

```
GET /api/v1/covid-cases          # COVID-19 data
GET /api/v1/vaccinations         # Vaccination stats
GET /api/v1/mobility             # Transportation trends
GET /api/v1/enriched-data        # Economic indicators
GET /api/v1/countries            # Available countries
GET /health                      # System health check
POST /api/v1/comments            # Create annotation
```

## 🛠️ Project Structure
```
COVID-19_Data_Analysis/
├── covid_19_dashboard.py     # Main dashboard
├── main.py                   # FastAPI backend
├── requirements.txt          # Dependencies
├── .env.example             # Environment template
├── KaggleDatasets/          # CSV data files
├── mongodb/                 # MongoDB scripts
└── sql/                     # SQL queries
```

## 🐛 Troubleshooting

**Snowflake Connection Issues:**
```bash
# Verify .env format (account should include region.cloud)
# Example: SNOWFLAKE_ACCOUNT=mycompany.us-east-1.aws
# Check credentials and warehouse status
```

**Dashboard Won't Load:**
```bash
# Install dependencies: pip install -r requirements.txt
# Check port availability: netstat -an | findstr 8050
```

**API Errors:**
```bash
# Start with: uvicorn main:app --host 0.0.0.0 --port 8003 --reload
# Check API status: http://localhost:8003/health
# Test endpoints: http://localhost:8003/docs
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Johns Hopkins University, Our World in Data, Apple, World Bank, CDC for data
- Plotly, FastAPI, and the open source community

---

**⭐ Star this project if it helped you!**
