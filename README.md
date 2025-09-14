# 📊 Life Expectancy Analytics Dashboard

A comprehensive Streamlit dashboard for analyzing global life expectancy patterns and their relationship with socioeconomic, health, and environmental factors.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

### Interactive Analytics Tabs
- **📈 Overview** - Key health indicators, distributions, trends, and comparisons
- **🌍 Geographic Analysis** - Interactive world maps and regional patterns
- **📊 Factor Analysis** - Statistical relationships and factor identification
- **🔬 Advanced Analytics** - PCA, clustering, anomaly detection, and predictive modeling
- **💡 Key Insights** - Intelligent pattern discovery and policy recommendations
- **🎯 Interactive Explorer** - Custom visualizations and country comparisons
- **📚 About** - Comprehensive methodology and documentation

### Statistical Methods
- Principal Component Analysis (PCA) with 3D visualization
- K-means clustering with silhouette score optimization
- Random Forest feature importance analysis
- Anomaly detection using Isolation Forest
- Time series trend analysis
- Correlation analysis with significance testing

## 🚀 Live Demo

You can run this dashboard locally using Streamlit.

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/life-expectancy-dashboard.git
cd life-expectancy-dashboard
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run life_expectancy_dashboard.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📊 Data Sources

The dashboard analyzes comprehensive health and socioeconomic data from:
- **World Health Organization (WHO)** - Global health statistics
- **World Bank** - Economic development and GDP data
- **United Nations** - Educational attainment and demographics
- Various national health agencies

### Key Variables Analyzed
- **Health Indicators**: Life expectancy, mortality rates, immunization coverage
- **Economic Factors**: GDP per capita, economic development status
- **Social Determinants**: Years of schooling, population demographics
- **Lifestyle Factors**: BMI, alcohol consumption
- **Disease Prevalence**: HIV/AIDS, hepatitis, measles

## 🏗️ Project Structure

```
life-expectancy-dashboard/
│
├── life_expectancy_dashboard.py    # Main dashboard application
├── Life_expectancy.csv             # Primary dataset
├── enhanced_overview.py            # Enhanced overview module
├── enhanced_geographic.py          # Geographic analysis module
├── enhanced_factor_analysis.py     # Factor analysis engine
├── enhanced_advanced_analytics.py  # Advanced analytics module
├── enhanced_interactive_explorer.py # Interactive exploration tools
├── enhanced_key_insights.py        # Intelligent insights generator
├── enhanced_about.py               # About page with documentation
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 💻 Technology Stack

- **Streamlit** - Interactive web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations and maps
- **Scikit-learn** - Machine learning and statistical modeling
- **SciPy** - Advanced statistical functions

## 📈 Key Features Explained

### 1. Dynamic Filtering
- Filter by countries, regions, years, and economic status
- Real-time updates across all visualizations

### 2. Statistical Validation
- All insights include p-values and significance testing
- Confidence intervals for predictions
- Cross-validation for machine learning models

### 3. Interactive Visualizations
- Hover tooltips with detailed information
- Zoom, pan, and export capabilities
- Multiple chart types and perspectives

### 4. Intelligent Insights
- Automated pattern detection
- Evidence-based policy recommendations
- Outlier and anomaly identification

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Dr. Paul Najarian** - *Initial work and statistical methodology*

## 🙏 Acknowledgments

- World Health Organization for providing comprehensive health data
- World Bank for economic indicators
- Streamlit team for the excellent framework
- Open source community for the statistical libraries

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This dashboard is for educational and research purposes. Always consult with healthcare professionals and policy experts for actual decision-making.