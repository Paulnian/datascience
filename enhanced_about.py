import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_enhanced_about(df):
    """
    Render enhanced About tab with comprehensive information about the application,
    methodology, and data sources.
    """

    # Main header
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1f77b4 0%, #17becf 100%);
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0;'>
            📊 Life Expectancy Analytics Dashboard
        </h1>
        <p style='color: white; text-align: center; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
            Comprehensive Analysis of Global Health Indicators & Life Expectancy Patterns
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different sections
    about_tab1, about_tab2, about_tab3, about_tab4 = st.tabs([
        "🎯 About This Study",
        "📋 Methodology & Approach",
        "📊 Data Overview",
        "🔧 Technical Details"
    ])

    with about_tab1:
        st.subheader("🎯 Study Purpose & Objectives")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### What We're Analyzing

            This comprehensive dashboard analyzes **global life expectancy patterns** and their relationship
            with socioeconomic, health, and environmental factors across countries and time periods.

            ### Key Research Questions

            🔍 **Factor Analysis**: Which factors most significantly influence life expectancy across different regions and economic development levels?

            🌍 **Geographic Patterns**: How do life expectancy trends vary by region, and what geographic clusters emerge?

            📈 **Temporal Trends**: What are the long-term trends in global health indicators, and which countries show the most improvement or decline?

            🏥 **Health Systems**: How do healthcare infrastructure investments correlate with population health outcomes?

            💰 **Economic Impact**: What is the relationship between economic development and health outcomes?

            ### Study Importance

            Understanding life expectancy patterns helps:
            - **Policy makers** identify priority areas for health investments
            - **Researchers** understand complex relationships between social determinants and health
            - **International organizations** allocate resources effectively
            - **Public health professionals** design targeted interventions
            """)

        with col2:
            # Data summary metrics
            st.markdown("### 📈 Dataset Overview")

            if not df.empty:
                total_countries = df['Country'].nunique() if 'Country' in df.columns else 0
                year_range = f"{df['Year'].min()}-{df['Year'].max()}" if 'Year' in df.columns else "N/A"
                total_records = len(df)

                st.metric("Countries Analyzed", total_countries)
                st.metric("Time Period", year_range)
                st.metric("Total Records", f"{total_records:,}")

                # Regional distribution
                if 'Region' in df.columns:
                    region_counts = df['Region'].value_counts()
                    st.markdown("#### Regional Coverage")
                    for region, count in region_counts.head(5).items():
                        st.text(f"• {region}: {count}")

    with about_tab2:
        st.subheader("📋 Analytical Methodology")

        # Statistical methods section
        st.markdown("### 🔬 Statistical Methods Employed")

        method_col1, method_col2 = st.columns(2)

        with method_col1:
            st.markdown("""
            #### **Factor Analysis**
            - **Principal Component Analysis (PCA)** for dimensionality reduction
            - **True Factor Analysis** with latent variable identification
            - **Feature importance** using Random Forest and permutation methods
            - **Multicollinearity detection** via Variance Inflation Factor (VIF)

            #### **Clustering Analysis**
            - **K-means clustering** with silhouette score optimization
            - **Hierarchical clustering** for country groupings
            - **Anomaly detection** using Isolation Forest algorithms
            - **Outlier identification** via statistical Z-scores
            """)

        with method_col2:
            st.markdown("""
            #### **Predictive Modeling**
            - **Multiple regression** with cross-validation
            - **Random Forest** for non-linear relationships
            - **Time series analysis** for trend detection
            - **Statistical significance testing** (ANOVA, correlation tests)

            #### **Geographic Analysis**
            - **Spatial clustering** and regional comparisons
            - **Interactive mapping** with temporal controls
            - **Geographic trend analysis** across regions
            - **Country-level comparative analysis**
            """)

        st.markdown("### 🎯 Data Processing Pipeline")

        st.markdown("""
        ```
        Raw Data → Data Cleaning → Feature Engineering → Statistical Analysis → Visualization → Insights
        ```

        1. **Data Cleaning**: Handle missing values, outlier detection, data type corrections
        2. **Feature Engineering**: Create derived variables, economic status indicators, regional groupings
        3. **Statistical Analysis**: Apply multiple analytical methods for comprehensive insights
        4. **Visualization**: Interactive charts, maps, and dashboards for exploration
        5. **Insight Generation**: Automated pattern detection and recommendation systems
        """)

    with about_tab3:
        st.subheader("📊 Data Sources & Variables")

        # Data source information
        st.markdown("""
        ### 🌐 Primary Data Sources

        This analysis is based on comprehensive health and socioeconomic data compiled from multiple authoritative sources:

        - **World Health Organization (WHO)** - Global health statistics and indicators
        - **World Bank** - Economic development and GDP data
        - **United Nations** - Educational attainment and demographic data
        - **Various National Health Agencies** - Country-specific health metrics
        """)

        # Variables breakdown
        if not df.empty:
            st.markdown("### 📋 Key Variables Analyzed")

            # Categorize columns
            health_vars = [col for col in df.columns if any(term in col.lower() for term in
                          ['life_expectancy', 'mortality', 'deaths', 'hepatitis', 'polio', 'diphtheria', 'hiv', 'measles'])]

            economic_vars = [col for col in df.columns if any(term in col.lower() for term in
                           ['gdp', 'economy', 'income'])]

            social_vars = [col for col in df.columns if any(term in col.lower() for term in
                          ['schooling', 'education', 'population'])]

            lifestyle_vars = [col for col in df.columns if any(term in col.lower() for term in
                            ['alcohol', 'bmi', 'thinness'])]

            var_col1, var_col2 = st.columns(2)

            with var_col1:
                if health_vars:
                    st.markdown("#### 🏥 Health Indicators")
                    for var in health_vars[:8]:
                        st.text(f"• {var}")

                if economic_vars:
                    st.markdown("#### 💰 Economic Factors")
                    for var in economic_vars:
                        st.text(f"• {var}")

            with var_col2:
                if social_vars:
                    st.markdown("#### 🎓 Social Determinants")
                    for var in social_vars:
                        st.text(f"• {var}")

                if lifestyle_vars:
                    st.markdown("#### 🍎 Lifestyle Factors")
                    for var in lifestyle_vars:
                        st.text(f"• {var}")

        # Data quality section
        st.markdown("### ✅ Data Quality & Limitations")

        quality_col1, quality_col2 = st.columns(2)

        with quality_col1:
            st.markdown("""
            #### **Data Strengths**
            - Multiple authoritative sources
            - Comprehensive country coverage
            - Multi-year time series data
            - Standardized WHO methodologies
            - Regular data updates and validation
            """)

        with quality_col2:
            st.markdown("""
            #### **Known Limitations**
            - Missing data for some countries/years
            - Varying data collection methodologies
            - Time lags in official reporting
            - Economic data may not reflect informal sectors
            - Rural/urban disparities not captured
            """)

    with about_tab4:
        st.subheader("🔧 Technical Implementation")

        # Technology stack
        tech_col1, tech_col2 = st.columns(2)

        with tech_col1:
            st.markdown("""
            ### 🐍 Technology Stack

            #### **Core Framework**
            - **Streamlit** - Interactive web application framework
            - **Python 3.8+** - Primary programming language
            - **Pandas** - Data manipulation and analysis
            - **NumPy** - Numerical computing

            #### **Statistical Libraries**
            - **Scikit-learn** - Machine learning and statistical modeling
            - **SciPy** - Advanced statistical functions
            - **Statsmodels** - Statistical analysis and econometrics

            #### **Visualization**
            - **Plotly** - Interactive charts and maps
            - **Seaborn & Matplotlib** - Statistical visualizations
            """)

        with tech_col2:
            st.markdown("""
            ### 🏗️ Architecture Features

            #### **Modular Design**
            - Separate modules for each analytical component
            - Enhanced factor analysis engine
            - Geographic analysis system
            - Interactive exploration tools

            #### **Performance Optimization**
            - Efficient data caching mechanisms
            - Optimized statistical computations
            - Responsive user interface design
            - Memory-efficient data handling

            #### **User Experience**
            - Interactive filtering and selection
            - Real-time chart updates
            - Export capabilities for further analysis
            - Mobile-responsive design
            """)

        # Usage instructions
        st.markdown("### 📖 How to Use This Dashboard")

        st.markdown("""
        1. **Filter Data** - Use sidebar controls to select countries, regions, and time periods
        2. **Explore Tabs** - Each tab provides different analytical perspectives:
           - **Overview** - Key health indicators and trends
           - **Geographic** - Regional patterns and mapping
           - **Factor Analysis** - Statistical relationships and factor identification
           - **Advanced Analytics** - PCA, clustering, and predictive modeling
           - **Interactive Explorer** - Custom visualizations and comparisons
           - **Key Insights** - Automated pattern discovery and recommendations
        3. **Interpret Results** - Look for statistical significance indicators and confidence intervals
        4. **Export Data** - Use the raw data view to download filtered datasets
        """)

        # Contact/credits
        st.markdown("### 👥 Development & Credits")

        st.markdown("""
        **Application Development**: Dr. Paul Najarian

        **Statistical Methodology**: Applied using established epidemiological and econometric methods

        **Data Sources**: WHO, World Bank, UN, and National Health Agencies

        ---

        *For technical questions or suggestions for improvement, please refer to the application documentation.*
        """)

def get_data_summary_stats(df):
    """Generate summary statistics for the about page"""
    if df.empty:
        return {}

    stats = {
        'total_countries': df['Country'].nunique() if 'Country' in df.columns else 0,
        'total_records': len(df),
        'year_range': f"{df['Year'].min()}-{df['Year'].max()}" if 'Year' in df.columns else "N/A",
        'regions': df['Region'].nunique() if 'Region' in df.columns else 0,
        'variables': len(df.columns)
    }

    return stats