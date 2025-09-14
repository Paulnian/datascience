import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Import enhanced analysis functions
from enhanced_factor_analysis import (
    perform_true_factor_analysis,
    calculate_enhanced_feature_importance,
    analyze_interaction_effects,
    identify_multicollinearity,
    render_enhanced_factor_analysis
)
from enhanced_overview import render_enhanced_overview
from enhanced_geographic import render_enhanced_geographic_analysis
from enhanced_advanced_analytics import render_enhanced_advanced_analytics
from enhanced_interactive_explorer import render_enhanced_interactive_explorer
from enhanced_key_insights import render_enhanced_key_insights
from enhanced_about import render_enhanced_about
from analytics_tracker import integrate_analytics, AnalyticsTracker

# Page configuration
st.set_page_config(
    page_title="Life Expectancy Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Analytics and Custom CSS
st.markdown("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-2HLVL8PYXQ"></script>
<script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-2HLVL8PYXQ');
    gtag('event', 'page_view', {
        'page_title': 'Life Expectancy Dashboard',
        'page_location': window.location.href
    });
</script>

<style>
    /* Remove top padding/margin */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #f0f2f6 0%, #e6e9ef 100%);
        border-radius: 10px;
        margin-bottom: 1.5rem;
        margin-top: 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insights-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('Life-Expectancy-Data-Updated.csv')
    # Data preprocessing
    df['Year'] = pd.to_numeric(df['Year'])
    df['Life_expectancy'] = pd.to_numeric(df['Life_expectancy'])
    # Create decade column for additional analysis
    df['Decade'] = (df['Year'] // 10) * 10
    # Create life expectancy categories
    df['Life_Exp_Category'] = pd.cut(df['Life_expectancy'],
                                      bins=[0, 60, 70, 80, 100],
                                      labels=['Low (<60)', 'Medium (60-70)', 'High (70-80)', 'Very High (>80)'])
    return df

# Feature importance calculation
@st.cache_data
def calculate_feature_importance(df):
    features = ['Infant_deaths', 'Under_five_deaths', 'Adult_mortality',
                'Alcohol_consumption', 'Hepatitis_B', 'Measles', 'BMI',
                'Polio', 'Diphtheria', 'Incidents_HIV', 'GDP_per_capita',
                'Population_mln', 'Thinness_ten_nineteen_years',
                'Thinness_five_nine_years', 'Schooling']

    df_clean = df.dropna(subset=features + ['Life_expectancy'])
    X = df_clean[features]
    y = df_clean['Life_expectancy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    return importance_df, rf.score(X_test, y_test)

# PCA analysis
@st.cache_data
def perform_pca_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Year', 'Decade', 'Economy_status_Developed', 'Economy_status_Developing']]

    df_pca = df[numeric_cols].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_pca)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Life_expectancy'] = df_pca['Life_expectancy'].values

    return pca_df, pca.explained_variance_ratio_

# Clustering analysis
@st.cache_data
def perform_clustering(df):
    features_for_clustering = ['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Adult_mortality']
    df_cluster = df[features_for_clustering].dropna()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cluster)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    df_cluster['Cluster'] = clusters
    return df_cluster

# Main app
def main():
    # Initialize analytics tracking
    tracker = integrate_analytics("Dashboard Home")

    # Header
    st.markdown('<h1 class="main-header">Global Life Expectancy Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    df = load_data()

    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")

    # Data filters
    st.sidebar.subheader("📋 Data Filters")

    # Year range slider
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max())),
        step=1
    )

    # Region filter
    regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique().tolist(),
        default=df['Region'].unique().tolist()
    )

    # Country filter
    countries_in_regions = df[df['Region'].isin(regions)]['Country'].unique().tolist()
    selected_countries = st.sidebar.multiselect(
        "Select Countries (Optional)",
        options=countries_in_regions,
        default=[]
    )

    # Economy status filter
    economy_filter = st.sidebar.radio(
        "Economy Status",
        options=['All', 'Developed', 'Developing'],
        index=0
    )

    # Apply filters
    filtered_df = df[
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1]) &
        (df['Region'].isin(regions))
    ]

    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

    if economy_filter == 'Developed':
        filtered_df = filtered_df[filtered_df['Economy_status_Developed'] == 1]
    elif economy_filter == 'Developing':
        filtered_df = filtered_df[filtered_df['Economy_status_Developing'] == 1]

    # Advanced options
    st.sidebar.subheader("⚙️ Advanced Options")
    show_raw_data = st.sidebar.checkbox("Show Raw Data Table")

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview",
        "🌍 Geographic Analysis",
        "📊 Factor Analysis",
        "🔬 Advanced Analytics",
        "💡 Key Insights",
        "🎯 Interactive Explorer",
        "📚 About"
    ])

    with tab1:
        tracker.track_page_view("Overview Tab")
        # Use the enhanced overview
        render_enhanced_overview(df, filtered_df)

    with tab2:
        tracker.track_page_view("Geographic Analysis Tab")
        # Use the enhanced geographic analysis
        render_enhanced_geographic_analysis(df, filtered_df)

    with tab3:
        tracker.track_page_view("Factor Analysis Tab")
        # Use the enhanced factor analysis
        render_enhanced_factor_analysis(filtered_df, tab3)

    with tab4:
        tracker.track_page_view("Advanced Analytics Tab")
        # Use the enhanced advanced analytics
        render_enhanced_advanced_analytics(df, filtered_df)

    with tab5:
        tracker.track_page_view("Key Insights Tab")
        # Use the enhanced key insights
        render_enhanced_key_insights(df, filtered_df)

    with tab6:
        tracker.track_page_view("Interactive Explorer Tab")
        # Use the enhanced interactive explorer
        render_enhanced_interactive_explorer(df, filtered_df)

    with tab7:
        tracker.track_page_view("About Tab")
        # Use the enhanced about page
        render_enhanced_about(df)

    # Show raw data if requested
    if show_raw_data:
        st.header("📋 Raw Data Table")
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400
        )

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name='life_expectancy_filtered.csv',
            mime='text/csv'
        )

    # Footer
    st.markdown("---")
    st.markdown("### 📊 Dashboard Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**Total Records**: {len(filtered_df):,}")
    with col2:
        st.markdown(f"**Countries**: {filtered_df['Country'].nunique()}")
    with col3:
        st.markdown(f"**Year Range**: {filtered_df['Year'].min()} - {filtered_df['Year'].max()}")

if __name__ == "__main__":
    main()