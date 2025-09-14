import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

def perform_enhanced_pca(df):
    """Enhanced PCA with better feature selection and interpretation"""
    # Select meaningful features for PCA
    health_features = ['Life_expectancy', 'Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Incidents_HIV']
    economic_features = ['GDP_per_capita', 'Population_mln']
    social_features = ['Schooling', 'Alcohol_consumption', 'BMI']
    healthcare_features = ['Hepatitis_B', 'Polio', 'Diphtheria', 'Measles']

    all_features = health_features + economic_features + social_features + healthcare_features

    # Clean data
    df_clean = df[all_features].dropna()

    if len(df_clean) < 50:
        return None, None, None, None

    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # Perform PCA with optimal number of components
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)

    # Find optimal number of components (90% variance)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_var >= 0.90) + 1
    n_components = min(n_components, 5)  # Limit to 5 for interpretability

    # Re-run PCA with optimal components
    pca_optimal = PCA(n_components=n_components)
    pca_result_optimal = pca_optimal.fit_transform(scaled_data)

    # Create result dataframe
    component_cols = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_result_optimal, columns=component_cols)
    pca_df['Life_expectancy'] = df_clean['Life_expectancy'].values

    # Interpret components based on loadings
    loadings = pd.DataFrame(
        pca_optimal.components_.T,
        columns=component_cols,
        index=all_features
    )

    # Component interpretations
    interpretations = {}
    for i in range(n_components):
        pc = f'PC{i+1}'
        top_positive = loadings[pc].nlargest(3)
        top_negative = loadings[pc].nsmallest(3)

        pos_features = list(top_positive.index)
        neg_features = list(top_negative.index)

        # Interpret based on feature types
        interpretation = interpret_pca_component(pos_features, neg_features)
        interpretations[pc] = interpretation

    return pca_df, pca_optimal.explained_variance_ratio_, loadings, interpretations

def interpret_pca_component(positive_features, negative_features):
    """Interpret PCA component based on top loading features"""
    health_terms = ['Life_expectancy', 'Infant_deaths', 'Adult_mortality', 'Incidents_HIV']
    economic_terms = ['GDP_per_capita', 'Population_mln']
    social_terms = ['Schooling', 'Alcohol_consumption', 'BMI']
    healthcare_terms = ['Hepatitis_B', 'Polio', 'Diphtheria']

    pos_health = sum(1 for f in positive_features if any(term in f for term in health_terms))
    pos_economic = sum(1 for f in positive_features if any(term in f for term in economic_terms))
    pos_social = sum(1 for f in positive_features if any(term in f for term in social_terms))
    pos_healthcare = sum(1 for f in positive_features if any(term in f for term in healthcare_terms))

    # Determine dominant theme
    if pos_health > 1:
        if 'Life_expectancy' in positive_features and any('deaths' in f or 'mortality' in f for f in negative_features):
            return "Overall Health Outcomes"
        else:
            return "Health Challenges"
    elif pos_economic > 0:
        return "Economic Development"
    elif pos_healthcare > 1:
        return "Healthcare Infrastructure"
    elif pos_social > 1:
        return "Social Development"
    else:
        return "Mixed Factors"

def perform_optimal_clustering(df):
    """Perform clustering with optimal cluster number selection"""
    # Select features for clustering
    clustering_features = ['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Adult_mortality',
                          'Hepatitis_B', 'Polio', 'Diphtheria', 'Infant_deaths']

    df_clean = df[clustering_features].dropna()

    if len(df_clean) < 20:
        return None, None, None, None

    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    calinski_scores = []
    k_range = range(2, min(10, len(df_clean)//5))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)

        sil_score = silhouette_score(scaled_data, cluster_labels)
        cal_score = calinski_harabasz_score(scaled_data, cluster_labels)

        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)

    # Choose optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]

    # Final clustering
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_clusters = kmeans_final.fit_predict(scaled_data)

    # Add cluster labels to dataframe
    df_clustered = df_clean.copy()
    df_clustered['Cluster'] = final_clusters

    # Interpret clusters
    cluster_interpretations = {}
    for cluster_id in range(optimal_k):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        interpretation = interpret_cluster(cluster_data, df_clustered)
        cluster_interpretations[cluster_id] = interpretation

    return df_clustered, optimal_k, silhouette_scores, cluster_interpretations

def interpret_cluster(cluster_data, all_data):
    """Interpret what each cluster represents"""
    # Calculate cluster characteristics relative to overall mean
    life_exp_mean = cluster_data['Life_expectancy'].mean()
    gdp_mean = cluster_data['GDP_per_capita'].mean()
    education_mean = cluster_data['Schooling'].mean()
    mortality_mean = cluster_data['Adult_mortality'].mean()

    # Overall means for comparison
    overall_life_exp = all_data['Life_expectancy'].mean()
    overall_gdp = all_data['GDP_per_capita'].mean()
    overall_education = all_data['Schooling'].mean()
    overall_mortality = all_data['Adult_mortality'].mean()

    # Classify cluster based on relative performance
    if life_exp_mean > overall_life_exp * 1.05:
        if gdp_mean > overall_gdp * 1.2:
            return "High-Income, High-Longevity"
        else:
            return "High-Longevity, Moderate-Income"
    elif life_exp_mean < overall_life_exp * 0.95:
        if gdp_mean < overall_gdp * 0.5:
            return "Low-Income, Health-Challenged"
        else:
            return "Middle-Income, Health-Challenged"
    else:
        if gdp_mean > overall_gdp:
            return "Developing, Upper-Middle Income"
        else:
            return "Developing, Lower-Middle Income"

def detect_anomalies(df):
    """Detect anomalous countries using multiple methods"""
    features = ['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Adult_mortality']
    df_clean = df[features + ['Country', 'Year']].dropna()

    if len(df_clean) < 20:
        return None

    # Method 1: Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers_iso = iso_forest.fit_predict(df_clean[features])

    # Method 2: Statistical outliers (Z-score > 3)
    z_scores = np.abs(zscore(df_clean[features]))
    outliers_zscore = (z_scores > 3).any(axis=1)

    # Combine methods
    df_clean['Anomaly_IsoForest'] = outliers_iso == -1
    df_clean['Anomaly_ZScore'] = outliers_zscore
    df_clean['Anomaly_Combined'] = df_clean['Anomaly_IsoForest'] | df_clean['Anomaly_ZScore']

    return df_clean

def perform_predictive_modeling(df):
    """Perform and validate predictive modeling"""
    features = ['GDP_per_capita', 'Schooling', 'Adult_mortality', 'Infant_deaths',
               'Hepatitis_B', 'Polio', 'Diphtheria', 'Alcohol_consumption', 'BMI']

    df_clean = df[features + ['Life_expectancy']].dropna()

    if len(df_clean) < 50:
        return None, None, None

    X = df_clean[features]
    y = df_clean['Life_expectancy']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Multiple models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }

    results = {}
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        # Fit and test
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Predictions
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_score': test_score,
            'predictions': y_pred,
            'residuals': residuals,
            'actual': y_test,
            'model': model
        }

    return results, X_test, y_test

def render_enhanced_advanced_analytics(df, filtered_df):
    """Render enhanced advanced analytics tab"""
    st.header("🔬 Advanced Analytics")

    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
        return

    # Create sub-tabs for different advanced analyses
    adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5 = st.tabs([
        "🎨 Dimensionality Analysis",
        "🎯 Intelligent Clustering",
        "🔍 Anomaly Detection",
        "🤖 Predictive Modeling",
        "📊 Statistical Insights"
    ])

    with adv_tab1:
        st.subheader("🎨 Principal Component Analysis")

        # Perform enhanced PCA
        pca_result = perform_enhanced_pca(filtered_df)

        if pca_result[0] is not None:
            pca_df, variance_ratio, loadings, interpretations = pca_result
            n_components = len(variance_ratio)

            # PCA Insights
            col1, col2 = st.columns([2, 1])

            with col1:
                # Scree plot with cumulative variance
                fig_scree = go.Figure()

                cumulative_var = np.cumsum(variance_ratio)

                # Individual variance
                fig_scree.add_trace(go.Bar(
                    x=[f'PC{i+1}' for i in range(n_components)],
                    y=variance_ratio * 100,
                    name='Individual Variance',
                    marker_color='lightblue'
                ))

                # Cumulative variance line
                fig_scree.add_trace(go.Scatter(
                    x=[f'PC{i+1}' for i in range(n_components)],
                    y=cumulative_var * 100,
                    mode='lines+markers',
                    name='Cumulative Variance',
                    line=dict(color='red', width=3),
                    yaxis='y2'
                ))

                fig_scree.update_layout(
                    title="PCA Variance Explanation",
                    xaxis_title="Principal Components",
                    yaxis_title="Individual Variance (%)",
                    yaxis2=dict(
                        title="Cumulative Variance (%)",
                        overlaying='y',
                        side='right',
                        range=[0, 100]
                    ),
                    height=400
                )

                st.plotly_chart(fig_scree, use_container_width=True)

            with col2:
                # Component interpretations
                st.markdown("#### 🎯 Component Meanings")

                for i, (pc, interpretation) in enumerate(interpretations.items()):
                    variance_pct = variance_ratio[i] * 100
                    st.info(f"""
                    **{pc}: {interpretation}**

                    Explains {variance_pct:.1f}% of variance
                    """)

            # Interactive PCA visualization
            st.markdown("#### 🎪 Interactive PCA Visualization")

            if n_components >= 3:
                # 3D and 2D options
                col1, col2 = st.columns([3, 1])

                with col2:
                    viz_type = st.radio(
                        "Visualization:",
                        options=['3D', '2D'],
                        index=0
                    )

                    color_by = st.selectbox(
                        "Color points by:",
                        options=['Life_expectancy', 'None'],
                        index=0
                    )

                    if viz_type == '2D':
                        show_loadings = st.checkbox("Show feature loadings", value=True)

                with col1:
                    if viz_type == '3D':
                        # 3D PCA plot (the original cool visualization)
                        fig_pca_3d = px.scatter_3d(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            z='PC3',
                            color='Life_expectancy' if color_by != 'None' else None,
                            title=f"3D PCA: {interpretations.get('PC1', 'PC1')} × {interpretations.get('PC2', 'PC2')} × {interpretations.get('PC3', 'PC3')}",
                            labels={
                                'PC1': f"PC1: {interpretations.get('PC1', 'PC1')}",
                                'PC2': f"PC2: {interpretations.get('PC2', 'PC2')}",
                                'PC3': f"PC3: {interpretations.get('PC3', 'PC3')}"
                            },
                            color_continuous_scale='Viridis'
                        )
                        fig_pca_3d.update_traces(marker=dict(size=4, line=dict(width=0.5, color='white')))
                        fig_pca_3d.update_layout(height=600)
                        st.plotly_chart(fig_pca_3d, use_container_width=True)

                    else:
                        # 2D PCA plot with loadings
                        fig_pca_2d = px.scatter(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            color='Life_expectancy' if color_by != 'None' else None,
                            title=f"2D PCA: {interpretations.get('PC1', 'PC1')} vs {interpretations.get('PC2', 'PC2')}",
                            labels={
                                'PC1': f"PC1: {interpretations.get('PC1', 'PC1')}",
                                'PC2': f"PC2: {interpretations.get('PC2', 'PC2')}"
                            },
                            color_continuous_scale='Viridis'
                        )

                        # Add loading vectors
                        if show_loadings and len(loadings) > 0:
                            scale_factor = 3  # Scale arrows for visibility

                            for feature in loadings.index[:8]:  # Limit to top 8 for clarity
                                fig_pca_2d.add_annotation(
                                    x=loadings.loc[feature, 'PC1'] * scale_factor,
                                    y=loadings.loc[feature, 'PC2'] * scale_factor,
                                    text=feature.replace('_', ' '),
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor="red",
                                    arrowwidth=1,
                                    ax=0,
                                    ay=0,
                                    font=dict(size=10, color="red")
                                )

                        fig_pca_2d.update_layout(height=500)
                        st.plotly_chart(fig_pca_2d, use_container_width=True)

            elif n_components >= 2:
                # Only 2D available
                fig_pca_2d = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Life_expectancy',
                    title=f"PCA: {interpretations.get('PC1', 'PC1')} vs {interpretations.get('PC2', 'PC2')}",
                    color_continuous_scale='Viridis'
                )
                fig_pca_2d.update_layout(height=500)
                st.plotly_chart(fig_pca_2d, use_container_width=True)
        else:
            st.error("Insufficient data for PCA analysis")

    with adv_tab2:
        st.subheader("🎯 Intelligent Country Clustering")

        # Perform optimal clustering
        clustering_result = perform_optimal_clustering(filtered_df)

        if clustering_result[0] is not None:
            df_clustered, optimal_k, silhouette_scores, cluster_interpretations = clustering_result

            col1, col2 = st.columns([2, 1])

            with col1:
                # Cluster visualization
                fig_cluster = px.scatter(
                    df_clustered,
                    x='GDP_per_capita',
                    y='Life_expectancy',
                    color='Cluster',
                    title=f"Country Clusters (k={optimal_k})",
                    labels={
                        'GDP_per_capita': 'GDP per Capita ($)',
                        'Life_expectancy': 'Life Expectancy (years)'
                    },
                    hover_data=['Schooling', 'Adult_mortality']
                )

                fig_cluster.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
                fig_cluster.update_layout(height=500)
                st.plotly_chart(fig_cluster, use_container_width=True)

            with col2:
                # Cluster interpretations
                st.markdown("#### 🏷️ Cluster Meanings")

                for cluster_id, interpretation in cluster_interpretations.items():
                    count = len(df_clustered[df_clustered['Cluster'] == cluster_id])
                    st.info(f"""
                    **Cluster {cluster_id}: {interpretation}**

                    Countries: {count}
                    """)

            # Cluster quality metrics
            st.markdown("#### 📊 Clustering Quality")

            col1, col2, col3 = st.columns(3)

            with col1:
                best_sil_score = max(silhouette_scores)
                st.metric("Silhouette Score", f"{best_sil_score:.3f}",
                         help="Higher is better (max = 1)")

            with col2:
                st.metric("Optimal Clusters", optimal_k)

            with col3:
                st.metric("Countries Clustered", len(df_clustered))

            # Cluster characteristics table
            st.markdown("#### 📋 Detailed Cluster Characteristics")

            cluster_stats = df_clustered.groupby('Cluster').agg({
                'Life_expectancy': ['mean', 'std'],
                'GDP_per_capita': ['mean', 'std'],
                'Schooling': ['mean', 'std'],
                'Adult_mortality': ['mean', 'std']
            }).round(2)

            # Flatten column names
            cluster_stats.columns = [f"{col[0]}_{col[1]}" for col in cluster_stats.columns]
            cluster_stats['Count'] = df_clustered.groupby('Cluster').size()
            cluster_stats['Interpretation'] = [cluster_interpretations[i] for i in cluster_stats.index]

            st.dataframe(cluster_stats, use_container_width=True)
        else:
            st.error("Insufficient data for clustering analysis")

    with adv_tab3:
        st.subheader("🔍 Anomaly Detection")

        # Detect anomalies
        anomaly_df = detect_anomalies(filtered_df)

        if anomaly_df is not None:
            # Anomaly statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                iso_anomalies = anomaly_df['Anomaly_IsoForest'].sum()
                st.metric("Isolation Forest Anomalies", iso_anomalies)

            with col2:
                zscore_anomalies = anomaly_df['Anomaly_ZScore'].sum()
                st.metric("Statistical Outliers", zscore_anomalies)

            with col3:
                combined_anomalies = anomaly_df['Anomaly_Combined'].sum()
                st.metric("Total Unique Anomalies", combined_anomalies)

            # Anomaly visualization
            col1, col2 = st.columns(2)

            with col1:
                # GDP vs Life Expectancy with anomalies highlighted
                fig_anomaly = px.scatter(
                    anomaly_df,
                    x='GDP_per_capita',
                    y='Life_expectancy',
                    color='Anomaly_Combined',
                    title="Anomalies: GDP vs Life Expectancy",
                    hover_data=['Country', 'Year'],
                    color_discrete_map={True: 'red', False: 'blue'}
                )
                fig_anomaly.update_traces(marker=dict(size=6))
                fig_anomaly.update_layout(height=400)
                st.plotly_chart(fig_anomaly, use_container_width=True)

            with col2:
                # Education vs Life Expectancy
                fig_anomaly2 = px.scatter(
                    anomaly_df,
                    x='Schooling',
                    y='Life_expectancy',
                    color='Anomaly_Combined',
                    title="Anomalies: Education vs Life Expectancy",
                    hover_data=['Country', 'Year'],
                    color_discrete_map={True: 'red', False: 'blue'}
                )
                fig_anomaly2.update_traces(marker=dict(size=6))
                fig_anomaly2.update_layout(height=400)
                st.plotly_chart(fig_anomaly2, use_container_width=True)

            # List of anomalous countries
            if combined_anomalies > 0:
                st.markdown("#### 🚨 Anomalous Countries")

                anomalous_countries = anomaly_df[anomaly_df['Anomaly_Combined']].copy()
                anomalous_countries = anomalous_countries.sort_values('Life_expectancy')

                display_cols = ['Country', 'Year', 'Life_expectancy', 'GDP_per_capita', 'Schooling', 'Adult_mortality']
                st.dataframe(anomalous_countries[display_cols], hide_index=True, use_container_width=True)
        else:
            st.error("Insufficient data for anomaly detection")

    with adv_tab4:
        st.subheader("🤖 Predictive Modeling & Validation")

        # Perform predictive modeling
        modeling_result = perform_predictive_modeling(filtered_df)

        if modeling_result[0] is not None:
            results, X_test, y_test = modeling_result

            # Model comparison
            st.markdown("#### 📊 Model Performance Comparison")

            comparison_data = []
            for model_name, model_results in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'CV Score (Mean)': f"{model_results['cv_mean']:.3f}",
                    'CV Score (Std)': f"{model_results['cv_std']:.3f}",
                    'Test Score': f"{model_results['test_score']:.3f}"
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)

            # Best model analysis
            best_model = max(results.items(), key=lambda x: x[1]['test_score'])
            best_name, best_results = best_model

            st.success(f"🏆 Best Model: **{best_name}** (Test R² = {best_results['test_score']:.3f})")

            # Prediction vs Actual plot
            col1, col2 = st.columns(2)

            with col1:
                fig_pred = go.Figure()

                # Perfect prediction line
                min_val = min(best_results['actual'].min(), best_results['predictions'].min())
                max_val = max(best_results['actual'].max(), best_results['predictions'].max())
                fig_pred.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))

                # Actual vs Predicted
                fig_pred.add_trace(go.Scatter(
                    x=best_results['actual'],
                    y=best_results['predictions'],
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', size=6)
                ))

                fig_pred.update_layout(
                    title=f"{best_name}: Predicted vs Actual",
                    xaxis_title="Actual Life Expectancy",
                    yaxis_title="Predicted Life Expectancy",
                    height=400
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            with col2:
                # Residuals plot
                fig_residuals = px.scatter(
                    x=best_results['predictions'],
                    y=best_results['residuals'],
                    title=f"{best_name}: Residuals Analysis",
                    labels={'x': 'Predicted Values', 'y': 'Residuals'}
                )
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                fig_residuals.update_layout(height=400)
                st.plotly_chart(fig_residuals, use_container_width=True)

            # Feature importance (if Random Forest)
            if best_name == 'Random Forest':
                st.markdown("#### 🎯 Feature Importance")

                rf_model = best_results['model']
                feature_names = list(X_test.columns)
                importances = rf_model.feature_importances_

                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                fig_importance = px.bar(
                    importance_df.head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Most Important Features"
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.error("Insufficient data for predictive modeling")

    with adv_tab5:
        st.subheader("🧪 Advanced Statistical Methods")

        st.info("💡 **Focus**: Advanced statistical techniques unique to this section. For basic correlations, see Factor Analysis tab.")

        # Advanced statistical methods
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔬 Distribution Analysis")

            # Advanced normality testing
            from scipy.stats import shapiro, jarque_bera

            le_data = filtered_df['Life_expectancy'].dropna()
            sample_size = min(5000, len(le_data))
            sample_data = le_data.sample(sample_size, random_state=42)

            # Multiple normality tests
            shapiro_stat, shapiro_p = shapiro(sample_data)
            jb_stat, jb_p = jarque_bera(sample_data)

            test_results = pd.DataFrame({
                'Test': ['Shapiro-Wilk', 'Jarque-Bera'],
                'Statistic': [f"{shapiro_stat:.4f}", f"{jb_stat:.4f}"],
                'P-value': [f"{shapiro_p:.6f}", f"{jb_p:.6f}"],
                'Normal?': [shapiro_p > 0.05, jb_p > 0.05]
            })

            st.dataframe(test_results, hide_index=True)

            # Q-Q plot for advanced normality assessment
            from scipy.stats import probplot
            qq_data = probplot(sample_data, dist="norm")

            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Sample Quantiles',
                marker=dict(color='blue', size=3, opacity=0.6)
            ))

            # Theoretical line
            fig_qq.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1],
                mode='lines',
                name='Perfect Normal',
                line=dict(color='red', dash='dash', width=2)
            ))

            fig_qq.update_layout(
                title="Q-Q Plot: Normality Assessment",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig_qq, use_container_width=True)

        with col2:
            st.markdown("#### 🧬 Advanced Hypothesis Testing")

            # Regional ANOVA with post-hoc analysis
            from scipy.stats import f_oneway, kruskal

            regional_groups = [group['Life_expectancy'].dropna() for name, group in filtered_df.groupby('Region')]
            region_names = list(filtered_df.groupby('Region').groups.keys())

            if len(regional_groups) > 1:
                # Parametric ANOVA
                f_stat, f_p = f_oneway(*regional_groups)

                # Non-parametric Kruskal-Wallis
                h_stat, h_p = kruskal(*regional_groups)

                hypothesis_results = pd.DataFrame({
                    'Test': ['ANOVA (parametric)', 'Kruskal-Wallis (non-parametric)'],
                    'Statistic': [f"{f_stat:.4f}", f"{h_stat:.4f}"],
                    'P-value': [f"{f_p:.6f}", f"{h_p:.6f}"],
                    'Significant?': [f_p < 0.05, h_p < 0.05]
                })

                st.dataframe(hypothesis_results, hide_index=True)

                if f_p < 0.05:
                    st.success("🎯 Significant regional differences detected")

                    # Effect size (eta-squared for ANOVA)
                    ss_between = sum(len(group) * (group.mean() - filtered_df['Life_expectancy'].mean())**2 for group in regional_groups)
                    ss_total = ((filtered_df['Life_expectancy'] - filtered_df['Life_expectancy'].mean())**2).sum()
                    eta_squared = ss_between / ss_total

                    st.info(f"**Effect Size** (η²): {eta_squared:.4f}")

                    if eta_squared < 0.06:
                        effect_interpretation = "Small effect"
                    elif eta_squared < 0.14:
                        effect_interpretation = "Medium effect"
                    else:
                        effect_interpretation = "Large effect"

                    st.write(f"Interpretation: {effect_interpretation}")
                else:
                    st.warning("No significant regional differences")

            # Advanced time series test (if multiple years)
            if len(filtered_df['Year'].unique()) > 2:
                st.markdown("##### 📈 Trend Analysis")

                from scipy.stats import spearmanr

                # Test for monotonic trend over time
                yearly_means = filtered_df.groupby('Year')['Life_expectancy'].mean()
                years = yearly_means.index.values
                values = yearly_means.values

                spearman_r, spearman_p = spearmanr(years, values)

                trend_result = pd.DataFrame({
                    'Test': ['Spearman Correlation (Year vs Life Exp)'],
                    'Coefficient': [f"{spearman_r:.4f}"],
                    'P-value': [f"{spearman_p:.6f}"],
                    'Significant Trend?': [spearman_p < 0.05]
                })

                st.dataframe(trend_result, hide_index=True)

                if spearman_p < 0.05:
                    trend_direction = "increasing" if spearman_r > 0 else "decreasing"
                    st.success(f"📈 Significant {trend_direction} trend detected")
                else:
                    st.info("No significant time trend")

        # Machine Learning Diagnostics
        st.markdown("#### 🤖 ML Model Diagnostics")

        if st.button("Run Advanced Model Diagnostics"):
            # Cross-validation with different metrics
            from sklearn.model_selection import cross_validate
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            # Prepare features
            features = ['GDP_per_capita', 'Schooling', 'Adult_mortality', 'Infant_deaths',
                       'Hepatitis_B', 'Polio', 'Diphtheria']

            df_clean = filtered_df[features + ['Life_expectancy']].dropna()

            if len(df_clean) > 50:
                X = df_clean[features]
                y = df_clean['Life_expectancy']

                # Multiple scoring metrics
                scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']

                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                cv_results = cross_validate(rf, X, y, cv=5, scoring=scoring, return_train_score=True)

                # Display results
                metrics_df = pd.DataFrame({
                    'Metric': ['R²', 'MAE', 'RMSE'],
                    'Test Mean': [
                        f"{cv_results['test_r2'].mean():.4f}",
                        f"{-cv_results['test_neg_mean_absolute_error'].mean():.2f}",
                        f"{np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()):.2f}"
                    ],
                    'Test Std': [
                        f"±{cv_results['test_r2'].std():.4f}",
                        f"±{cv_results['test_neg_mean_absolute_error'].std():.2f}",
                        f"±{np.sqrt(cv_results['test_neg_mean_squared_error']).std():.2f}"
                    ],
                    'Train Mean': [
                        f"{cv_results['train_r2'].mean():.4f}",
                        f"{-cv_results['train_neg_mean_absolute_error'].mean():.2f}",
                        f"{np.sqrt(-cv_results['train_neg_mean_squared_error'].mean()):.2f}"
                    ]
                })

                st.dataframe(metrics_df, hide_index=True)

                # Check for overfitting
                r2_diff = cv_results['train_r2'].mean() - cv_results['test_r2'].mean()
                if r2_diff > 0.1:
                    st.warning(f"⚠️ Possible overfitting detected (Train-Test R² gap: {r2_diff:.3f})")
                else:
                    st.success(f"✅ Model appears well-generalized (Train-Test R² gap: {r2_diff:.3f})")
            else:
                st.error("Insufficient data for advanced diagnostics")