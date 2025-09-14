import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def perform_true_factor_analysis(df, n_factors=4):
    """
    Performs true factor analysis to identify latent variables
    """
    # Define feature groups for factor analysis
    health_indicators = ['Infant_deaths', 'Under_five_deaths', 'Adult_mortality',
                        'Incidents_HIV', 'Measles']
    immunization_indicators = ['Hepatitis_B', 'Polio', 'Diphtheria']
    economic_indicators = ['GDP_per_capita', 'Population_mln']
    lifestyle_indicators = ['Alcohol_consumption', 'BMI',
                           'Thinness_ten_nineteen_years', 'Thinness_five_nine_years']
    education_indicators = ['Schooling']

    all_features = health_indicators + immunization_indicators + economic_indicators + \
                  lifestyle_indicators + education_indicators

    # Clean data
    df_clean = df[all_features].dropna()

    if len(df_clean) < 100:
        return None, None, None, None

    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # Perform factor analysis
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    factors = fa.fit_transform(scaled_data)

    # Create factor loadings dataframe
    loadings = pd.DataFrame(
        fa.components_.T,
        columns=[f'Factor {i+1}' for i in range(n_factors)],
        index=all_features
    )

    # Identify dominant features for each factor
    factor_interpretations = {}
    for i in range(n_factors):
        factor_col = f'Factor {i+1}'
        top_features = loadings[factor_col].abs().nlargest(5)

        # Interpret factors based on dominant features
        interpretation = interpret_factor(top_features.index.tolist(),
                                         health_indicators,
                                         immunization_indicators,
                                         economic_indicators,
                                         lifestyle_indicators,
                                         education_indicators)
        factor_interpretations[factor_col] = interpretation

    # Calculate variance explained
    eigenvalues = np.var(factors, axis=0)
    variance_explained = eigenvalues / np.sum(eigenvalues)

    return factors, loadings, factor_interpretations, variance_explained

def interpret_factor(top_features, health, immunization, economic, lifestyle, education):
    """
    Interprets factors based on dominant features with better naming
    """
    health_count = sum(1 for f in top_features if f in health)
    immun_count = sum(1 for f in top_features if f in immunization)
    econ_count = sum(1 for f in top_features if f in economic)
    lifestyle_count = sum(1 for f in top_features if f in lifestyle)
    edu_count = sum(1 for f in top_features if f in education)

    # More descriptive interpretations
    interpretations = {
        'Health & Mortality': (health_count, "Disease burden and mortality rates"),
        'Immunization': (immun_count, "Preventive healthcare coverage"),
        'Economic': (econ_count, "Economic development level"),
        'Lifestyle': (lifestyle_count, "Nutrition and lifestyle factors"),
        'Education': (edu_count, "Educational development")
    }

    # Find dominant category
    max_count = max(health_count, immun_count, econ_count, lifestyle_count, edu_count)

    if max_count < 2:
        # Check for combinations
        if health_count > 0 and immun_count > 0:
            return "Healthcare System Quality"
        elif econ_count > 0 and edu_count > 0:
            return "Socioeconomic Development"
        elif lifestyle_count > 0 and health_count > 0:
            return "Public Health Challenges"
        else:
            return "Mixed Indicators"

    # Return the most dominant category with better description
    for name, (count, description) in interpretations.items():
        if count == max_count:
            return description.title()

    return "Mixed Indicators"

def calculate_enhanced_feature_importance(df):
    """
    Enhanced feature importance with cross-validation and permutation importance
    """
    # Group features by category
    feature_groups = {
        'Mortality': ['Infant_deaths', 'Under_five_deaths', 'Adult_mortality'],
        'Immunization': ['Hepatitis_B', 'Polio', 'Diphtheria'],
        'Economic': ['GDP_per_capita', 'Population_mln'],
        'Lifestyle': ['Alcohol_consumption', 'BMI'],
        'Health_Risks': ['Thinness_ten_nineteen_years', 'Thinness_five_nine_years',
                        'Incidents_HIV', 'Measles'],
        'Education': ['Schooling']
    }

    all_features = [f for group in feature_groups.values() for f in group]

    # Clean data
    df_clean = df[all_features + ['Life_expectancy']].dropna()

    if len(df_clean) < 100:
        return None, None, None, None

    X = df_clean[all_features]
    y = df_clean['Life_expectancy']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest with cross-validation
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Cross-validation scores
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')

    # Fit the model
    rf.fit(X_train, y_train)
    test_score = rf.score(X_test, y_test)

    # Calculate permutation importance
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'RF_Importance': rf.feature_importances_,
        'Permutation_Importance': perm_importance.importances_mean,
        'Perm_Std': perm_importance.importances_std
    })

    # Add feature groups
    importance_df['Group'] = importance_df['Feature'].apply(
        lambda x: next((k for k, v in feature_groups.items() if x in v), 'Other')
    )

    # Calculate group-level importance
    group_importance = importance_df.groupby('Group').agg({
        'RF_Importance': 'sum',
        'Permutation_Importance': 'sum'
    }).reset_index()

    return importance_df, group_importance, cv_scores, test_score

def analyze_interaction_effects(df):
    """
    Analyzes interaction effects between key variables
    """
    # Define key interactions to test
    interactions = [
        ('GDP_per_capita', 'Schooling', 'Economic × Education'),
        ('Immunization_Avg', 'GDP_per_capita', 'Healthcare × Economic'),
        ('Adult_mortality', 'Alcohol_consumption', 'Mortality × Lifestyle'),
        ('BMI', 'GDP_per_capita', 'Nutrition × Economic')
    ]

    # Calculate average immunization
    df['Immunization_Avg'] = df[['Hepatitis_B', 'Polio', 'Diphtheria']].mean(axis=1)

    interaction_results = []

    for var1, var2, label in interactions:
        if var1 in df.columns and var2 in df.columns:
            # Clean data for this interaction
            clean_df = df[[var1, var2, 'Life_expectancy']].dropna()

            if len(clean_df) > 50:
                # Standardize variables
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(clean_df[[var1, var2]])

                # Create interaction term
                interaction_term = X_scaled[:, 0] * X_scaled[:, 1]

                # Build model with and without interaction
                X_no_interaction = X_scaled
                X_with_interaction = np.column_stack([X_scaled, interaction_term])

                y = clean_df['Life_expectancy'].values

                # Fit models
                model_no_int = LinearRegression().fit(X_no_interaction, y)
                model_with_int = LinearRegression().fit(X_with_interaction, y)

                # Calculate R² improvement
                r2_no_int = model_no_int.score(X_no_interaction, y)
                r2_with_int = model_with_int.score(X_with_interaction, y)

                interaction_results.append({
                    'Interaction': label,
                    'Variables': f"{var1} × {var2}",
                    'R² without interaction': r2_no_int,
                    'R² with interaction': r2_with_int,
                    'R² improvement': r2_with_int - r2_no_int,
                    'Interaction coefficient': model_with_int.coef_[2] if len(model_with_int.coef_) > 2 else 0,
                    'Significant': (r2_with_int - r2_no_int) > 0.01
                })

    return pd.DataFrame(interaction_results)

def identify_multicollinearity(df):
    """
    Identifies multicollinearity issues in the features
    """
    features = ['Infant_deaths', 'Under_five_deaths', 'Adult_mortality',
                'Alcohol_consumption', 'Hepatitis_B', 'Measles', 'BMI',
                'Polio', 'Diphtheria', 'Incidents_HIV', 'GDP_per_capita',
                'Population_mln', 'Thinness_ten_nineteen_years',
                'Thinness_five_nine_years', 'Schooling']

    df_clean = df[features].dropna()

    if len(df_clean) < 50:
        return None

    # Calculate correlation matrix
    corr_matrix = df_clean.corr()

    # Find highly correlated pairs (>0.8)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    # Calculate VIF (Variance Inflation Factor) - simplified version
    vif_data = []
    for feature in features:
        others = [f for f in features if f != feature]
        clean_subset = df[others + [feature]].dropna()

        if len(clean_subset) > 50:
            X = clean_subset[others]
            y = clean_subset[feature]

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Calculate R² for this feature predicted by others
            model = LinearRegression().fit(X_scaled, y)
            r2 = model.score(X_scaled, y)

            # VIF = 1 / (1 - R²)
            vif = 1 / (1 - r2) if r2 < 0.999 else 999

            vif_data.append({
                'Feature': feature,
                'VIF': vif,
                'High_Multicollinearity': vif > 10
            })

    return pd.DataFrame(high_corr_pairs), pd.DataFrame(vif_data)

def render_enhanced_factor_analysis(df, tab):
    """
    Renders the enhanced factor analysis tab
    """
    with tab:
        st.header("🔍 Enhanced Factor Analysis")

        if df.empty:
            st.warning("⚠️ No data available for the selected filters.")
            return

        # Create subtabs for different analyses
        subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
            "📊 True Factor Analysis",
            "🎯 Enhanced Feature Importance",
            "🔄 Interaction Effects",
            "📈 Multicollinearity Analysis",
            "💡 Insights & Recommendations"
        ])

        with subtab1:
            st.subheader("🔬 Latent Factor Identification")

            # Factor analysis settings
            col1, col2 = st.columns([3, 1])
            with col2:
                n_factors = st.slider("Number of Factors", 2, 6, 4)

            # Perform factor analysis
            factors, loadings, interpretations, variance_explained = perform_true_factor_analysis(df, n_factors)

            if factors is not None:
                # Display factor interpretations
                st.markdown("### 🎯 Identified Latent Factors")

                # Use a more readable layout for factors with cards
                # First, show a quick summary
                cols = st.columns(n_factors)
                for i, (factor_name, interpretation) in enumerate(interpretations.items()):
                    with cols[i]:
                        # Color code based on variance explained
                        var_pct = variance_explained[i]*100
                        if var_pct > 20:
                            color = "🟢"
                        elif var_pct > 10:
                            color = "🟡"
                        else:
                            color = "🔵"

                        st.markdown(f"""
                        <div style='padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;'>
                            <h4 style='margin: 0;'>{color} {factor_name}</h4>
                            <p style='margin: 5px 0; font-weight: bold;'>{interpretation}</p>
                            <p style='margin: 0; color: #666;'>Explains {var_pct:.1f}% variance</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Add total variance explained
                st.success(f"📊 **Total Variance Explained**: {sum(variance_explained)*100:.1f}%")

                # Add a summary explanation
                with st.expander("📖 **Understanding Factor Analysis**", expanded=False):
                    st.markdown("""
                    **What are Latent Factors?**
                    - Hidden dimensions that explain patterns in your data
                    - Each factor represents a combination of related variables
                    - Similar to finding "themes" in the data

                    **How to interpret:**
                    - **Higher variance %** = Factor explains more variation in the data
                    - **Factor loadings** = How strongly each variable contributes to the factor
                    - **Positive loading** = Variable increases with the factor
                    - **Negative loading** = Variable decreases with the factor

                    **Example:** If Factor 1 is "Healthcare Quality" with high positive loadings for
                    immunization rates and negative loadings for mortality rates, it means countries
                    with better healthcare have higher immunization and lower mortality.
                    """)

                # Factor loadings heatmap
                st.markdown("### 📊 Factor Loadings Matrix")

                # Add color scale explanation
                col1, col2 = st.columns([3, 1])
                with col1:
                    fig_loadings = px.imshow(
                        loadings.T,
                        x=loadings.index,
                        y=loadings.columns,
                        color_continuous_scale='RdBu',
                        aspect="auto",
                        title="How Variables Load onto Each Factor",
                        labels=dict(x="Variables", y="Factors", color="Loading Strength"),
                        text_auto='.2f',
                        color_continuous_midpoint=0
                    )
                    fig_loadings.update_layout(height=400)
                    fig_loadings.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig_loadings, use_container_width=True)

                with col2:
                    st.markdown("#### 🎨 Color Guide")
                    st.markdown("""
                    - **🔵 Deep Blue**: Strong negative loading (-1.0)
                    - **⚪ White**: No relationship (0.0)
                    - **🔴 Deep Red**: Strong positive loading (+1.0)

                    **Reading the Matrix:**
                    - Each row = One factor
                    - Each column = One variable
                    - Color intensity = Strength of relationship
                    """)

                # Scree plot
                col1, col2 = st.columns(2)

                with col1:
                    fig_scree = px.bar(
                        x=[f"Factor {i+1}" for i in range(n_factors)],
                        y=variance_explained,
                        title="Variance Explained by Each Factor",
                        labels={'x': 'Factor', 'y': 'Proportion of Variance'}
                    )
                    fig_scree.add_scatter(
                        x=[f"Factor {i+1}" for i in range(n_factors)],
                        y=np.cumsum(variance_explained),
                        mode='lines+markers',
                        name='Cumulative',
                        yaxis='y2'
                    )
                    fig_scree.update_layout(
                        yaxis2=dict(title='Cumulative Variance', overlaying='y', side='right'),
                        height=400
                    )
                    st.plotly_chart(fig_scree, use_container_width=True)

                with col2:
                    # Top contributing features for each factor
                    st.markdown("### 🔝 Top Features per Factor")

                    # Create an expander for each factor for better organization
                    for i in range(n_factors):
                        factor_col = f'Factor {i+1}'
                        top_5 = loadings[factor_col].abs().nlargest(5)

                        with st.expander(f"**{factor_col}: {interpretations[factor_col]}**", expanded=(i==0)):
                            for feat, val in top_5.items():
                                loading_val = loadings[factor_col][feat]
                                direction = "↑" if loading_val > 0 else "↓"
                                color = "🟢" if loading_val > 0 else "🔴"

                                # Create a progress bar visualization for loadings
                                st.markdown(f"{color} **{feat}** {direction}")
                                st.progress(abs(loading_val), text=f"Loading: {loading_val:.3f}")

                            # Add interpretation help
                            if i == 0:
                                st.caption("🟢↑ = Higher values increase this factor | 🔴↓ = Higher values decrease this factor")
            else:
                st.error("Insufficient data for factor analysis")

        with subtab2:
            st.subheader("🎯 Enhanced Feature Importance Analysis")

            # Calculate enhanced feature importance
            importance_df, group_importance, cv_scores, test_score = calculate_enhanced_feature_importance(df)

            if importance_df is not None:
                # Model performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test R² Score", f"{test_score:.3f}")
                with col2:
                    st.metric("CV Mean Score", f"{np.mean(cv_scores):.3f}")
                with col3:
                    st.metric("CV Std Dev", f"{np.std(cv_scores):.3f}")

                # Feature importance comparison
                st.markdown("### 📊 Feature Importance Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    # Random Forest importance
                    fig_rf = px.bar(
                        importance_df.nlargest(10, 'RF_Importance'),
                        x='RF_Importance',
                        y='Feature',
                        orientation='h',
                        title="Random Forest Feature Importance",
                        color='Group',
                        labels={'RF_Importance': 'Importance Score'}
                    )
                    fig_rf.update_layout(height=400)
                    st.plotly_chart(fig_rf, use_container_width=True)

                with col2:
                    # Permutation importance with error bars
                    top_10_perm = importance_df.nlargest(10, 'Permutation_Importance')
                    fig_perm = go.Figure()
                    fig_perm.add_trace(go.Bar(
                        x=top_10_perm['Permutation_Importance'],
                        y=top_10_perm['Feature'],
                        orientation='h',
                        error_x=dict(
                            type='data',
                            array=top_10_perm['Perm_Std'],
                            visible=True
                        ),
                        marker_color=px.colors.qualitative.Set2
                    ))
                    fig_perm.update_layout(
                        title="Permutation Feature Importance (with std)",
                        xaxis_title="Importance Score",
                        yaxis_title="Feature",
                        height=400
                    )
                    st.plotly_chart(fig_perm, use_container_width=True)

                # Group-level importance
                st.markdown("### 🎨 Feature Group Importance")

                fig_group = px.pie(
                    group_importance,
                    values='RF_Importance',
                    names='Group',
                    title="Relative Importance of Feature Groups"
                )
                st.plotly_chart(fig_group, use_container_width=True)

                # Detailed feature table
                if st.checkbox("Show detailed feature importance table"):
                    st.dataframe(
                        importance_df.sort_values('RF_Importance', ascending=False),
                        use_container_width=True
                    )
            else:
                st.error("Insufficient data for feature importance analysis")

        with subtab3:
            st.subheader("🔄 Variable Interaction Effects")

            # Calculate interaction effects
            interaction_df = analyze_interaction_effects(df)

            if not interaction_df.empty:
                # Display significant interactions
                st.markdown("### 🎯 Significant Interaction Effects")

                significant = interaction_df[interaction_df['Significant']]
                if not significant.empty:
                    for _, row in significant.iterrows():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"**{row['Interaction']}**")
                            st.caption(row['Variables'])
                        with col2:
                            st.metric("R² Improvement", f"{row['R² improvement']:.3f}")
                        with col3:
                            st.metric("Interaction Coef", f"{row['Interaction coefficient']:.3f}")
                else:
                    st.info("No significant interaction effects detected")

                # Visualization of interaction effects
                st.markdown("### 📊 Interaction Effects Visualization")

                fig_int = px.bar(
                    interaction_df,
                    x='Interaction',
                    y='R² improvement',
                    title="Model Improvement from Interaction Terms",
                    color='Significant',
                    color_discrete_map={True: 'green', False: 'gray'}
                )
                fig_int.add_hline(y=0.01, line_dash="dash", line_color="red",
                                 annotation_text="Significance threshold")
                st.plotly_chart(fig_int, use_container_width=True)

                # Detailed table
                if st.checkbox("Show detailed interaction analysis"):
                    st.dataframe(interaction_df, use_container_width=True)
            else:
                st.error("Insufficient data for interaction analysis")

        with subtab4:
            st.subheader("📈 Multicollinearity Analysis")

            high_corr_pairs, vif_data = identify_multicollinearity(df)

            if high_corr_pairs is not None and not high_corr_pairs.empty:
                st.markdown("### ⚠️ Highly Correlated Feature Pairs (|r| > 0.8)")

                for _, row in high_corr_pairs.iterrows():
                    st.warning(
                        f"{row['Feature 1']} ↔ {row['Feature 2']}: "
                        f"r = {row['Correlation']:.3f}"
                    )

                st.markdown("### 📊 Variance Inflation Factors (VIF)")

                # VIF visualization
                fig_vif = px.bar(
                    vif_data.sort_values('VIF', ascending=False),
                    x='VIF',
                    y='Feature',
                    orientation='h',
                    title="VIF by Feature (>10 indicates multicollinearity)",
                    color='High_Multicollinearity',
                    color_discrete_map={True: 'red', False: 'green'}
                )
                fig_vif.add_vline(x=10, line_dash="dash", line_color="red",
                                 annotation_text="Threshold")
                fig_vif.update_layout(height=500)
                st.plotly_chart(fig_vif, use_container_width=True)

                # Recommendations
                problematic = vif_data[vif_data['High_Multicollinearity']]
                if not problematic.empty:
                    st.markdown("### 💡 Recommendations")
                    st.markdown("Consider removing or combining these highly correlated features:")
                    for feat in problematic['Feature']:
                        st.markdown(f"• {feat}")
            else:
                st.info("No significant multicollinearity issues detected")

        with subtab5:
            st.subheader("💡 Key Insights & Recommendations")

            st.markdown("### 🎯 Factor Analysis Insights")

            if factors is not None and interpretations:
                st.markdown("**Identified Latent Dimensions:**")
                for factor, interp in interpretations.items():
                    var_exp = variance_explained[int(factor.split()[-1]) - 1]
                    st.markdown(f"• **{interp}** explains {var_exp*100:.1f}% of variance")

                st.info(
                    "These latent factors suggest that life expectancy is influenced by "
                    "multiple underlying dimensions rather than individual variables alone."
                )

            st.markdown("### 📊 Feature Importance Insights")

            if importance_df is not None:
                top_features = importance_df.nlargest(5, 'RF_Importance')
                st.markdown("**Most Important Predictors:**")
                for _, row in top_features.iterrows():
                    st.markdown(f"• **{row['Feature']}** ({row['Group']}): {row['RF_Importance']:.3f}")

                st.success(
                    f"The Random Forest model achieves an R² of {test_score:.3f}, "
                    f"with cross-validation scores of {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}"
                )

            st.markdown("### 🔄 Interaction Effects")

            if not interaction_df.empty:
                significant_int = interaction_df[interaction_df['Significant']]
                if not significant_int.empty:
                    st.markdown("**Significant Interactions:**")
                    for _, row in significant_int.iterrows():
                        st.markdown(f"• {row['Interaction']} improves prediction by {row['R² improvement']*100:.1f}%")
                else:
                    st.info("Variables appear to have mainly independent effects on life expectancy")

            st.markdown("### ⚠️ Data Quality Considerations")

            if high_corr_pairs is not None and not high_corr_pairs.empty:
                st.warning(
                    f"Found {len(high_corr_pairs)} pairs of highly correlated features. "
                    "Consider using dimensionality reduction or feature selection."
                )

            st.markdown("### 📋 Recommendations for Better Analysis")

            recommendations = [
                "**Use Factor Scores**: Instead of individual variables, use factor scores for cleaner models",
                "**Address Multicollinearity**: Remove redundant features like Under_five_deaths (correlated with Infant_deaths)",
                "**Consider Non-linear Models**: Some relationships may be non-linear (e.g., GDP effect plateaus)",
                "**Regional Stratification**: Build separate models for different regions/development levels",
                "**Time Series Analysis**: Include temporal trends and cohort effects",
                "**Interaction Terms**: Include GDP×Education and Healthcare×Infrastructure interactions"
            ]

            for rec in recommendations:
                st.markdown(f"• {rec}")