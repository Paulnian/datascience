import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class IntelligentInsightGenerator:
    """Advanced insight generation system that adapts to filtered data"""

    def __init__(self, df, filtered_df):
        self.df = df
        self.filtered_df = filtered_df
        self.insights = []
        self.recommendations = []

    def generate_all_insights(self):
        """Generate comprehensive insights about the filtered data"""

        if self.filtered_df.empty:
            return [], []

        # Generate different types of insights
        self._detect_outliers()
        self._analyze_development_gap()
        self._detect_regional_patterns()
        self._analyze_time_trends()
        self._find_strong_correlations()
        self._detect_health_challenges()
        self._analyze_progress_leaders()
        self._generate_adaptive_recommendations()

        return self.insights, self.recommendations

    def _detect_outliers(self):
        """Find countries that don't fit expected patterns"""
        try:
            # Find countries with surprisingly high/low life expectancy given their GDP
            clean_data = self.filtered_df[['GDP_per_capita', 'Life_expectancy', 'Country']].dropna()

            if len(clean_data) < 10:
                return

            # Simple regression to find expected life expectancy
            from sklearn.linear_model import LinearRegression
            X = clean_data[['GDP_per_capita']]
            y = clean_data['Life_expectancy']

            model = LinearRegression().fit(X, y)
            predicted = model.predict(X)
            residuals = y - predicted

            # Find positive outliers (high performers)
            high_performers = clean_data[residuals > residuals.quantile(0.9)]
            if not high_performers.empty:
                best_performer = high_performers.loc[residuals.idxmax()]
                self.insights.append({
                    'type': 'outlier_positive',
                    'title': f"🌟 High Performer Alert",
                    'content': f"**{best_performer['Country']}** achieves {best_performer['Life_expectancy']:.1f} years life expectancy with GDP of ${best_performer['GDP_per_capita']:,.0f} - significantly above expectation!",
                    'significance': abs(residuals.max()),
                    'evidence': high_performers[['Country', 'Life_expectancy', 'GDP_per_capita']].to_dict('records')
                })

            # Find negative outliers (underperformers)
            low_performers = clean_data[residuals < residuals.quantile(0.1)]
            if not low_performers.empty:
                worst_performer = low_performers.loc[residuals.idxmin()]
                self.insights.append({
                    'type': 'outlier_negative',
                    'title': f"⚠️ Underperformer Alert",
                    'content': f"**{worst_performer['Country']}** has {worst_performer['Life_expectancy']:.1f} years life expectancy despite GDP of ${worst_performer['GDP_per_capita']:,.0f} - needs investigation.",
                    'significance': abs(residuals.min()),
                    'evidence': low_performers[['Country', 'Life_expectancy', 'GDP_per_capita']].to_dict('records')
                })

        except Exception:
            pass  # Skip if insufficient data

    def _analyze_development_gap(self):
        """Analyze gap between developed and developing countries"""
        try:
            developed = self.filtered_df[self.filtered_df['Economy_status_Developed'] == 1]['Life_expectancy']
            developing = self.filtered_df[self.filtered_df['Economy_status_Developing'] == 1]['Life_expectancy']

            if len(developed) > 5 and len(developing) > 5:
                # Statistical test
                t_stat, p_value = ttest_ind(developed, developing)
                gap = developed.mean() - developing.mean()

                if p_value < 0.001:
                    significance_text = "highly significant"
                elif p_value < 0.01:
                    significance_text = "very significant"
                elif p_value < 0.05:
                    significance_text = "significant"
                else:
                    significance_text = "not statistically significant"

                self.insights.append({
                    'type': 'development_gap',
                    'title': f"💰 Development Gap Analysis",
                    'content': f"The {gap:.1f}-year gap between developed ({developed.mean():.1f}) and developing ({developing.mean():.1f}) countries is **{significance_text}** (p={p_value:.4f})",
                    'significance': abs(t_stat),
                    'evidence': {
                        'developed_mean': developed.mean(),
                        'developing_mean': developing.mean(),
                        'p_value': p_value,
                        'gap_years': gap
                    }
                })

        except Exception:
            pass

    def _detect_regional_patterns(self):
        """Find interesting regional patterns"""
        try:
            regional_stats = self.filtered_df.groupby('Region')['Life_expectancy'].agg(['mean', 'std', 'count']).reset_index()
            regional_stats = regional_stats[regional_stats['count'] >= 3]  # At least 3 countries

            if len(regional_stats) < 2:
                return

            # Find region with highest/lowest average
            best_region = regional_stats.loc[regional_stats['mean'].idxmax()]
            worst_region = regional_stats.loc[regional_stats['mean'].idxmin()]

            # Find region with most/least variation
            most_varied = regional_stats.loc[regional_stats['std'].idxmax()]
            least_varied = regional_stats.loc[regional_stats['std'].idxmin()]

            self.insights.append({
                'type': 'regional_leaders',
                'title': f"🌍 Regional Performance",
                'content': f"**{best_region['Region']}** leads with {best_region['mean']:.1f} years average, while **{worst_region['Region']}** trails at {worst_region['mean']:.1f} years ({best_region['mean'] - worst_region['mean']:.1f} year gap)",
                'significance': best_region['mean'] - worst_region['mean'],
                'evidence': regional_stats.to_dict('records')
            })

            if most_varied['std'] > least_varied['std'] * 2:
                self.insights.append({
                    'type': 'regional_inequality',
                    'title': f"⚖️ Regional Inequality",
                    'content': f"**{most_varied['Region']}** shows high internal inequality (σ={most_varied['std']:.1f}) while **{least_varied['Region']}** is more uniform (σ={least_varied['std']:.1f})",
                    'significance': most_varied['std'] / least_varied['std'],
                    'evidence': {'most_varied': most_varied.to_dict(), 'least_varied': least_varied.to_dict()}
                })

        except Exception:
            pass

    def _analyze_time_trends(self):
        """Detect significant time trends"""
        try:
            if len(self.filtered_df['Year'].unique()) < 3:
                return

            # Global trend
            yearly_means = self.filtered_df.groupby('Year')['Life_expectancy'].mean()
            years = yearly_means.index.values
            values = yearly_means.values

            # Statistical trend test
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)

            if p_value < 0.05:
                trend_direction = "increasing" if slope > 0 else "decreasing"
                annual_change = slope

                self.insights.append({
                    'type': 'time_trend',
                    'title': f"📈 Time Trend Detection",
                    'content': f"Life expectancy is **{trend_direction}** at {annual_change:.3f} years per year (R²={r_value**2:.3f}, p={p_value:.4f}) - this trend is statistically significant",
                    'significance': abs(slope / std_err) if std_err > 0 else 0,
                    'evidence': {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'trend_years': len(years)
                    }
                })

            # Find countries with fastest improvement/decline
            country_trends = []
            for country in self.filtered_df['Country'].unique():
                country_data = self.filtered_df[self.filtered_df['Country'] == country]
                if len(country_data) >= 3:
                    try:
                        slope, _, r_val, p_val, _ = stats.linregress(country_data['Year'], country_data['Life_expectancy'])
                        if p_val < 0.1:  # Significant trend
                            country_trends.append({
                                'country': country,
                                'slope': slope,
                                'r_squared': r_val**2,
                                'p_value': p_val
                            })
                    except:
                        continue

            if country_trends:
                # Best improver
                best_improver = max(country_trends, key=lambda x: x['slope'])
                if best_improver['slope'] > 0.5:  # At least 0.5 years/year improvement
                    self.insights.append({
                        'type': 'country_improvement',
                        'title': f"🚀 Fastest Improver",
                        'content': f"**{best_improver['country']}** shows remarkable improvement of +{best_improver['slope']:.2f} years annually (R²={best_improver['r_squared']:.3f})",
                        'significance': best_improver['slope'],
                        'evidence': best_improver
                    })

                # Worst decliner
                worst_decliner = min(country_trends, key=lambda x: x['slope'])
                if worst_decliner['slope'] < -0.2:  # Declining by more than 0.2 years/year
                    self.insights.append({
                        'type': 'country_decline',
                        'title': f"📉 Concerning Decline",
                        'content': f"**{worst_decliner['country']}** shows worrying decline of {worst_decliner['slope']:.2f} years annually - urgent intervention needed",
                        'significance': abs(worst_decliner['slope']),
                        'evidence': worst_decliner
                    })

        except Exception:
            pass

    def _find_strong_correlations(self):
        """Find statistically significant correlations"""
        try:
            numeric_cols = ['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Adult_mortality',
                          'Infant_deaths', 'Hepatitis_B', 'Polio', 'Diphtheria']

            clean_data = self.filtered_df[numeric_cols].dropna()
            if len(clean_data) < 20:
                return

            # Find strongest correlations with life expectancy
            correlations = []
            for col in numeric_cols:
                if col != 'Life_expectancy':
                    r, p = pearsonr(clean_data['Life_expectancy'], clean_data[col])
                    if p < 0.05 and abs(r) > 0.3:  # Significant and meaningful
                        correlations.append({
                            'variable': col,
                            'correlation': r,
                            'p_value': p,
                            'strength': abs(r)
                        })

            if correlations:
                # Strongest positive correlation
                strongest_pos = max([c for c in correlations if c['correlation'] > 0],
                                  key=lambda x: x['strength'], default=None)
                if strongest_pos:
                    self.insights.append({
                        'type': 'strong_correlation_positive',
                        'title': f"🔗 Strongest Health Driver",
                        'content': f"**{strongest_pos['variable']}** shows the strongest positive relationship with life expectancy (r={strongest_pos['correlation']:.3f}, p={strongest_pos['p_value']:.4f})",
                        'significance': strongest_pos['strength'],
                        'evidence': strongest_pos
                    })

                # Strongest negative correlation
                strongest_neg = max([c for c in correlations if c['correlation'] < 0],
                                  key=lambda x: x['strength'], default=None)
                if strongest_neg:
                    self.insights.append({
                        'type': 'strong_correlation_negative',
                        'title': f"⚠️ Biggest Health Risk",
                        'content': f"**{strongest_neg['variable']}** shows the strongest negative relationship with life expectancy (r={strongest_neg['correlation']:.3f}, p={strongest_neg['p_value']:.4f})",
                        'significance': strongest_neg['strength'],
                        'evidence': strongest_neg
                    })

        except Exception:
            pass

    def _detect_health_challenges(self):
        """Identify specific health challenges in the filtered data"""
        try:
            # High infant mortality
            high_infant = self.filtered_df[self.filtered_df['Infant_deaths'] > self.filtered_df['Infant_deaths'].quantile(0.9)]
            if not high_infant.empty:
                avg_life_exp = high_infant['Life_expectancy'].mean()
                global_avg = self.filtered_df['Life_expectancy'].mean()

                self.insights.append({
                    'type': 'health_challenge',
                    'title': f"👶 Infant Mortality Crisis",
                    'content': f"Countries with high infant mortality ({len(high_infant)} countries) average {avg_life_exp:.1f} years life expectancy - {global_avg - avg_life_exp:.1f} years below global average",
                    'significance': global_avg - avg_life_exp,
                    'evidence': high_infant[['Country', 'Infant_deaths', 'Life_expectancy']].to_dict('records')
                })

            # Low immunization coverage
            immun_cols = ['Hepatitis_B', 'Polio', 'Diphtheria']
            self.filtered_df['Immunization_Avg'] = self.filtered_df[immun_cols].mean(axis=1)
            low_immun = self.filtered_df[self.filtered_df['Immunization_Avg'] < 70]  # Below 70% coverage

            if not low_immun.empty:
                avg_life_exp = low_immun['Life_expectancy'].mean()
                high_immun_avg = self.filtered_df[self.filtered_df['Immunization_Avg'] >= 90]['Life_expectancy'].mean()

                if not np.isnan(high_immun_avg):
                    self.insights.append({
                        'type': 'immunization_gap',
                        'title': f"💉 Vaccination Gap",
                        'content': f"{len(low_immun)} countries have <70% immunization coverage, averaging {avg_life_exp:.1f} years life expectancy vs {high_immun_avg:.1f} years for >90% coverage",
                        'significance': high_immun_avg - avg_life_exp,
                        'evidence': {
                            'low_coverage_countries': len(low_immun),
                            'low_coverage_avg_life': avg_life_exp,
                            'high_coverage_avg_life': high_immun_avg
                        }
                    })

        except Exception:
            pass

    def _analyze_progress_leaders(self):
        """Find countries making exceptional progress"""
        try:
            if len(self.filtered_df['Year'].unique()) < 5:
                return

            # Calculate improvement for each country
            improvements = []
            for country in self.filtered_df['Country'].unique():
                country_data = self.filtered_df[self.filtered_df['Country'] == country].sort_values('Year')
                if len(country_data) >= 5:
                    total_improvement = country_data['Life_expectancy'].iloc[-1] - country_data['Life_expectancy'].iloc[0]
                    years_span = country_data['Year'].iloc[-1] - country_data['Year'].iloc[0]
                    if years_span > 0:
                        annual_improvement = total_improvement / years_span
                        improvements.append({
                            'country': country,
                            'total_improvement': total_improvement,
                            'annual_improvement': annual_improvement,
                            'years_span': years_span
                        })

            if improvements:
                # Top improver
                top_improver = max(improvements, key=lambda x: x['annual_improvement'])
                if top_improver['annual_improvement'] > 0.3:
                    self.insights.append({
                        'type': 'progress_leader',
                        'title': f"🏆 Progress Champion",
                        'content': f"**{top_improver['country']}** leads in progress with +{top_improver['total_improvement']:.1f} years improvement over {top_improver['years_span']} years ({top_improver['annual_improvement']:.2f} years/year)",
                        'significance': top_improver['annual_improvement'],
                        'evidence': top_improver
                    })

        except Exception:
            pass

    def _generate_adaptive_recommendations(self):
        """Generate recommendations based on discovered patterns"""

        # Base recommendations on discovered insights
        insight_types = [insight['type'] for insight in self.insights]

        if 'development_gap' in insight_types:
            self.recommendations.append({
                'priority': 'high',
                'category': 'Economic',
                'title': '💰 Bridge Development Gap',
                'content': 'Focus on economic development and poverty reduction as data shows significant life expectancy gaps between developed and developing regions',
                'evidence_based': True
            })

        if 'immunization_gap' in insight_types:
            self.recommendations.append({
                'priority': 'high',
                'category': 'Healthcare',
                'title': '💉 Strengthen Immunization Programs',
                'content': 'Data reveals countries with low immunization coverage have significantly lower life expectancy - prioritize vaccination campaigns',
                'evidence_based': True
            })

        if 'health_challenge' in insight_types:
            self.recommendations.append({
                'priority': 'urgent',
                'category': 'Child Health',
                'title': '👶 Address Infant Mortality',
                'content': 'High infant mortality is strongly associated with reduced life expectancy in your selected data - focus on maternal and child health',
                'evidence_based': True
            })

        if 'country_decline' in insight_types:
            self.recommendations.append({
                'priority': 'urgent',
                'category': 'Crisis Response',
                'title': '🚨 Emergency Intervention',
                'content': 'Some countries show declining life expectancy trends - immediate investigation and targeted interventions needed',
                'evidence_based': True
            })

        if 'progress_leader' in insight_types:
            self.recommendations.append({
                'priority': 'medium',
                'category': 'Learning',
                'title': '🎓 Study Success Stories',
                'content': 'Analyze and replicate successful strategies from countries showing exceptional improvement in life expectancy',
                'evidence_based': True
            })

        # Add some general recommendations if specific ones are limited
        if len(self.recommendations) < 3:
            strong_correlations = [i for i in self.insights if 'correlation' in i['type']]
            if strong_correlations:
                strongest = max(strong_correlations, key=lambda x: x['significance'])
                variable = strongest['evidence']['variable'].replace('_', ' ').title()

                self.recommendations.append({
                    'priority': 'high',
                    'category': 'Evidence-Based',
                    'title': f'🎯 Focus on {variable}',
                    'content': f'Statistical analysis shows {variable} has the strongest relationship with life expectancy in your data',
                    'evidence_based': True
                })

def render_enhanced_key_insights(df, filtered_df):
    """Render enhanced key insights with intelligent analysis"""
    st.header("💡 Intelligent Key Insights")

    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
        return

    # Generate insights
    generator = IntelligentInsightGenerator(df, filtered_df)
    insights, recommendations = generator.generate_all_insights()

    # Create insight tabs
    insight_tab1, insight_tab2, insight_tab3 = st.tabs([
        "🔍 Discovered Patterns",
        "📋 Smart Recommendations",
        "📊 Evidence Dashboard"
    ])

    with insight_tab1:
        st.subheader("🧠 AI-Discovered Insights")

        if not insights:
            st.info("🤖 No significant patterns detected in the current data selection. Try adjusting your filters to include more data.")
        else:
            # Sort insights by significance
            insights_sorted = sorted(insights, key=lambda x: x.get('significance', 0), reverse=True)

            for i, insight in enumerate(insights_sorted[:8]):  # Show top 8 insights
                # Create insight card with visual evidence
                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        # Insight content
                        if insight['type'].endswith('_positive') or 'leader' in insight['type'] or 'improvement' in insight['type']:
                            st.success(f"**{insight['title']}**")
                        elif insight['type'].endswith('_negative') or 'decline' in insight['type'] or 'crisis' in insight['type']:
                            st.error(f"**{insight['title']}**")
                        else:
                            st.info(f"**{insight['title']}**")

                        st.write(insight['content'])

                    with col2:
                        # Significance indicator
                        significance = insight.get('significance', 0)
                        if significance > 10:
                            st.metric("Impact", "Very High", f"{significance:.1f}")
                        elif significance > 5:
                            st.metric("Impact", "High", f"{significance:.1f}")
                        elif significance > 1:
                            st.metric("Impact", "Medium", f"{significance:.1f}")
                        else:
                            st.metric("Impact", "Low", f"{significance:.2f}")

                st.divider()

    with insight_tab2:
        st.subheader("📋 Policy Analysis & Findings")

        # Generate specific recommendations based on actual data patterns
        data_recs = []

        # Analyze current data to create specific recommendations
        try:
            # Economic analysis
            if 'GDP_per_capita' in filtered_df.columns:
                low_gdp_countries = filtered_df[filtered_df['GDP_per_capita'] < filtered_df['GDP_per_capita'].median()]
                if len(low_gdp_countries) > 0:
                    avg_life_low_gdp = low_gdp_countries['Life_expectancy'].mean()
                    avg_life_high_gdp = filtered_df[filtered_df['GDP_per_capita'] >= filtered_df['GDP_per_capita'].median()]['Life_expectancy'].mean()
                    gap = avg_life_high_gdp - avg_life_low_gdp

                    if gap > 5:
                        data_recs.append({
                            "category": "Economic Policy",
                            "finding": f"Countries with GDP per capita below ${filtered_df['GDP_per_capita'].median():.0f} show {gap:.1f} years lower life expectancy",
                            "action": f"Target economic development programs for {len(low_gdp_countries)} countries with GDP < median"
                        })

            # Education analysis
            if 'Schooling' in filtered_df.columns:
                low_education = filtered_df[filtered_df['Schooling'] < 10]
                if len(low_education) > 0:
                    education_impact = filtered_df['Schooling'].corr(filtered_df['Life_expectancy'])
                    data_recs.append({
                        "category": "Education",
                        "finding": f"{len(low_education)} countries have <10 years average schooling (correlation: {education_impact:.3f})",
                        "action": f"Prioritize education expansion in identified countries - each additional year correlates with health gains"
                    })

            # Health infrastructure analysis
            immunization_cols = [col for col in ['Hepatitis_B', 'Polio', 'Diphtheria'] if col in filtered_df.columns]
            if immunization_cols:
                for country_idx, row in filtered_df.iterrows():
                    low_coverage = sum(row[col] < 80 for col in immunization_cols if pd.notna(row[col]))
                    if low_coverage >= 2:  # 2+ vaccines with low coverage
                        coverage_rates = [row[col] for col in immunization_cols if pd.notna(row[col])]
                        avg_coverage = sum(coverage_rates) / len(coverage_rates)

                        data_recs.append({
                            "category": "Health Systems",
                            "finding": f"{row.get('Country', 'Selected countries')} shows {avg_coverage:.1f}% average immunization coverage",
                            "action": f"Scale up vaccination programs - target 90%+ coverage for measurable health improvements"
                        })
                        break  # Just show one example

            # Mortality analysis
            if 'Adult_mortality' in filtered_df.columns:
                high_mortality = filtered_df[filtered_df['Adult_mortality'] > filtered_df['Adult_mortality'].quantile(0.75)]
                if len(high_mortality) > 0:
                    data_recs.append({
                        "category": "Healthcare Delivery",
                        "finding": f"{len(high_mortality)} countries show adult mortality >75th percentile ({filtered_df['Adult_mortality'].quantile(0.75):.0f}/1000)",
                        "action": f"Focus preventive care and chronic disease management in high-mortality regions"
                    })

        except Exception:
            pass

        # Display findings-based recommendations
        if data_recs:
            st.markdown("### Key Findings from Your Data")

            for i, rec in enumerate(data_recs[:4]):  # Show top 4
                with st.container():
                    st.markdown(f"**{rec['category']}**")
                    st.markdown(f"• *Finding*: {rec['finding']}")
                    st.markdown(f"• *Suggested Focus*: {rec['action']}")
                    if i < len(data_recs) - 1:
                        st.divider()
        else:
            st.markdown("### General Health Policy Areas")
            st.markdown("• **Primary Healthcare**: Strengthen basic health services access")
            st.markdown("• **Education Systems**: Expand schooling duration and quality")
            st.markdown("• **Economic Development**: Focus on poverty reduction strategies")
            st.markdown("• **Preventive Care**: Scale immunization and screening programs")

    with insight_tab3:
        st.subheader("📊 Statistical Evidence")

        if insights:
            # Show evidence for top insights
            st.markdown("#### 🔬 Supporting Evidence")

            # Create evidence visualizations
            for insight in insights_sorted[:3]:  # Top 3 insights
                with st.expander(f"📈 Evidence for: {insight['title']}", expanded=False):
                    evidence = insight.get('evidence')

                    if insight['type'] == 'development_gap' and isinstance(evidence, dict):
                        # Development gap chart
                        gap_data = pd.DataFrame({
                            'Status': ['Developed', 'Developing'],
                            'Life_Expectancy': [evidence['developed_mean'], evidence['developing_mean']]
                        })

                        fig = px.bar(gap_data, x='Status', y='Life_Expectancy',
                                   title="Development Gap Visualization",
                                   color='Status', color_discrete_sequence=['green', 'orange'])
                        st.plotly_chart(fig, use_container_width=True)

                        st.metric("Statistical Significance", f"p = {evidence['p_value']:.6f}")

                    elif insight['type'] in ['outlier_positive', 'outlier_negative'] and isinstance(evidence, list):
                        # Outlier evidence table
                        st.dataframe(pd.DataFrame(evidence), hide_index=True)

                    elif 'correlation' in insight['type'] and isinstance(evidence, dict):
                        # Correlation evidence
                        st.metric("Correlation Coefficient", f"{evidence['correlation']:.3f}")
                        st.metric("P-value", f"{evidence['p_value']:.6f}")

                        if evidence['p_value'] < 0.001:
                            st.success("Highly statistically significant")
                        elif evidence['p_value'] < 0.01:
                            st.success("Very statistically significant")
                        elif evidence['p_value'] < 0.05:
                            st.success("Statistically significant")

                    else:
                        st.json(evidence)  # Fallback for other evidence types

        # Data quality summary
        st.markdown("#### 📋 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Insights Found", len(insights))
        with col2:
            st.metric("Recommendations", len(recommendations))
        with col3:
            significance_scores = [i.get('significance', 0) for i in insights]
            avg_significance = np.mean(significance_scores) if significance_scores else 0
            st.metric("Avg Impact Score", f"{avg_significance:.2f}")
        with col4:
            evidence_based = sum(1 for r in recommendations if r.get('evidence_based', False))
            st.metric("Evidence-Based Recs", f"{evidence_based}/{len(recommendations)}")