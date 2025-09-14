import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def render_enhanced_interactive_explorer(df, filtered_df):
    """Render enhanced interactive explorer with multiple analysis tools"""
    st.header("🎯 Interactive Data Explorer")

    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
        return

    # Create tabs for different exploration modes
    explore_tab1, explore_tab2, explore_tab3, explore_tab4 = st.tabs([
        "🎨 Custom Visualizer",
        "🔍 Data Detective",
        "⏱️ Time Explorer",
        "🏆 Country Comparator"
    ])

    with explore_tab1:
        st.subheader("🎨 Build Your Own Visualization")

        # Enhanced controls
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📊 Chart Configuration")

            chart_type = st.selectbox(
                "Chart Type",
                options=['Scatter Plot', 'Line Chart', 'Bar Chart', 'Box Plot', 'Heatmap'],
                index=0
            )

            x_axis = st.selectbox(
                "X-axis Variable",
                options=['GDP_per_capita', 'Schooling', 'Adult_mortality', 'Infant_deaths',
                        'Alcohol_consumption', 'BMI', 'Population_mln', 'Incidents_HIV', 'Year'],
                index=0
            )

            y_axis = st.selectbox(
                "Y-axis Variable",
                options=['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Adult_mortality',
                        'Infant_deaths', 'Under_five_deaths', 'Hepatitis_B', 'Polio', 'Diphtheria'],
                index=0
            )

        with col2:
            st.markdown("#### 🎨 Visual Styling")

            color_by = st.selectbox(
                "Color By",
                options=['None', 'Region', 'Economy_status_Developed', 'Life_Exp_Category', 'Decade'],
                index=1
            )

            size_by = st.selectbox(
                "Size By",
                options=['None', 'Population_mln', 'GDP_per_capita', 'Life_expectancy'],
                index=1
            )

            opacity = st.slider("Opacity", 0.1, 1.0, 0.7, 0.1)

        with col3:
            st.markdown("#### ⚙️ Advanced Options")

            log_scale_x = st.checkbox("Log Scale X-axis")
            log_scale_y = st.checkbox("Log Scale Y-axis")
            show_trendline = st.checkbox("Show Trendline", value=True)
            animate_by_year = st.checkbox("Animate by Year") if chart_type == 'Scatter Plot' else False

        # Validate selections
        if x_axis == y_axis:
            st.error("⚠️ Please select different variables for X and Y axes.")
        else:
            # Create the visualization
            create_custom_visualization(
                filtered_df, chart_type, x_axis, y_axis, color_by, size_by,
                opacity, log_scale_x, log_scale_y, show_trendline, animate_by_year
            )

    with explore_tab2:
        st.subheader("🔍 Data Detective Mode")

        # Interactive data exploration
        st.markdown("#### 🕵️ Investigate Your Data")

        col1, col2 = st.columns(2)

        with col1:
            # Variable to investigate
            focus_var = st.selectbox(
                "Focus Variable to Investigate",
                options=['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Adult_mortality'],
                index=0
            )

            # Outlier detection
            if st.button("🔍 Find Outliers"):
                find_and_display_outliers(filtered_df, focus_var)

        with col2:
            # Correlation explorer
            st.markdown("##### 🔗 Quick Correlation Check")

            var1 = st.selectbox("Variable 1", options=get_numeric_columns(filtered_df), index=0, key="corr1")
            var2 = st.selectbox("Variable 2", options=get_numeric_columns(filtered_df), index=1, key="corr2")

            if var1 != var2:
                correlation_analysis(filtered_df, var1, var2)

        # Interactive data filtering
        st.markdown("#### 🎛️ Interactive Filters")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Life expectancy range
            life_exp_range = st.slider(
                "Life Expectancy Range",
                float(filtered_df['Life_expectancy'].min()),
                float(filtered_df['Life_expectancy'].max()),
                (float(filtered_df['Life_expectancy'].min()), float(filtered_df['Life_expectancy'].max())),
                key="life_exp_filter"
            )

        with col2:
            # GDP range
            gdp_range = st.slider(
                "GDP per Capita Range",
                float(filtered_df['GDP_per_capita'].min()),
                float(filtered_df['GDP_per_capita'].max()),
                (float(filtered_df['GDP_per_capita'].min()), float(filtered_df['GDP_per_capita'].max())),
                key="gdp_filter"
            )

        with col3:
            # Education range
            edu_range = st.slider(
                "Education Years Range",
                float(filtered_df['Schooling'].min()),
                float(filtered_df['Schooling'].max()),
                (float(filtered_df['Schooling'].min()), float(filtered_df['Schooling'].max())),
                key="edu_filter"
            )

        # Apply filters
        detective_df = filtered_df[
            (filtered_df['Life_expectancy'].between(life_exp_range[0], life_exp_range[1])) &
            (filtered_df['GDP_per_capita'].between(gdp_range[0], gdp_range[1])) &
            (filtered_df['Schooling'].between(edu_range[0], edu_range[1]))
        ]

        st.info(f"🔍 **Detective Results**: {len(detective_df)} records match your criteria (from {len(filtered_df)} total)")

        # Show filtered results
        if not detective_df.empty:
            display_detective_insights(detective_df, filtered_df)

    with explore_tab3:
        st.subheader("⏱️ Time Travel Explorer")

        # Time-based analysis
        available_years = sorted(filtered_df['Year'].unique())

        if len(available_years) > 1:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📅 Time Period Selection")

                time_mode = st.radio(
                    "Time Analysis Mode",
                    options=["Single Year", "Year Comparison", "Time Series"],
                    index=2
                )

                if time_mode == "Single Year":
                    selected_year = st.select_slider(
                        "Select Year",
                        options=available_years,
                        value=available_years[-1]
                    )

                elif time_mode == "Year Comparison":
                    year1 = st.selectbox("First Year", options=available_years, index=0)
                    year2 = st.selectbox("Second Year", options=available_years, index=len(available_years)-1)

                else:  # Time Series
                    year_range = st.select_slider(
                        "Year Range",
                        options=available_years,
                        value=(available_years[0], available_years[-1])
                    )

            with col2:
                st.markdown("#### 📊 Time Metric")

                time_metric = st.selectbox(
                    "What to analyze over time",
                    options=['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Adult_mortality'],
                    index=0
                )

                time_grouping = st.selectbox(
                    "Group by",
                    options=['Global', 'Region', 'Country'],
                    index=1
                )

            # Create time-based visualization
            create_time_visualization(filtered_df, time_mode, time_metric, time_grouping, locals())

        else:
            st.info("⏱️ Time analysis requires multiple years of data")

    with explore_tab4:
        st.subheader("🏆 Country Head-to-Head Comparator")

        # Country comparison tool
        st.markdown("#### 🥊 Select Countries to Compare")

        available_countries = sorted(filtered_df['Country'].unique())

        col1, col2 = st.columns(2)

        with col1:
            selected_countries = st.multiselect(
                "Choose Countries (2-6 recommended)",
                options=available_countries,
                default=available_countries[:3] if len(available_countries) >= 3 else available_countries,
                max_selections=6
            )

        with col2:
            comparison_metrics = st.multiselect(
                "Metrics to Compare",
                options=['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Adult_mortality',
                        'Infant_deaths', 'Hepatitis_B', 'Polio', 'Diphtheria'],
                default=['Life_expectancy', 'GDP_per_capita', 'Schooling']
            )

        if len(selected_countries) >= 2 and len(comparison_metrics) >= 1:
            create_country_comparison(filtered_df, selected_countries, comparison_metrics)
        else:
            st.info("👆 Select at least 2 countries and 1 metric to start comparing")

def create_custom_visualization(df, chart_type, x_axis, y_axis, color_by, size_by, opacity, log_x, log_y, trendline, animate):
    """Create custom visualization based on user selections"""

    # Prepare data
    plot_df = df.dropna(subset=[x_axis, y_axis])

    if plot_df.empty:
        st.error("No data available for selected variables")
        return

    # Set up color and size variables
    color_var = None if color_by == 'None' else color_by
    size_var = None if size_by == 'None' else size_by

    # Create the plot based on chart type
    if chart_type == 'Scatter Plot':
        fig = px.scatter(
            plot_df, x=x_axis, y=y_axis, color=color_var, size=size_var,
            title=f"{y_axis} vs {x_axis}",
            hover_data=['Country', 'Year'],
            opacity=opacity,
            animation_frame='Year' if animate else None,
            trendline='ols' if trendline else None,
            log_x=log_x, log_y=log_y
        )

    elif chart_type == 'Line Chart':
        fig = px.line(
            plot_df, x=x_axis, y=y_axis, color=color_var,
            title=f"{y_axis} over {x_axis}",
            hover_data=['Country', 'Year'],
            log_x=log_x, log_y=log_y
        )

    elif chart_type == 'Bar Chart':
        # Aggregate data for bar chart
        agg_df = plot_df.groupby(x_axis)[y_axis].mean().reset_index()
        fig = px.bar(
            agg_df, x=x_axis, y=y_axis,
            title=f"Average {y_axis} by {x_axis}",
            log_x=log_x, log_y=log_y
        )

    elif chart_type == 'Box Plot':
        fig = px.box(
            plot_df, x=color_var, y=y_axis,
            title=f"{y_axis} Distribution by {color_var}",
            log_y=log_y
        )

    elif chart_type == 'Heatmap':
        # Create correlation heatmap
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns
        corr_matrix = plot_df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            title="Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect='auto'
        )

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Show statistics
    if st.checkbox("📊 Show Detailed Statistics", key="custom_stats"):
        show_variable_statistics(plot_df, x_axis, y_axis)

def find_and_display_outliers(df, variable):
    """Find and display outliers for a given variable"""
    data = df[variable].dropna()

    # Calculate outliers using IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[variable] < lower_bound) | (df[variable] > upper_bound)]

    if not outliers.empty:
        st.success(f"🔍 Found {len(outliers)} outliers for {variable}")

        # Show outliers
        outlier_summary = outliers[['Country', 'Year', variable, 'Region']].sort_values(variable)
        st.dataframe(outlier_summary, use_container_width=True)

        # Visualize outliers
        fig = px.box(df, y=variable, title=f"{variable} Distribution with Outliers")
        fig.add_scatter(
            x=[0] * len(outliers),
            y=outliers[variable],
            mode='markers',
            marker=dict(color='red', size=8),
            name='Outliers',
            text=outliers['Country']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No outliers detected for {variable}")

def correlation_analysis(df, var1, var2):
    """Quick correlation analysis between two variables"""
    clean_data = df[[var1, var2]].dropna()

    if len(clean_data) > 1:
        pearson_r, pearson_p = pearsonr(clean_data[var1], clean_data[var2])
        spearman_r, spearman_p = spearmanr(clean_data[var1], clean_data[var2])

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Pearson Correlation", f"{pearson_r:.3f}")
            st.caption(f"p-value: {pearson_p:.4f}")

        with col2:
            st.metric("Spearman Correlation", f"{spearman_r:.3f}")
            st.caption(f"p-value: {spearman_p:.4f}")

def display_detective_insights(filtered_data, original_data):
    """Display insights from filtered data"""
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_life_exp = filtered_data['Life_expectancy'].mean()
        global_avg = original_data['Life_expectancy'].mean()
        st.metric(
            "Average Life Expectancy",
            f"{avg_life_exp:.1f} years",
            f"{avg_life_exp - global_avg:+.1f} vs global"
        )

    with col2:
        unique_countries = filtered_data['Country'].nunique()
        st.metric("Countries", unique_countries)

    with col3:
        year_span = filtered_data['Year'].max() - filtered_data['Year'].min() + 1
        st.metric("Year Span", f"{year_span} years")

def create_time_visualization(df, mode, metric, grouping, context):
    """Create time-based visualizations"""
    if mode == "Single Year":
        selected_year = context['selected_year']
        year_df = df[df['Year'] == selected_year]

        if year_df.empty:
            st.warning(f"No data available for {selected_year}")
            return

        if grouping == "Global":
            global_avg = year_df[metric].mean()
            st.success(f"🌍 **Global {metric} in {selected_year}**: {global_avg:.2f}")

            # Show distribution
            fig = px.histogram(year_df, x=metric,
                             title=f"Global {metric} Distribution in {selected_year}",
                             nbins=30)
            fig.add_vline(x=global_avg, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {global_avg:.1f}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        elif grouping == "Region":
            regional_data = year_df.groupby('Region')[metric].mean().sort_values(ascending=False).reset_index()

            fig = px.bar(regional_data, x='Region', y=metric,
                        title=f"{metric} by Region in {selected_year}",
                        color=metric, color_continuous_scale='RdYlGn')
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        else:  # Country
            # Show top and bottom 10 countries
            country_data = year_df.groupby('Country')[metric].mean().sort_values(ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                top_10 = country_data.head(10).reset_index()
                fig_top = px.bar(top_10, x=metric, y='Country', orientation='h',
                               title=f"Top 10: {metric} in {selected_year}",
                               color=metric, color_continuous_scale='Greens')
                fig_top.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_top, use_container_width=True)

            with col2:
                bottom_10 = country_data.tail(10).reset_index()
                fig_bottom = px.bar(bottom_10, x=metric, y='Country', orientation='h',
                                  title=f"Bottom 10: {metric} in {selected_year}",
                                  color=metric, color_continuous_scale='Reds')
                fig_bottom.update_layout(height=400, yaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_bottom, use_container_width=True)

    elif mode == "Time Series":
        year_range = context['year_range']
        time_df = df[df['Year'].between(year_range[0], year_range[1])]

        if grouping == "Global":
            agg_df = time_df.groupby('Year')[metric].mean().reset_index()
            fig = px.line(agg_df, x='Year', y=metric, title=f"Global {metric} Trend",
                         markers=True)

        elif grouping == "Region":
            agg_df = time_df.groupby(['Year', 'Region'])[metric].mean().reset_index()
            fig = px.line(agg_df, x='Year', y=metric, color='Region',
                         title=f"{metric} Trends by Region", markers=True)

        else:  # Country
            # Show top 10 countries by average
            top_countries = df.groupby('Country')[metric].mean().nlargest(10).index
            country_df = time_df[time_df['Country'].isin(top_countries)]
            fig = px.line(country_df, x='Year', y=metric, color='Country',
                         title=f"Top 10 Countries: {metric} Trends", markers=True)

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif mode == "Year Comparison":
        year1, year2 = context['year1'], context['year2']

        year1_df = df[df['Year'] == year1]
        year2_df = df[df['Year'] == year2]

        if year1_df.empty or year2_df.empty:
            st.warning("No data available for selected years")
            return

        if grouping == "Global":
            data1 = year1_df[metric].mean()
            data2 = year2_df[metric].mean()

            change = data2 - data1
            change_pct = (change / data1) * 100 if data1 != 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{year1}", f"{data1:.2f}")
            with col2:
                st.metric(f"{year2}", f"{data2:.2f}")
            with col3:
                st.metric("Change", f"{change:+.2f}", f"{change_pct:+.1f}%")

            # Add comparison visualization
            comparison_data = pd.DataFrame({
                'Year': [year1, year2],
                'Value': [data1, data2]
            })

            fig = px.bar(comparison_data, x='Year', y='Value',
                        title=f"Global {metric}: {year1} vs {year2}",
                        color='Year')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        elif grouping == "Region":
            # Regional comparison between years
            reg1 = year1_df.groupby('Region')[metric].mean().reset_index()
            reg1['Year'] = year1
            reg2 = year2_df.groupby('Region')[metric].mean().reset_index()
            reg2['Year'] = year2

            combined = pd.concat([reg1, reg2])

            fig = px.bar(combined, x='Region', y=metric, color='Year', barmode='group',
                        title=f"{metric} by Region: {year1} vs {year2}")
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        else:  # Country
            # Show countries with biggest changes
            countries1 = year1_df.groupby('Country')[metric].mean()
            countries2 = year2_df.groupby('Country')[metric].mean()

            # Find common countries
            common_countries = countries1.index.intersection(countries2.index)
            if len(common_countries) > 0:
                changes = countries2[common_countries] - countries1[common_countries]
                changes_df = changes.sort_values(ascending=False).head(10).reset_index()
                changes_df.columns = ['Country', 'Change']

                fig = px.bar(changes_df, x='Country', y='Change',
                           title=f"Biggest Changes in {metric}: {year1} to {year2}",
                           color='Change', color_continuous_scale='RdYlGn')
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No common countries found between selected years")

def create_country_comparison(df, countries, metrics):
    """Create comprehensive country comparison"""

    # Get latest data for each country
    latest_data = df.groupby('Country').last().reset_index()
    comparison_df = latest_data[latest_data['Country'].isin(countries)]

    if comparison_df.empty:
        st.error("No data available for selected countries")
        return

    # Radar chart comparison
    if len(metrics) >= 3:
        create_radar_comparison(comparison_df, countries, metrics)

    # Bar chart comparison
    create_bar_comparison(comparison_df, countries, metrics)

    # Detailed table
    st.markdown("#### 📋 Detailed Comparison Table")
    display_cols = ['Country'] + metrics + ['Year', 'Region']
    comparison_table = comparison_df[display_cols].round(2)
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)

def create_radar_comparison(df, countries, metrics):
    """Create radar chart for country comparison"""
    st.markdown("#### 🎯 Multi-Metric Radar Comparison")

    # Normalize metrics for radar chart
    normalized_df = df.copy()
    for metric in metrics:
        if metric in df.columns:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                normalized_df[f'{metric}_norm'] = ((df[metric] - min_val) / (max_val - min_val)) * 100
            else:
                normalized_df[f'{metric}_norm'] = 50

    fig = go.Figure()

    for country in countries:
        country_data = normalized_df[normalized_df['Country'] == country]
        if not country_data.empty:
            values = []
            for metric in metrics:
                norm_col = f'{metric}_norm'
                if norm_col in country_data.columns:
                    values.append(country_data[norm_col].iloc[0])
                else:
                    values.append(0)

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=country
            ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Normalized Country Comparison",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def create_bar_comparison(df, countries, metrics):
    """Create bar chart comparison"""
    st.markdown("#### 📊 Metric-by-Metric Comparison")

    for metric in metrics:
        if metric in df.columns:
            metric_data = df[df['Country'].isin(countries)][['Country', metric]].sort_values(metric, ascending=False)

            fig = px.bar(
                metric_data,
                x='Country',
                y=metric,
                title=f"{metric} Comparison",
                color='Country'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_variable_statistics(df, x_var, y_var):
    """Show detailed statistics for variables"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {x_var} Statistics")
        x_data = df[x_var].dropna()
        st.write(f"Mean: {x_data.mean():.2f}")
        st.write(f"Median: {x_data.median():.2f}")
        st.write(f"Std Dev: {x_data.std():.2f}")
        st.write(f"Min: {x_data.min():.2f}")
        st.write(f"Max: {x_data.max():.2f}")

    with col2:
        st.markdown(f"#### {y_var} Statistics")
        y_data = df[y_var].dropna()
        st.write(f"Mean: {y_data.mean():.2f}")
        st.write(f"Median: {y_data.median():.2f}")
        st.write(f"Std Dev: {y_data.std():.2f}")
        st.write(f"Min: {y_data.min():.2f}")
        st.write(f"Max: {y_data.max():.2f}")

def get_numeric_columns(df):
    """Get list of numeric columns for correlation analysis"""
    return df.select_dtypes(include=[np.number]).columns.tolist()