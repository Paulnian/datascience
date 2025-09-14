import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_trend(df, column='Life_expectancy'):
    """Calculate trend over time"""
    if len(df) < 2:
        return 0, "→"

    yearly_avg = df.groupby('Year')[column].mean().sort_index()
    if len(yearly_avg) < 2:
        return 0, "→"

    # Calculate year-over-year change
    recent_change = yearly_avg.iloc[-1] - yearly_avg.iloc[-2] if len(yearly_avg) > 1 else 0
    overall_change = yearly_avg.iloc[-1] - yearly_avg.iloc[0]

    # Determine trend arrow
    if recent_change > 0.5:
        arrow = "↗"
    elif recent_change < -0.5:
        arrow = "↘"
    else:
        arrow = "→"

    return recent_change, arrow

def get_health_grade(life_exp):
    """Assign a grade based on life expectancy"""
    if life_exp >= 80:
        return "A+", "#00b300"
    elif life_exp >= 75:
        return "A", "#00e600"
    elif life_exp >= 70:
        return "B", "#ffcc00"
    elif life_exp >= 65:
        return "C", "#ff9900"
    elif life_exp >= 60:
        return "D", "#ff6600"
    else:
        return "F", "#ff0000"

def render_enhanced_overview(df, filtered_df):
    """Render enhanced overview tab"""
    st.header("🌍 Global Life Expectancy Overview")

    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters. Please adjust your filter settings.")
        return

    # Calculate key statistics
    avg_life_exp = filtered_df['Life_expectancy'].mean()
    median_life_exp = filtered_df['Life_expectancy'].median()
    recent_change, trend_arrow = calculate_trend(filtered_df)
    grade, grade_color = get_health_grade(avg_life_exp)

    # Get comparison data
    global_avg = df['Life_expectancy'].mean()
    latest_year = filtered_df['Year'].max()
    previous_year = latest_year - 1

    # Top section: Key Performance Indicators
    st.markdown("### 📊 Key Health Indicators")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "Average Life Expectancy",
            f"{avg_life_exp:.1f} years",
            f"{recent_change:.1f} vs last year",
            delta_color="normal" if recent_change > 0 else "inverse"
        )

    with col2:
        # Calculate median for better comparison
        median_life_exp = filtered_df['Life_expectancy'].median()
        iqr = filtered_df['Life_expectancy'].quantile(0.75) - filtered_df['Life_expectancy'].quantile(0.25)

        st.metric(
            "Median",
            f"{median_life_exp:.1f} years",
            f"IQR: {iqr:.1f}",
            help="Median is less affected by outliers. IQR shows the middle 50% spread."
        )

    with col3:
        st.metric(
            "Health Grade",
            grade,
            f"Trend: {trend_arrow}"
        )
        st.markdown(f"<span style='color: {grade_color}'>●●●●●</span>", unsafe_allow_html=True)

    with col4:
        life_exp_range = filtered_df['Life_expectancy'].max() - filtered_df['Life_expectancy'].min()
        st.metric(
            "Inequality Gap",
            f"{life_exp_range:.1f} years",
            "Max - Min difference"
        )

    with col5:
        improvement_rate = (recent_change / avg_life_exp) * 100 if avg_life_exp > 0 else 0
        st.metric(
            "Growth Rate",
            f"{improvement_rate:.2f}%",
            "Year-over-year"
        )

    with col6:
        data_coverage = (len(filtered_df) / len(df)) * 100
        st.metric(
            "Data Coverage",
            f"{data_coverage:.1f}%",
            f"{len(filtered_df):,} records"
        )

    # Insights cards
    st.markdown("### 💡 Quick Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Best performing countries
        top_countries = filtered_df.groupby('Country')['Life_expectancy'].mean().nlargest(3)

        # Build the message based on available data
        top_msg = "**🏆 Top Performers**\n"
        if len(top_countries) > 0:
            top_msg += f"1. {top_countries.index[0]}: {top_countries.iloc[0]:.1f} years\n"
        if len(top_countries) > 1:
            top_msg += f"2. {top_countries.index[1]}: {top_countries.iloc[1]:.1f} years\n"
        else:
            top_msg += "2. N/A\n"
        if len(top_countries) > 2:
            top_msg += f"3. {top_countries.index[2]}: {top_countries.iloc[2]:.1f} years"
        else:
            top_msg += "3. N/A"

        st.info(top_msg)

    with col2:
        # Countries needing attention
        bottom_countries = filtered_df.groupby('Country')['Life_expectancy'].mean().nsmallest(3)

        # Build the message based on available data
        bottom_msg = "**⚠️ Need Attention**\n"
        if len(bottom_countries) > 0:
            bottom_msg += f"1. {bottom_countries.index[0]}: {bottom_countries.iloc[0]:.1f} years\n"
        if len(bottom_countries) > 1:
            bottom_msg += f"2. {bottom_countries.index[1]}: {bottom_countries.iloc[1]:.1f} years\n"
        else:
            bottom_msg += "2. N/A\n"
        if len(bottom_countries) > 2:
            bottom_msg += f"3. {bottom_countries.index[2]}: {bottom_countries.iloc[2]:.1f} years"
        else:
            bottom_msg += "3. N/A"

        st.warning(bottom_msg)

    with col3:
        # Regional insights
        best_region = filtered_df.groupby('Region')['Life_expectancy'].mean().idxmax()
        best_region_avg = filtered_df.groupby('Region')['Life_expectancy'].mean().max()
        worst_region = filtered_df.groupby('Region')['Life_expectancy'].mean().idxmin()
        worst_region_avg = filtered_df.groupby('Region')['Life_expectancy'].mean().min()

        st.success(f"""
        **🌍 Regional Snapshot**
        Best: {best_region} ({best_region_avg:.1f} years)
        Worst: {worst_region} ({worst_region_avg:.1f} years)
        Gap: {best_region_avg - worst_region_avg:.1f} years
        """)

    # Main visualizations
    st.markdown("### 📈 Data Visualizations")

    # Create tabs for different views
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "Distribution", "Trends", "Comparisons", "Deep Dive"
    ])

    with viz_tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Enhanced distribution plot with annotations
            fig_dist = go.Figure()

            # Add histogram
            hist_data = np.histogram(filtered_df['Life_expectancy'], bins=30)
            fig_dist.add_trace(go.Bar(
                x=hist_data[1][:-1],
                y=hist_data[0],
                name='Distribution',
                marker_color='lightblue',
                opacity=0.7
            ))

            # Add statistical markers
            fig_dist.add_vline(x=avg_life_exp, line_dash="dash", line_color="red",
                             annotation_text=f"Mean: {avg_life_exp:.1f}")
            fig_dist.add_vline(x=median_life_exp, line_dash="dot", line_color="green",
                             annotation_text=f"Median: {median_life_exp:.1f}")

            # Highlight zones
            fig_dist.add_vrect(x0=0, x1=60, fillcolor="red", opacity=0.1, annotation_text="Critical")
            fig_dist.add_vrect(x0=60, x1=70, fillcolor="orange", opacity=0.1, annotation_text="Low")
            fig_dist.add_vrect(x0=70, x1=80, fillcolor="yellow", opacity=0.1, annotation_text="Moderate")
            fig_dist.add_vrect(x0=80, x1=100, fillcolor="green", opacity=0.1, annotation_text="High")

            fig_dist.update_layout(
                title="Life Expectancy Distribution with Health Zones",
                xaxis_title="Life Expectancy (years)",
                yaxis_title="Frequency",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            # Statistics summary
            st.markdown("#### 📊 Statistical Summary")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %ile', '75th %ile'],
                'Value': [
                    f"{avg_life_exp:.2f}",
                    f"{median_life_exp:.2f}",
                    f"{filtered_df['Life_expectancy'].std():.2f}",
                    f"{filtered_df['Life_expectancy'].min():.2f}",
                    f"{filtered_df['Life_expectancy'].max():.2f}",
                    f"{filtered_df['Life_expectancy'].quantile(0.25):.2f}",
                    f"{filtered_df['Life_expectancy'].quantile(0.75):.2f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

    with viz_tab2:
        # Enhanced time series with multiple metrics
        time_data = filtered_df.groupby('Year').agg({
            'Life_expectancy': ['mean', 'min', 'max', 'std'],
            'Country': 'nunique'
        }).reset_index()
        time_data.columns = ['Year', 'Mean', 'Min', 'Max', 'StdDev', 'Countries']

        # Create subplot with secondary y-axis
        fig_trends = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Life Expectancy Trends", "Year-over-Year Change"),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )

        # Main trend lines
        fig_trends.add_trace(
            go.Scatter(x=time_data['Year'], y=time_data['Mean'],
                      mode='lines+markers', name='Average',
                      line=dict(color='blue', width=3)),
            row=1, col=1, secondary_y=False
        )

        # Add range
        fig_trends.add_trace(
            go.Scatter(x=time_data['Year'], y=time_data['Max'],
                      mode='lines', name='Maximum',
                      line=dict(color='green', dash='dot')),
            row=1, col=1, secondary_y=False
        )

        fig_trends.add_trace(
            go.Scatter(x=time_data['Year'], y=time_data['Min'],
                      mode='lines', name='Minimum',
                      line=dict(color='red', dash='dot')),
            row=1, col=1, secondary_y=False
        )

        # Add standard deviation as area
        fig_trends.add_trace(
            go.Scatter(x=time_data['Year'], y=time_data['StdDev'],
                      mode='lines', name='Std Dev',
                      line=dict(color='orange'),
                      yaxis='y2'),
            row=1, col=1, secondary_y=True
        )

        # Calculate and plot year-over-year change
        yoy_change = time_data['Mean'].diff()
        fig_trends.add_trace(
            go.Bar(x=time_data['Year'], y=yoy_change,
                  name='YoY Change',
                  marker_color=np.where(yoy_change > 0, 'green', 'red')),
            row=2, col=1
        )

        fig_trends.update_xaxes(title_text="Year", row=2, col=1)
        fig_trends.update_yaxes(title_text="Life Expectancy", row=1, col=1, secondary_y=False)
        fig_trends.update_yaxes(title_text="Std Dev", row=1, col=1, secondary_y=True)
        fig_trends.update_yaxes(title_text="Change (years)", row=2, col=1)

        fig_trends.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig_trends, use_container_width=True)

    with viz_tab3:
        # Comparative analysis
        col1, col2 = st.columns(2)

        with col1:
            # Link to Geographic tab for detailed regional analysis
            st.info("🌍 **For detailed regional analysis**, see the Geographic Analysis tab which provides comprehensive regional comparisons, trends, and inequality measures.")

        with col2:
            # Development status comparison
            dev_comparison = filtered_df.groupby(['Year', 'Economy_status_Developed']).agg({
                'Life_expectancy': 'mean'
            }).reset_index()
            dev_comparison['Status'] = dev_comparison['Economy_status_Developed'].map({
                1: 'Developed',
                0: 'Developing'
            })

            fig_dev = px.line(
                dev_comparison,
                x='Year',
                y='Life_expectancy',
                color='Status',
                title="Developed vs Developing Countries",
                markers=True
            )
            fig_dev.update_layout(height=400)
            st.plotly_chart(fig_dev, use_container_width=True)

    with viz_tab4:
        # Detailed analysis
        st.markdown("#### 🔍 Detailed Country Analysis")

        # Allow country selection
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_countries = st.multiselect(
                "Select countries to compare (max 5):",
                options=sorted(filtered_df['Country'].unique()),
                default=filtered_df.groupby('Country')['Life_expectancy'].mean().nlargest(3).index.tolist()[:3],
                max_selections=5
            )
        with col2:
            analysis_year = st.selectbox(
                "Focus Year:",
                options=sorted(filtered_df['Year'].unique(), reverse=True),
                index=0
            )

        if selected_countries:
            country_data = filtered_df[filtered_df['Country'].isin(selected_countries)]

            # Create color map for countries
            colors = px.colors.qualitative.Set1[:len(selected_countries)]
            color_map = dict(zip(selected_countries, colors))

            # First row: Time trends and current status
            col1, col2 = st.columns(2)

            with col1:
                # Life expectancy trends over time
                fig_trends = go.Figure()
                for country in selected_countries:
                    country_df = country_data[country_data['Country'] == country].sort_values('Year')
                    fig_trends.add_trace(go.Scatter(
                        x=country_df['Year'],
                        y=country_df['Life_expectancy'],
                        mode='lines+markers',
                        name=country,
                        line=dict(color=color_map[country], width=2),
                        marker=dict(size=6)
                    ))

                fig_trends.update_layout(
                    title="Life Expectancy Evolution",
                    xaxis_title="Year",
                    yaxis_title="Life Expectancy (years)",
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_trends, use_container_width=True)

            with col2:
                # Current comparison radar chart
                year_data = country_data[country_data['Year'] == analysis_year]

                if not year_data.empty:
                    # Prepare metrics for radar
                    metrics = ['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Immunization_Avg']
                    year_data['Immunization_Avg'] = year_data[['Hepatitis_B', 'Polio', 'Diphtheria']].mean(axis=1)

                    # Normalize metrics to 0-100 scale for radar
                    normalized_data = {}
                    for metric in metrics:
                        if metric in year_data.columns:
                            min_val = year_data[metric].min()
                            max_val = year_data[metric].max()
                            if max_val > min_val:
                                year_data[f'{metric}_norm'] = ((year_data[metric] - min_val) / (max_val - min_val)) * 100
                            else:
                                year_data[f'{metric}_norm'] = 50

                    fig_radar = go.Figure()

                    categories = ['Life Expectancy', 'GDP/Capita', 'Education', 'Immunization']

                    for country in selected_countries:
                        country_year_data = year_data[year_data['Country'] == country]
                        if not country_year_data.empty:
                            values = []
                            for metric in metrics:
                                norm_col = f'{metric}_norm'
                                if norm_col in country_year_data.columns:
                                    values.append(country_year_data[norm_col].iloc[0])
                                else:
                                    values.append(0)

                            fig_radar.add_trace(go.Scatterpolar(
                                r=values,
                                theta=categories,
                                fill='toself',
                                name=country,
                                line=dict(color=color_map[country])
                            ))

                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        title=f"Multi-Metric Comparison ({analysis_year})",
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info(f"No data available for {analysis_year}")

            # Second row: Relationship analyses
            st.markdown("##### 📊 Key Relationships")

            col1, col2 = st.columns(2)

            with col1:
                # GDP vs Life Expectancy with trend lines per country
                fig_gdp = go.Figure()

                # Add scatter points and trend lines for each country
                for country in selected_countries:
                    country_df = country_data[country_data['Country'] == country].dropna(subset=['GDP_per_capita', 'Life_expectancy'])
                    if len(country_df) > 1:
                        # Add scatter points
                        fig_gdp.add_trace(go.Scatter(
                            x=country_df['GDP_per_capita'],
                            y=country_df['Life_expectancy'],
                            mode='markers',
                            name=country,
                            marker=dict(
                                color=color_map[country],
                                size=8,
                                line=dict(width=1, color='white')
                            ),
                            text=[f"{country}<br>Year: {year}" for year in country_df['Year']],
                            hovertemplate='%{text}<br>GDP: $%{x:,.0f}<br>Life Exp: %{y:.1f} years<extra></extra>'
                        ))

                        # Add trend line if enough data
                        if len(country_df) > 2:
                            z = np.polyfit(country_df['GDP_per_capita'].fillna(0), country_df['Life_expectancy'], 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(country_df['GDP_per_capita'].min(), country_df['GDP_per_capita'].max(), 100)
                            fig_gdp.add_trace(go.Scatter(
                                x=x_trend,
                                y=p(x_trend),
                                mode='lines',
                                name=f'{country} trend',
                                line=dict(color=color_map[country], dash='dash', width=1),
                                showlegend=False
                            ))

                fig_gdp.update_layout(
                    title="GDP vs Life Expectancy Relationship",
                    xaxis_title="GDP per Capita ($)",
                    yaxis_title="Life Expectancy (years)",
                    height=350,
                    xaxis_type="log"  # Log scale for better GDP visualization
                )
                st.plotly_chart(fig_gdp, use_container_width=True)

            with col2:
                # Education vs Life Expectancy
                fig_edu = go.Figure()

                for country in selected_countries:
                    country_df = country_data[country_data['Country'] == country].dropna(subset=['Schooling', 'Life_expectancy'])
                    if not country_df.empty:
                        fig_edu.add_trace(go.Scatter(
                            x=country_df['Schooling'],
                            y=country_df['Life_expectancy'],
                            mode='markers',
                            name=country,
                            marker=dict(
                                color=color_map[country],
                                size=8,
                                line=dict(width=1, color='white')
                            ),
                            text=[f"{country}<br>Year: {year}" for year in country_df['Year']],
                            hovertemplate='%{text}<br>Schooling: %{x:.1f} years<br>Life Exp: %{y:.1f} years<extra></extra>'
                        ))

                        # Add trend line
                        if len(country_df) > 2:
                            z = np.polyfit(country_df['Schooling'], country_df['Life_expectancy'], 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(country_df['Schooling'].min(), country_df['Schooling'].max(), 100)
                            fig_edu.add_trace(go.Scatter(
                                x=x_trend,
                                y=p(x_trend),
                                mode='lines',
                                line=dict(color=color_map[country], dash='dash', width=1),
                                showlegend=False
                            ))

                fig_edu.update_layout(
                    title="Education vs Life Expectancy",
                    xaxis_title="Years of Schooling",
                    yaxis_title="Life Expectancy (years)",
                    height=350
                )
                st.plotly_chart(fig_edu, use_container_width=True)

            # Statistics table
            st.markdown("##### 📈 Statistical Summary")

            # Create summary statistics for selected countries
            summary_stats = []
            for country in selected_countries:
                country_df = country_data[country_data['Country'] == country]
                latest_data = country_df[country_df['Year'] == country_df['Year'].max()].iloc[0] if not country_df.empty else None

                if latest_data is not None:
                    summary_stats.append({
                        'Country': country,
                        'Avg Life Exp': f"{country_df['Life_expectancy'].mean():.1f}",
                        'Latest Life Exp': f"{latest_data['Life_expectancy']:.1f}",
                        'Trend': '↑' if country_df['Life_expectancy'].iloc[-1] > country_df['Life_expectancy'].iloc[0] else '↓',
                        'Avg GDP': f"${country_df['GDP_per_capita'].mean():.0f}",
                        'Avg Education': f"{country_df['Schooling'].mean():.1f} yrs"
                    })

            if summary_stats:
                stats_df = pd.DataFrame(summary_stats)
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
        else:
            st.info("👆 Select countries above to see detailed comparison")

    # Data quality section
    st.markdown("### 📊 Data Quality & Coverage")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        missing_pct = (filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100
        st.metric("Data Completeness", f"{100 - missing_pct:.1f}%")

    with col2:
        year_range = f"{filtered_df['Year'].min()} - {filtered_df['Year'].max()}"
        st.metric("Time Period", year_range)

    with col3:
        unique_countries = filtered_df['Country'].nunique()
        st.metric("Countries", unique_countries)

    with col4:
        unique_regions = filtered_df['Region'].nunique()
        st.metric("Regions", unique_regions)