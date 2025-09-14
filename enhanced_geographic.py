import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_enhanced_geographic_analysis(df, filtered_df):
    """Render enhanced geographic analysis tab"""
    st.header("🗺️ Geographic Analysis")

    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters. Please adjust your filter settings.")
        return

    # Create sub-tabs for different geographic analyses
    geo_tab1, geo_tab2, geo_tab3 = st.tabs([
        "🌍 Interactive World Map",
        "📊 Regional Analysis",
        "📈 Geographic Trends"
    ])

    with geo_tab1:
        st.subheader("🌍 Interactive World Life Expectancy Map")

        # Controls for the map
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Year slider for map animation
            available_years = sorted(filtered_df['Year'].unique())
            selected_year = st.select_slider(
                "Select Year:",
                options=available_years,
                value=available_years[-1]  # Default to latest year
            )

        with col2:
            # Map projection
            projection = st.selectbox(
                "Map Projection:",
                options=['natural earth', 'mercator', 'orthographic', 'equirectangular'],
                index=0
            )

        with col3:
            # Color scale
            color_scale = st.selectbox(
                "Color Scale:",
                options=['Viridis', 'RdYlGn', 'Plasma', 'Turbo', 'RdBu'],
                index=1
            )

        # Prepare map data
        map_data = filtered_df[filtered_df['Year'] == selected_year].groupby('Country').agg({
            'Life_expectancy': 'mean',
            'GDP_per_capita': 'mean',
            'Schooling': 'mean',
            'Region': 'first',
            'Population_mln': 'mean'
        }).reset_index()

        # Create enhanced choropleth map
        fig_map = px.choropleth(
            map_data,
            locations='Country',
            locationmode='country names',
            color='Life_expectancy',
            hover_name='Country',
            hover_data={
                'Region': True,
                'Life_expectancy': ':.1f',
                'GDP_per_capita': ':,.0f',
                'Schooling': ':.1f',
                'Population_mln': ':.1f'
            },
            color_continuous_scale=color_scale,
            title=f"Life Expectancy by Country ({selected_year})",
            labels={
                'Life_expectancy': 'Life Expectancy (years)',
                'GDP_per_capita': 'GDP per Capita ($)',
                'Schooling': 'Years of Education',
                'Population_mln': 'Population (millions)'
            }
        )

        fig_map.update_layout(
            height=600,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type=projection,
                bgcolor='rgba(0,0,0,0)'
            ),
            title_x=0.5,
            coloraxis_colorbar=dict(
                title="Life Expectancy<br>(years)",
                thickness=15,
                len=0.7
            )
        )

        # Add range slider for better color mapping
        min_val = map_data['Life_expectancy'].min()
        max_val = map_data['Life_expectancy'].max()
        fig_map.update_coloraxes(
            cmin=min_val,
            cmax=max_val,
            colorbar_tickformat='.1f'
        )

        st.plotly_chart(fig_map, use_container_width=True)

        # Map insights
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            highest_country = map_data.loc[map_data['Life_expectancy'].idxmax()]
            st.metric(
                "Highest",
                f"{highest_country['Life_expectancy']:.1f} years",
                highest_country['Country']
            )
        with col2:
            lowest_country = map_data.loc[map_data['Life_expectancy'].idxmin()]
            st.metric(
                "Lowest",
                f"{lowest_country['Life_expectancy']:.1f} years",
                lowest_country['Country']
            )
        with col3:
            life_exp_range = highest_country['Life_expectancy'] - lowest_country['Life_expectancy']
            st.metric("Global Gap", f"{life_exp_range:.1f} years")
        with col4:
            countries_with_data = len(map_data)
            st.metric("Countries Mapped", countries_with_data)

    with geo_tab2:
        st.subheader("📊 Regional Deep Dive")

        # Regional statistics
        regional_stats = filtered_df.groupby(['Region', 'Year']).agg({
            'Life_expectancy': ['mean', 'std', 'min', 'max'],
            'Country': 'nunique',
            'GDP_per_capita': 'mean',
            'Schooling': 'mean'
        }).reset_index()

        # Flatten column names
        regional_stats.columns = ['Region', 'Year', 'Mean_LE', 'Std_LE', 'Min_LE', 'Max_LE',
                                'Countries', 'Mean_GDP', 'Mean_Education']

        # Latest year regional comparison
        latest_regional = regional_stats[regional_stats['Year'] == regional_stats['Year'].max()]

        col1, col2 = st.columns(2)

        with col1:
            # Regional rankings with enhanced info
            fig_regional_rank = px.bar(
                latest_regional.sort_values('Mean_LE', ascending=True),
                x='Mean_LE',
                y='Region',
                orientation='h',
                title="Regional Life Expectancy Rankings (Latest Year)",
                color='Mean_LE',
                color_continuous_scale='RdYlGn',
                hover_data={
                    'Mean_LE': ':.1f',
                    'Std_LE': ':.2f',
                    'Countries': True
                }
            )

            # Add error bars for standard deviation
            fig_regional_rank.update_traces(
                error_x=dict(
                    type='data',
                    array=latest_regional.sort_values('Mean_LE', ascending=True)['Std_LE'],
                    visible=True
                )
            )

            fig_regional_rank.update_layout(
                height=400,
                xaxis_title='Life Expectancy (years)',
                margin=dict(l=150, r=50)
            )
            st.plotly_chart(fig_regional_rank, use_container_width=True)

        with col2:
            # Regional inequality (spread within regions)
            fig_inequality = px.bar(
                latest_regional.sort_values('Std_LE', ascending=False),
                x='Std_LE',
                y='Region',
                orientation='h',
                title="Within-Region Inequality",
                color='Std_LE',
                color_continuous_scale='Reds',
                labels={'Std_LE': 'Standard Deviation (years)'}
            )
            fig_inequality.update_layout(
                height=400,
                xaxis_title='Standard Deviation (years)',
                margin=dict(l=150, r=50)
            )
            st.plotly_chart(fig_inequality, use_container_width=True)

        # Regional trends over time
        st.markdown("#### 📈 Regional Trends Over Time")

        fig_regional_trends = px.line(
            regional_stats,
            x='Year',
            y='Mean_LE',
            color='Region',
            title='Life Expectancy Trends by Region',
            markers=True,
            hover_data=['Countries', 'Mean_GDP', 'Mean_Education']
        )

        # Add confidence intervals
        for region in regional_stats['Region'].unique():
            region_data = regional_stats[regional_stats['Region'] == region]

            # Upper and lower bounds
            upper = region_data['Mean_LE'] + region_data['Std_LE']
            lower = region_data['Mean_LE'] - region_data['Std_LE']

            fig_regional_trends.add_scatter(
                x=region_data['Year'],
                y=upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )

            fig_regional_trends.add_scatter(
                x=region_data['Year'],
                y=lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=px.colors.qualitative.Set1[list(regional_stats['Region'].unique()).index(region) % 10],
                opacity=0.1,
                showlegend=False,
                hoverinfo='skip'
            )

        fig_regional_trends.update_layout(height=500)
        st.plotly_chart(fig_regional_trends, use_container_width=True)

    with geo_tab3:
        st.subheader("📈 Geographic Trends & Patterns")

        # Convergence/Divergence Analysis
        st.markdown("##### 🎯 Global Convergence Analysis")

        # Calculate yearly statistics
        yearly_stats = filtered_df.groupby('Year').agg({
            'Life_expectancy': ['mean', 'std', 'min', 'max']
        }).reset_index()
        yearly_stats.columns = ['Year', 'Global_Mean', 'Global_Std', 'Global_Min', 'Global_Max']

        col1, col2 = st.columns(2)

        with col1:
            # Global convergence plot
            fig_conv = make_subplots(specs=[[{"secondary_y": True}]])

            # Add mean line
            fig_conv.add_trace(
                go.Scatter(
                    x=yearly_stats['Year'],
                    y=yearly_stats['Global_Mean'],
                    mode='lines+markers',
                    name='Global Average',
                    line=dict(color='blue', width=3)
                ),
                secondary_y=False
            )

            # Add standard deviation
            fig_conv.add_trace(
                go.Scatter(
                    x=yearly_stats['Year'],
                    y=yearly_stats['Global_Std'],
                    mode='lines+markers',
                    name='Global Inequality',
                    line=dict(color='red', width=2)
                ),
                secondary_y=True
            )

            fig_conv.update_xaxes(title_text="Year")
            fig_conv.update_yaxes(title_text="Life Expectancy (years)", secondary_y=False)
            fig_conv.update_yaxes(title_text="Standard Deviation", secondary_y=True)
            fig_conv.update_layout(
                title="Global Life Expectancy: Average vs Inequality",
                height=400
            )
            st.plotly_chart(fig_conv, use_container_width=True)

        with col2:
            # Min-Max range over time
            fig_range = go.Figure()

            fig_range.add_trace(go.Scatter(
                x=yearly_stats['Year'],
                y=yearly_stats['Global_Max'],
                mode='lines',
                name='Maximum',
                line=dict(color='green'),
                fill=None
            ))

            fig_range.add_trace(go.Scatter(
                x=yearly_stats['Year'],
                y=yearly_stats['Global_Min'],
                mode='lines',
                name='Minimum',
                line=dict(color='red'),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ))

            fig_range.add_trace(go.Scatter(
                x=yearly_stats['Year'],
                y=yearly_stats['Global_Mean'],
                mode='lines',
                name='Global Average',
                line=dict(color='blue', dash='dash')
            ))

            fig_range.update_layout(
                title="Global Life Expectancy Range",
                xaxis_title="Year",
                yaxis_title="Life Expectancy (years)",
                height=400
            )
            st.plotly_chart(fig_range, use_container_width=True)

        # Regional performance matrix
        st.markdown("##### 🌍 Regional Performance Matrix")

        # Calculate regional growth rates and current status
        regional_growth = []
        for region in filtered_df['Region'].unique():
            region_data = filtered_df[filtered_df['Region'] == region].groupby('Year')['Life_expectancy'].mean()
            if len(region_data) > 1:
                growth_rate = (region_data.iloc[-1] - region_data.iloc[0]) / len(region_data)
                current_level = region_data.iloc[-1]

                regional_growth.append({
                    'Region': region,
                    'Current_Level': current_level,
                    'Growth_Rate': growth_rate,
                    'Countries': filtered_df[filtered_df['Region'] == region]['Country'].nunique()
                })

        if regional_growth:
            growth_df = pd.DataFrame(regional_growth)

            # Performance matrix scatter plot
            fig_matrix = px.scatter(
                growth_df,
                x='Growth_Rate',
                y='Current_Level',
                size='Countries',
                color='Region',
                title='Regional Performance Matrix',
                labels={
                    'Growth_Rate': 'Annual Growth Rate (years)',
                    'Current_Level': 'Current Life Expectancy (years)',
                    'Countries': 'Number of Countries'
                },
                hover_data=['Countries']
            )

            # Add quadrant lines
            mean_growth = growth_df['Growth_Rate'].mean()
            mean_level = growth_df['Current_Level'].mean()

            fig_matrix.add_hline(y=mean_level, line_dash="dash", line_color="gray",
                                annotation_text="Average Level")
            fig_matrix.add_vline(x=mean_growth, line_dash="dash", line_color="gray",
                                annotation_text="Average Growth")

            # Add quadrant labels
            fig_matrix.add_annotation(x=growth_df['Growth_Rate'].max() * 0.8,
                                    y=growth_df['Current_Level'].max() * 0.95,
                                    text="High Performance<br>High Growth", showarrow=False,
                                    bgcolor="lightgreen", opacity=0.7)

            fig_matrix.add_annotation(x=growth_df['Growth_Rate'].min() * 0.8,
                                    y=growth_df['Current_Level'].min() * 1.05,
                                    text="Low Performance<br>Slow Growth", showarrow=False,
                                    bgcolor="lightcoral", opacity=0.7)

            fig_matrix.update_layout(height=500)
            st.plotly_chart(fig_matrix, use_container_width=True)

        # Summary insights
        st.markdown("##### 💡 Geographic Insights")

        if yearly_stats['Global_Std'].iloc[-1] < yearly_stats['Global_Std'].iloc[0]:
            convergence_trend = "Countries are converging (reducing inequality)"
            convergence_color = "success"
        else:
            convergence_trend = "Countries are diverging (increasing inequality)"
            convergence_color = "warning"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.success(f"**Global Trend**: Life expectancy increased by {yearly_stats['Global_Mean'].iloc[-1] - yearly_stats['Global_Mean'].iloc[0]:.1f} years")

        with col2:
            if convergence_color == "success":
                st.success(f"**Inequality**: {convergence_trend}")
            else:
                st.warning(f"**Inequality**: {convergence_trend}")

        with col3:
            global_gap = yearly_stats['Global_Max'].iloc[-1] - yearly_stats['Global_Min'].iloc[-1]
            st.info(f"**Current Gap**: {global_gap:.1f} years between highest and lowest countries")