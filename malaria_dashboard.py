import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="Global Malaria Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv('estimated_numbers.csv')
    
    def clean_numeric_column(col):
        if col.dtype == 'object':
            col = col.str.extract(r'(\d+)', expand=False)
            col = pd.to_numeric(col, errors='coerce')
        return col
    
    df['No. of cases'] = clean_numeric_column(df['No. of cases'])
    df['No. of deaths'] = clean_numeric_column(df['No. of deaths'])
    df['No. of cases_median'] = clean_numeric_column(df['No. of cases_median'])
    df['No. of deaths_median'] = clean_numeric_column(df['No. of deaths_median'])
    df['No. of cases_min'] = clean_numeric_column(df['No. of cases_min'])
    df['No. of cases_max'] = clean_numeric_column(df['No. of cases_max'])
    df['No. of deaths_min'] = clean_numeric_column(df['No. of deaths_min'])
    df['No. of deaths_max'] = clean_numeric_column(df['No. of deaths_max'])
    
    if df['Year'].dtype == 'object':
        df['Year'] = df['Year'].str.strip()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    return df

df = load_data()

st.title("🦟 Global Malaria Dashboard")
st.markdown("### Interactive Analysis of Malaria Cases and Deaths Worldwide")

col1, col2 = st.columns([1, 3])

with col1:
    st.sidebar.header("🎯 Filters")
    
    years = sorted(df['Year'].dropna().unique())
    selected_year = st.sidebar.selectbox(
        "Select Year", 
        years,
        index=len(years)-1 if years else 0
    )
    
    regions = ['All'] + list(df['WHO Region'].dropna().unique())
    selected_region = st.sidebar.selectbox("Select WHO Region", regions)
    
    metric_type = st.sidebar.radio(
        "Select Metric",
        ["Cases", "Deaths", "Both"]
    )
    
    show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=False)

df_filtered = df[df['Year'] == selected_year].copy()
if selected_region != 'All':
    df_filtered = df_filtered[df_filtered['WHO Region'] == selected_region]

col1, col2, col3, col4 = st.columns(4)

total_cases = df_filtered['No. of cases'].sum()
total_deaths = df_filtered['No. of deaths'].sum()
affected_countries = df_filtered[df_filtered['No. of cases'] > 0]['Country'].nunique()
mortality_rate = (total_deaths / total_cases * 100) if total_cases > 0 else 0

with col1:
    st.metric(
        "Total Cases",
        f"{total_cases:,.0f}",
        f"Year {selected_year}"
    )

with col2:
    st.metric(
        "Total Deaths",
        f"{total_deaths:,.0f}",
        f"Mortality Rate: {mortality_rate:.2f}%"
    )

with col3:
    st.metric(
        "Affected Countries",
        f"{affected_countries}",
        f"Out of {df_filtered['Country'].nunique()}"
    )

with col4:
    avg_cases = total_cases / affected_countries if affected_countries > 0 else 0
    st.metric(
        "Avg Cases/Country",
        f"{avg_cases:,.0f}",
        "Among affected countries"
    )

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ World Map", "📊 Top Countries", "📈 Trends", "🥧 Regional Analysis", "📋 Data Table"])

with tab1:
    st.subheader(f"Malaria Distribution - {selected_year}")
    
    map_metric = st.radio(
        "Show on map:",
        ["Cases", "Deaths"],
        horizontal=True,
        key="map_metric"
    )
    
    if map_metric == "Cases":
        color_col = 'No. of cases'
        hover_data = ['No. of cases', 'No. of deaths', 'WHO Region']
    else:
        color_col = 'No. of deaths'
        hover_data = ['No. of deaths', 'No. of cases', 'WHO Region']
    
    df_map = df_filtered[df_filtered[color_col] > 0].copy()
    
    fig_map = px.choropleth(
        df_map,
        locations="Country",
        locationmode="country names",
        color=color_col,
        hover_name="Country",
        hover_data=hover_data,
        color_continuous_scale="Reds",
        labels={color_col: f"Number of {map_metric}"},
        title=f"Global Malaria {map_metric} Distribution ({selected_year})"
    )
    
    fig_map.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=600
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    st.subheader("Top 15 Most Affected Countries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_cases = df_filtered.nlargest(15, 'No. of cases')[['Country', 'No. of cases', 'WHO Region']]
        fig_cases = px.bar(
            top_cases,
            x='No. of cases',
            y='Country',
            orientation='h',
            color='WHO Region',
            title="Top 15 Countries by Cases",
            labels={'No. of cases': 'Number of Cases'},
            hover_data={'No. of cases': ':,.0f'}
        )
        fig_cases.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_cases, use_container_width=True)
    
    with col2:
        top_deaths = df_filtered[df_filtered['No. of deaths'] > 0].nlargest(15, 'No. of deaths')[['Country', 'No. of deaths', 'WHO Region']]
        fig_deaths = px.bar(
            top_deaths,
            x='No. of deaths',
            y='Country',
            orientation='h',
            color='WHO Region',
            title="Top 15 Countries by Deaths",
            labels={'No. of deaths': 'Number of Deaths'},
            hover_data={'No. of deaths': ':,.0f'}
        )
        fig_deaths.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_deaths, use_container_width=True)

with tab3:
    st.subheader("Temporal Trends Analysis")
    
    countries_for_trend = st.multiselect(
        "Select countries to compare:",
        df['Country'].unique(),
        default=df.nlargest(5, 'No. of cases')['Country'].tolist()[:5]
    )
    
    if countries_for_trend:
        df_trend = df[df['Country'].isin(countries_for_trend)].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_trend_cases = px.line(
                df_trend,
                x='Year',
                y='No. of cases',
                color='Country',
                title="Malaria Cases Over Time",
                markers=True
            )
            fig_trend_cases.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of Cases",
                hovermode='x unified'
            )
            st.plotly_chart(fig_trend_cases, use_container_width=True)
        
        with col2:
            fig_trend_deaths = px.line(
                df_trend,
                x='Year',
                y='No. of deaths',
                color='Country',
                title="Malaria Deaths Over Time",
                markers=True
            )
            fig_trend_deaths.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of Deaths",
                hovermode='x unified'
            )
            st.plotly_chart(fig_trend_deaths, use_container_width=True)

with tab4:
    st.subheader("Regional Analysis")
    
    regional_data = df_filtered.groupby('WHO Region').agg({
        'No. of cases': 'sum',
        'No. of deaths': 'sum',
        'Country': 'nunique'
    }).reset_index()
    regional_data.columns = ['WHO Region', 'Total Cases', 'Total Deaths', 'Countries']
    regional_data['Mortality Rate (%)'] = (regional_data['Total Deaths'] / regional_data['Total Cases'] * 100).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie_cases = px.pie(
            regional_data,
            values='Total Cases',
            names='WHO Region',
            title=f"Distribution of Cases by Region ({selected_year})",
            hole=0.4
        )
        fig_pie_cases.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie_cases, use_container_width=True)
    
    with col2:
        fig_pie_deaths = px.pie(
            regional_data,
            values='Total Deaths',
            names='WHO Region',
            title=f"Distribution of Deaths by Region ({selected_year})",
            hole=0.4
        )
        fig_pie_deaths.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie_deaths, use_container_width=True)
    
    st.subheader("Regional Comparison")
    fig_regional = go.Figure()
    
    fig_regional.add_trace(go.Bar(
        name='Cases (scaled by 1000)',
        x=regional_data['WHO Region'],
        y=regional_data['Total Cases']/1000,
        yaxis='y',
        offsetgroup=1
    ))
    
    fig_regional.add_trace(go.Bar(
        name='Deaths',
        x=regional_data['WHO Region'],
        y=regional_data['Total Deaths'],
        yaxis='y2',
        offsetgroup=2
    ))
    
    fig_regional.update_layout(
        title=f"Regional Cases vs Deaths ({selected_year})",
        xaxis=dict(title='WHO Region'),
        yaxis=dict(title='Cases (in thousands)', side='left'),
        yaxis2=dict(title='Deaths', overlaying='y', side='right'),
        hovermode='x',
        barmode='group'
    )
    
    st.plotly_chart(fig_regional, use_container_width=True)

with tab5:
    st.subheader("Detailed Data Table")
    
    search_country = st.text_input("Search for a country:", "")
    
    display_df = df_filtered.copy()
    if search_country:
        display_df = display_df[display_df['Country'].str.contains(search_country, case=False, na=False)]
    
    columns_to_display = ['Country', 'WHO Region', 'No. of cases', 'No. of deaths']
    
    if show_confidence:
        columns_to_display.extend(['No. of cases_min', 'No. of cases_max', 'No. of deaths_min', 'No. of deaths_max'])
    
    display_df = display_df[columns_to_display].sort_values('No. of cases', ascending=False)
    
    st.dataframe(
        display_df.style.format({
            'No. of cases': '{:,.0f}',
            'No. of deaths': '{:,.0f}',
            'No. of cases_min': '{:,.0f}',
            'No. of cases_max': '{:,.0f}',
            'No. of deaths_min': '{:,.0f}',
            'No. of deaths_max': '{:,.0f}'
        }, na_rep='0'),
        use_container_width=True,
        height=500
    )
    
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download filtered data as CSV",
        data=csv,
        file_name=f'malaria_data_{selected_year}_{selected_region}.csv',
        mime='text/csv'
    )

st.markdown("---")
st.markdown("### 📊 Data Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"**Data Year Range:** {df['Year'].min():.0f} - {df['Year'].max():.0f}")

with col2:
    st.info(f"**Total Countries:** {df['Country'].nunique()}")

with col3:
    st.info(f"**WHO Regions:** {df['WHO Region'].nunique()}")

with st.expander("ℹ️ About this Dashboard"):
    st.markdown("""
    This interactive dashboard visualizes global malaria statistics including:
    - **Cases and Deaths**: Track malaria cases and mortality across countries
    - **Regional Analysis**: Compare malaria burden across WHO regions
    - **Temporal Trends**: Analyze how malaria cases change over time
    - **Country Rankings**: Identify most affected countries
    
    **Data Source**: WHO Malaria Estimates
    
    **Note**: Numbers in brackets represent confidence intervals (min-max range)
    """)