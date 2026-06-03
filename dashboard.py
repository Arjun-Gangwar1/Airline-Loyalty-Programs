"""
Airline Loyalty Behavioral Intelligence Dashboard
4-page Streamlit executive dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

st.set_page_config(
    page_title="Airline Loyalty Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    .section-header {
        font-size: 1.4rem; font-weight: 600; color: #1e3c72;
        border-bottom: 2px solid #2a5298; padding-bottom: 6px; margin-bottom: 16px;
    }
    .insight-box {
        background: #f0f4ff; border-left: 4px solid #2a5298;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    final = Path("data/final")
    reports = Path("outputs/reports")

    features = pd.read_csv(final / "customer_segments.csv")
    retention = pd.read_csv(reports / "retention_actions.csv")
    model_comp = pd.read_csv(reports / "model_comparison.csv")
    importance = pd.read_csv(reports / "feature_importance.csv")

    with open(reports / "pipeline_summary.json") as f:
        summary = json.load(f)

    return features, retention, model_comp, importance, summary


try:
    features_df, retention_df, model_comp, importance_df, summary = load_data()
    DATA_OK = True
except Exception as e:
    st.error(f"Data not found. Please run `python complete_pipeline.py` first.\n\nError: {e}")
    DATA_OK = False
    st.stop()

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/airplane-mode-on.png", width=70)
st.sidebar.title("✈️ Loyalty Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    ["Executive Overview", "Churn Intelligence", "Segment Analysis", "Retention Actions"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last updated:** {summary.get('analysis_date', 'N/A')[:10]}")
st.sidebar.markdown(f"**Total customers:** {summary['total_customers']:,}")
st.sidebar.markdown(f"**Best model AUC:** {summary['best_model_auc']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Executive Overview":
    st.title("Executive Overview")
    st.markdown("*Canadian Airline Loyalty Program · 16,737 Members · 2017–2018 Data*")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['at_risk_high']:,}</div>
            <div class="metric-label">🚨 High-Risk Customers</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        rev = summary['total_revenue_at_risk_cad']
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #c0392b, #e74c3c);">
            <div class="metric-value">${rev/1e6:.1f}M</div>
            <div class="metric-label">💰 Revenue at Risk (CAD)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        save = summary['total_potential_save_cad']
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #1a6b3a, #27ae60);">
            <div class="metric-value">${save/1e6:.1f}M</div>
            <div class="metric-label">✅ Potential Savings (CAD)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        auc = summary['xgboost_auc']
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4a235a, #8e44ad);">
            <div class="metric-value">{auc:.3f}</div>
            <div class="metric-label">🤖 Model AUC (XGBoost)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">Churn Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = features_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Customers']
        risk_order = ['High', 'Medium', 'Low']
        risk_counts['Risk Level'] = pd.Categorical(risk_counts['Risk Level'], categories=risk_order, ordered=True)
        risk_counts = risk_counts.sort_values('Risk Level')
        color_map = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}
        fig = px.bar(risk_counts, x='Risk Level', y='Customers', color='Risk Level',
                     color_discrete_map=color_map,
                     text='Customers', title='Customer Count by Risk Level')
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=350, showlegend=False,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Revenue at Risk by Segment</div>', unsafe_allow_html=True)
        rev_seg = features_df.groupby('segment_name')['revenue_at_risk'].sum().reset_index()
        rev_seg.columns = ['Segment', 'Revenue at Risk']
        fig2 = px.pie(rev_seg, values='Revenue at Risk', names='Segment',
                      color_discrete_sequence=px.colors.qualitative.Bold,
                      hole=0.4)
        fig2.update_layout(height=350, showlegend=True,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # Churn definition comparison
    st.markdown('<div class="section-header">Churn Definition Comparison</div>', unsafe_allow_html=True)
    churn_comp = pd.DataFrame({
        'Definition': ['Hard Churn (Formal Cancellation)', 'Activity Churn (No 2018 Flights)', 'Combined (Used)'],
        'Rate': [f"{summary['churn_rate_hard']:.1%}", f"{summary['churn_rate_activity']:.1%}",
                 f"{summary['churn_rate_combined']:.1%}"],
        'Count': [int(summary['total_customers'] * summary['churn_rate_hard']),
                  int(summary['total_customers'] * summary['churn_rate_activity']),
                  int(summary['total_customers'] * summary['churn_rate_combined'])],
        'Captures': ['Formal cancellations only', 'Silent disengagement', 'Full churn picture'],
        'Used': ['No', 'No', '✅ Yes']
    })
    st.dataframe(churn_comp, use_container_width=True, hide_index=True)

    # Key insight boxes
    st.markdown("---")
    st.markdown("### Key Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="insight-box">
        <b>Activity is the #1 predictor:</b> Q4 2017 flight activity alone explains 31% of model importance.
        Customers who didn't fly in Q4 2017 are 4× more likely to churn in 2018.
        </div>""", unsafe_allow_html=True)
    with col2:
        seasonal_churn = summary['segment_profiles'].get('Seasonal Travelers', {}).get('churn_rate', 0)
        st.markdown(f"""<div class="insight-box">
        <b>Seasonal Travelers at highest risk:</b> {seasonal_churn:.0%} churn rate.
        This segment has the lowest active month ratio and highest recency — they disappear for months at a time.
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="insight-box">
        <b>${save/1e6:.1f}M recoverable:</b> With targeted interventions, the model identifies
        {summary['at_risk_high']:,} high-priority customers whose revenue can be saved with the right offer.
        </div>""", unsafe_allow_html=True)

    # Top 10 Priority Customers
    st.markdown("---")
    st.markdown('<div class="section-header">Top 10 Priority Customers (Highest Revenue at Risk)</div>',
                unsafe_allow_html=True)
    top10 = retention_df.nlargest(10, 'revenue_at_risk')[
        ['loyalty_number', 'loyalty_card', 'segment_name', 'risk_level',
         'churn_probability', 'clv', 'revenue_at_risk', 'recommended_action', 'specific_offer']
    ].copy()
    top10['churn_probability'] = top10['churn_probability'].map('{:.1%}'.format)
    top10['clv'] = top10['clv'].map('${:,.0f}'.format)
    top10['revenue_at_risk'] = top10['revenue_at_risk'].map('${:,.0f}'.format)
    st.dataframe(top10, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: CHURN INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Churn Intelligence":
    st.title("Churn Intelligence")
    st.markdown("*Model performance, churn probability distribution, and feature drivers*")
    st.markdown("---")

    # Model performance metrics
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    model_cols = st.columns(3)
    model_names = model_comp['Model'].tolist()
    colors_m = ['#3498db', '#27ae60', '#e74c3c']
    for i, (mc, col) in enumerate(zip(model_names, model_cols)):
        row = model_comp[model_comp['Model'] == mc].iloc[0]
        with col:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, {colors_m[i]}cc, {colors_m[i]});">
                <div class="metric-value">{row['AUC']:.3f}</div>
                <div class="metric-label">{mc}<br>AUC · Recall {row['Recall']:.1%} · F1 {row['F1']:.1%}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Churn Probability Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(features_df, x='churn_probability', nbins=40,
                           color='churned',
                           color_discrete_map={0: '#27ae60', 1: '#e74c3c'},
                           labels={'churned': 'Churned', 'churn_probability': 'Churn Probability'},
                           title='Predicted Churn Probability by Actual Churn Status',
                           barmode='overlay', opacity=0.7)
        fig.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Top Feature Importances</div>', unsafe_allow_html=True)
        top12 = importance_df.head(12).copy()
        top12['feature_clean'] = top12['feature'].str.replace('_', ' ').str.title()
        fig2 = px.bar(top12.sort_values('importance'), x='importance', y='feature_clean',
                      orientation='h', color='importance',
                      color_continuous_scale='Reds',
                      labels={'importance': 'Importance Score', 'feature_clean': ''},
                      title='XGBoost Feature Importance (Top 12)')
        fig2.update_layout(height=380, coloraxis_showscale=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # Churn by demographics
    st.markdown('<div class="section-header">Churn by Customer Attributes</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        tier_churn = features_df.groupby('loyalty_card')['churned'].mean().reset_index()
        tier_churn.columns = ['Loyalty Tier', 'Churn Rate']
        fig = px.bar(tier_churn, x='Loyalty Tier', y='Churn Rate', title='Churn Rate by Loyalty Tier',
                     color='Churn Rate', color_continuous_scale='RdYlGn_r', text='Churn Rate')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(height=320, coloraxis_showscale=False,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        edu_churn = features_df.groupby('education')['churned'].mean().reset_index()
        edu_churn.columns = ['Education', 'Churn Rate']
        edu_order = ['High School or Below', 'College', 'Bachelor', 'Master', 'Doctor']
        edu_churn['Education'] = pd.Categorical(edu_churn['Education'], categories=edu_order, ordered=True)
        edu_churn = edu_churn.sort_values('Education')
        fig2 = px.bar(edu_churn, x='Education', y='Churn Rate', title='Churn Rate by Education',
                      color='Churn Rate', color_continuous_scale='RdYlGn_r', text='Churn Rate')
        fig2.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig2.update_layout(height=320, coloraxis_showscale=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        gender_churn = features_df.groupby('gender')['churned'].mean().reset_index()
        gender_churn.columns = ['Gender', 'Churn Rate']
        fig3 = px.bar(gender_churn, x='Gender', y='Churn Rate', title='Churn Rate by Gender',
                      color='Gender', color_discrete_sequence=['#3498db', '#e91e63'],
                      text='Churn Rate')
        fig3.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig3.update_layout(height=320, showlegend=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    # Churn vs CLV scatter
    st.markdown('<div class="section-header">Churn Risk vs Customer Value</div>', unsafe_allow_html=True)
    sample = features_df.sample(min(3000, len(features_df)), random_state=42)
    fig = px.scatter(sample, x='clv', y='churn_probability',
                     color='segment_name', size='revenue_at_risk',
                     hover_data=['loyalty_card', 'loyalty_number'],
                     labels={'clv': 'Customer Lifetime Value (CAD)',
                             'churn_probability': 'Churn Probability',
                             'segment_name': 'Segment'},
                     title='Churn Probability vs CLV (sized by Revenue at Risk)',
                     opacity=0.6)
    fig.add_hline(y=0.70, line_dash='dash', line_color='red', annotation_text='High Risk Threshold (70%)')
    fig.add_hline(y=0.40, line_dash='dash', line_color='orange', annotation_text='Medium Risk Threshold (40%)')
    fig.update_layout(height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # Model metrics table
    st.markdown('<div class="section-header">Detailed Model Metrics</div>', unsafe_allow_html=True)
    display_comp = model_comp.copy()
    for col in ['AUC', 'Recall', 'Precision', 'F1']:
        display_comp[col] = display_comp[col].map('{:.4f}'.format)
    st.dataframe(display_comp, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="insight-box">
    <b>Interpretation:</b> XGBoost (AUC 0.870) correctly identifies 68.5% of churning customers before they leave,
    with 53.8% precision. Random Forest achieves slightly higher AUC (0.873) with higher precision (76.5%) but lower recall.
    For churn prevention, recall matters more — catching 10 more churners justifies a few extra false alerts.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: SEGMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Segment Analysis":
    st.title("Customer Segment Analysis")
    st.markdown("*Behavioral profiles, risk heatmaps, and segment-level insights*")
    st.markdown("---")

    seg_summary = features_df.groupby('segment_name').agg(
        Customers=('loyalty_number', 'count'),
        Churn_Rate=('churned', 'mean'),
        Avg_CLV=('clv', 'mean'),
        Avg_Flights_Monthly=('avg_flights_per_month', 'mean'),
        Avg_Recency_Months=('recency_months', 'mean'),
        Avg_Engagement=('engagement_health_score', 'mean'),
        Revenue_At_Risk=('revenue_at_risk', 'sum'),
        Avg_Tenure=('tenure_months', 'mean'),
    ).reset_index()

    # Summary cards per segment
    st.markdown('<div class="section-header">Segment Overview</div>', unsafe_allow_html=True)
    cols = st.columns(len(seg_summary))
    seg_colors = {'Active Champions': '#27ae60', 'Miles Hoarders': '#3498db',
                  'Seasonal Travelers': '#e74c3c', 'Premium Loyalists': '#9b59b6',
                  'Silent Drifters': '#c0392b', 'Rising Stars': '#f39c12',
                  'At-Risk Starters': '#e67e22'}

    for col, (_, row) in zip(cols, seg_summary.iterrows()):
        seg = row['segment_name']
        color = seg_colors.get(seg, '#1e3c72')
        with col:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, {color}cc, {color});">
                <div class="metric-value" style="font-size:1.4rem">{seg}</div>
                <div class="metric-label">
                    {row['Customers']:,} customers<br>
                    Churn: {row['Churn_Rate']:.0%}<br>
                    Avg CLV: ${row['Avg_CLV']:,.0f}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">Churn Rate vs Avg CLV by Segment</div>', unsafe_allow_html=True)
        fig = px.scatter(seg_summary, x='Avg_CLV', y='Churn_Rate',
                         size='Customers', color='segment_name', text='segment_name',
                         labels={'Avg_CLV': 'Avg CLV (CAD)', 'Churn_Rate': 'Churn Rate'},
                         title='Risk vs Value Matrix',
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(textposition='top center')
        fig.update_layout(height=400, showlegend=False,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Revenue at Risk by Segment</div>', unsafe_allow_html=True)
        fig2 = px.bar(seg_summary.sort_values('Revenue_At_Risk', ascending=True),
                      x='Revenue_At_Risk', y='segment_name', orientation='h',
                      color='segment_name', color_discrete_map=seg_colors,
                      labels={'Revenue_At_Risk': 'Total Revenue at Risk (CAD)', 'segment_name': ''},
                      title='Segment Revenue at Risk')
        fig2.update_traces(showlegend=False)
        fig2.update_xaxes(tickprefix='$', tickformat=',')
        fig2.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # Behavioral heatmap
    st.markdown('<div class="section-header">Behavioral Profile Heatmap</div>', unsafe_allow_html=True)
    heat_features = ['avg_flights_per_month', 'recency_months', 'active_month_ratio',
                     'redemption_rate', 'clv_log', 'engagement_health_score',
                     'tenure_months', 'tier_numeric']
    heat_display = ['Avg Flights/Month', 'Recency (Months)', 'Active Month Ratio',
                    'Redemption Rate', 'CLV (log)', 'Engagement Score',
                    'Tenure (Months)', 'Tier Level']
    heatmap_data = features_df.groupby('segment_name')[heat_features].mean()
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-6)
    heatmap_norm.columns = heat_display

    fig_heat = px.imshow(
        heatmap_norm.T,
        color_continuous_scale='RdYlGn',
        zmin=0, zmax=1,
        aspect='auto',
        title='Normalized Behavioral Features per Segment (0=Low, 1=High)',
        text_auto='.2f'
    )
    fig_heat.update_layout(height=420, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_heat, use_container_width=True)

    # Segment detail drill-down
    st.markdown('<div class="section-header">Segment Deep Dive</div>', unsafe_allow_html=True)
    selected_seg = st.selectbox("Select a segment to explore:", features_df['segment_name'].unique())
    seg_data = features_df[features_df['segment_name'] == selected_seg]

    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        st.metric("Customers", f"{len(seg_data):,}")
    with scol2:
        st.metric("Churn Rate", f"{seg_data['churned'].mean():.1%}")
    with scol3:
        st.metric("Avg CLV", f"${seg_data['clv'].mean():,.0f}")
    with scol4:
        st.metric("Revenue at Risk", f"${seg_data['revenue_at_risk'].sum():,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(seg_data, x='churn_probability', nbins=30,
                           color_discrete_sequence=['#3498db'],
                           title=f'{selected_seg}: Churn Probability Distribution',
                           labels={'churn_probability': 'Churn Probability'})
        fig.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(seg_data, x='clv', nbins=30,
                            color_discrete_sequence=['#27ae60'],
                            title=f'{selected_seg}: CLV Distribution',
                            labels={'clv': 'Customer Lifetime Value (CAD)'})
        fig2.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: RETENTION ACTIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Retention Actions":
    st.title("Retention Action Center")
    st.markdown("*Prioritized action list for the marketing operations team*")
    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect("Risk Level", ['High', 'Medium', 'Low'],
                                     default=['High', 'Medium'])
    with col2:
        seg_filter = st.multiselect("Segment", sorted(retention_df['segment_name'].unique()),
                                    default=sorted(retention_df['segment_name'].unique()))
    with col3:
        tier_filter = st.multiselect("Loyalty Tier", sorted(retention_df['loyalty_card'].unique()),
                                     default=sorted(retention_df['loyalty_card'].unique()))

    filtered = retention_df[
        retention_df['risk_level'].isin(risk_filter) &
        retention_df['segment_name'].isin(seg_filter) &
        retention_df['loyalty_card'].isin(tier_filter)
    ].copy()

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Customers Selected", f"{len(filtered):,}")
    with mc2:
        st.metric("Revenue at Risk", f"${filtered['revenue_at_risk'].sum():,.0f}")
    with mc3:
        st.metric("Potential Save", f"${filtered['potential_save'].sum():,.0f}")
    with mc4:
        st.metric("Avg Churn Prob", f"{filtered['churn_probability'].mean():.1%}")

    st.markdown("---")

    # Action distribution charts
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">Actions by Channel</div>', unsafe_allow_html=True)
        channel_counts = filtered['channel'].value_counts().reset_index()
        channel_counts.columns = ['Channel', 'Count']
        fig = px.bar(channel_counts, x='Count', y='Channel', orientation='h',
                     color='Count', color_continuous_scale='Blues',
                     title='Customers by Recommended Channel')
        fig.update_layout(height=300, coloraxis_showscale=False,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Revenue at Risk by Risk Level</div>', unsafe_allow_html=True)
        rev_risk = filtered.groupby('risk_level')['revenue_at_risk'].sum().reset_index()
        fig2 = px.bar(rev_risk, x='risk_level', y='revenue_at_risk',
                      color='risk_level', color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'},
                      title='Revenue at Risk by Risk Level',
                      labels={'revenue_at_risk': 'Revenue at Risk (CAD)', 'risk_level': 'Risk Level'})
        fig2.update_yaxes(tickprefix='$', tickformat=',')
        fig2.update_layout(height=300, showlegend=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # Action table
    st.markdown('<div class="section-header">Prioritized Action List</div>', unsafe_allow_html=True)

    display_cols = ['loyalty_number', 'loyalty_card', 'segment_name', 'risk_level',
                    'churn_probability', 'clv', 'revenue_at_risk', 'potential_save',
                    'recommended_action', 'specific_offer', 'channel', 'timing',
                    'incentive_cost_cad']

    display_df = filtered[display_cols].sort_values('revenue_at_risk', ascending=False).copy()
    display_df['churn_probability'] = display_df['churn_probability'].map('{:.1%}'.format)
    display_df['clv'] = display_df['clv'].map('${:,.0f}'.format)
    display_df['revenue_at_risk'] = display_df['revenue_at_risk'].map('${:,.0f}'.format)
    display_df['potential_save'] = display_df['potential_save'].map('${:,.0f}'.format)
    display_df['incentive_cost_cad'] = display_df['incentive_cost_cad'].map('${}'.format)

    st.dataframe(display_df.head(200), use_container_width=True, hide_index=True,
                 column_config={
                     'risk_level': st.column_config.TextColumn('Risk', width='small'),
                     'loyalty_number': st.column_config.NumberColumn('ID', format='%d'),
                 })

    # Download button
    csv_data = filtered[display_cols].sort_values('revenue_at_risk', ascending=False).to_csv(index=False)
    st.download_button(
        label="📥 Download Action List (CSV)",
        data=csv_data,
        file_name=f"retention_actions_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
        type='primary'
    )

    # Playbook
    st.markdown("---")
    st.markdown('<div class="section-header">Retention Playbook by Segment</div>', unsafe_allow_html=True)

    playbook_data = {
        'Silent Drifters': {
            'Profile': 'Inactive members with no Q4 2017 flights. High recency, declining engagement.',
            'High Risk Action': '8,000 bonus miles + waived change fees — SMS + Email + App, within 48h',
            'Medium Risk Action': '5,000 bonus miles + 3-month tier extension — Email + SMS, within 7 days',
            'Low Risk Action': '2,000 bonus miles on next booking — Email, within 21 days',
            'KPI to Track': 'Booking rate within 90 days of intervention',
        },
        'Seasonal Travelers': {
            'Profile': 'Fly only during peak seasons. Miss off-peak months entirely.',
            'High Risk Action': '4,000 bonus miles + exclusive off-peak deal — SMS + Email, within 72h',
            'Medium Risk Action': 'Upcoming season preview + 2,000 miles — Email, within 14 days',
            'Low Risk Action': 'Seasonal deals newsletter — Email, within 30 days',
            'KPI to Track': 'Off-season booking conversion rate',
        },
        'Premium Loyalists': {
            'Profile': 'High CLV Aurora/Nova members with consistent activity.',
            'High Risk Action': 'Companion ticket + Aurora status guarantee — Personal call, within 24h',
            'Medium Risk Action': '10,000 bonus miles + lounge upgrade — Premium email, within 3 days',
            'Low Risk Action': '3,000 appreciation miles — Email, within 30 days',
            'KPI to Track': 'Tier renewal rate, annual flight frequency',
        },
        'Active Champions': {
            'Profile': 'Consistent fliers, high engagement, low churn risk.',
            'High Risk Action': '6,000 bonus miles + priority check-in 6 months — Email + App, within 3 days',
            'Medium Risk Action': '3,000 bonus miles on next flight — Email, within 14 days',
            'Low Risk Action': '1,500 appreciation miles — Email, within 30 days',
            'KPI to Track': 'Points redemption rate, NPS score',
        },
    }

    for seg_name, details in playbook_data.items():
        with st.expander(f"📋 {seg_name}"):
            pcol1, pcol2 = st.columns([1, 2])
            with pcol1:
                st.markdown(f"**Profile:** {details['Profile']}")
                st.markdown(f"**KPI:** {details['KPI to Track']}")
            with pcol2:
                st.markdown(f"🔴 **High Risk:** {details['High Risk Action']}")
                st.markdown(f"🟡 **Medium Risk:** {details['Medium Risk Action']}")
                st.markdown(f"🟢 **Low Risk:** {details['Low Risk Action']}")
