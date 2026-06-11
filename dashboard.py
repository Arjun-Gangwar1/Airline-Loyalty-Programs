"""
Airline Loyalty Behavioral Intelligence — Streamlit Dashboard
==============================================================
5 pages:
  1. Executive Overview     — KPIs, risk distribution, top priority list
  2. Churn Intelligence     — Model performance, feature importance, distributions
  3. Segment Analysis       — Behavioral profiles, heatmap, drill-down
  4. Geographic & Demographics — Province map, cohort analysis, demographic splits
  5. Retention Actions      — Filterable table, download, playbooks

Run:  venv/bin/streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Loyalty Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 8px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
    .metric-delta { font-size: 0.8rem; margin-top: 4px; }
    .risk-high    { color: #e74c3c; font-weight: bold; }
    .risk-medium  { color: #f39c12; font-weight: bold; }
    .risk-low     { color: #27ae60; font-weight: bold; }
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #1e3c72;
        border-bottom: 2px solid #2a5298;
        padding-bottom: 6px; margin: 16px 0 12px 0;
    }
    .insight-box {
        background: #f0f4ff; border-left: 4px solid #2a5298;
        padding: 10px 14px; border-radius: 4px; margin: 6px 0;
        font-size: 0.9rem;
    }
    .warning-box {
        background: #fff3cd; border-left: 4px solid #f39c12;
        padding: 10px 14px; border-radius: 4px; margin: 6px 0;
        font-size: 0.9rem;
    }
    .success-box {
        background: #d4edda; border-left: 4px solid #27ae60;
        padding: 10px 14px; border-radius: 4px; margin: 6px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}
SEG_PALETTE = px.colors.qualitative.Set2


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_all():
    final   = Path("data/final")
    reports = Path("outputs/reports")

    df      = pd.read_csv(final / "customer_segments.csv")
    ret     = pd.read_csv(reports / "retention_actions.csv")
    mc      = pd.read_csv(reports / "model_comparison.csv")
    fi      = pd.read_csv(reports / "feature_importance.csv")
    with open(reports / "pipeline_summary.json") as f:
        summary = json.load(f)

    # Ensure churn_probability and risk_level exist
    # (they are saved by the pipeline into customer_segments.csv)
    if 'churn_probability' not in df.columns:
        # Fallback: merge from retention CSV
        prob_map = ret.set_index('loyalty_number')['churn_probability']
        df['churn_probability'] = df['loyalty_number'].map(prob_map).fillna(0)
    if 'risk_level' not in df.columns:
        df['risk_level'] = pd.cut(
            df['churn_probability'],
            bins=[0, 0.40, 0.70, 1.0],
            labels=['Low', 'Medium', 'High']
        ).astype(str)
    if 'revenue_at_risk' not in df.columns:
        df['revenue_at_risk'] = (df['clv'] * df['churn_probability'] * 0.82).round(2)
    if 'potential_save' not in df.columns:
        df['potential_save'] = (df['revenue_at_risk'] * 0.27).round(2)

    # Province — load from raw file only if not already in segments CSV
    if 'province' not in df.columns:
        raw_path = Path("data/raw/Customer Loyalty History.csv")
        if raw_path.exists():
            loyalty_raw = pd.read_csv(raw_path)
            loyalty_raw.columns = [c.lower().replace(' ', '_') for c in loyalty_raw.columns]
            df = df.merge(
                loyalty_raw[['loyalty_number', 'province']].drop_duplicates(),
                on='loyalty_number', how='left'
            )
        else:
            df['province'] = 'Unknown'

    return df, ret, mc, fi, summary


# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df, ret, mc, fi, summary = load_all()
except Exception as e:
    st.error(f"❌ Data not found. Run `venv/bin/python complete_pipeline.py` first.\n\n{e}")
    st.stop()

# Derived globals
segments    = sorted(df['segment_name'].unique())
risk_levels = ['High', 'Medium', 'Low']
total_cust  = len(df)
churn_rate  = df['churned'].mean()
total_rar   = df['revenue_at_risk'].sum()
total_save  = df['potential_save'].sum()
n_high      = (df['risk_level'] == 'High').sum()
n_medium    = (df['risk_level'] == 'Medium').sum()
best_auc    = summary.get('best_model_auc', summary.get('xgboost_auc', 0))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/airplane-mode-on.png", width=70)
    st.title("✈️ Loyalty Intelligence")
    st.caption("Airline Behavioral Analytics Platform")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Executive Overview",
         "📊 Churn Intelligence",
         "👥 Segment Analysis",
         "🗺️ Geography & Demographics",
         "🎯 Retention Actions"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"**Dataset:** {total_cust:,} customers")
    st.markdown(f"**Analysis date:** {summary.get('analysis_date','')[:10]}")
    st.markdown(f"**Best AUC:** {best_auc:.4f}")
    st.markdown("---")
    st.caption("IIT Guwahati C&A Club · Summer Projects '26")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Overview":
    st.title("🏠 Executive Overview")
    st.caption("Real-time view of loyalty health, churn risk, and revenue at stake")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total_cust:,}</div>
            <div class="metric-label">Total Members</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{churn_rate:.1%}</div>
            <div class="metric-label">Combined Churn Rate</div>
            <div class="metric-delta">Hard 3.9% + Silent 12.7%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#c0392b,#e74c3c)">
            <div class="metric-value">{n_high:,}</div>
            <div class="metric-label">High-Risk Customers</div>
            <div class="metric-delta">Churn prob ≥ 70%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#c0392b,#e74c3c)">
            <div class="metric-value">${total_rar/1e6:.1f}M</div>
            <div class="metric-label">Revenue at Risk (CAD)</div>
            <div class="metric-delta">CLV × churn prob × 0.82</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#1a6b3c,#27ae60)">
            <div class="metric-value">${total_save/1e6:.1f}M</div>
            <div class="metric-label">Potential Savings (CAD)</div>
            <div class="metric-delta">27% retention lift est.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two column layout ─────────────────────────────────────────────────────
    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown('<div class="section-header">Revenue at Risk by Segment</div>', unsafe_allow_html=True)
        seg_rev = df.groupby('segment_name').agg(
            revenue=('revenue_at_risk','sum'),
            customers=('loyalty_number','count'),
            churn_rate=('churned','mean')
        ).reset_index().sort_values('revenue', ascending=False)

        fig = px.bar(
            seg_rev, x='segment_name', y='revenue',
            color='churn_rate',
            color_continuous_scale=['#27ae60','#f39c12','#e74c3c'],
            labels={'revenue':'Revenue at Risk (CAD)','segment_name':'Segment',
                    'churn_rate':'Churn Rate'},
            text=seg_rev['revenue'].apply(lambda x: f'${x/1e6:.1f}M'),
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, margin=dict(t=20,b=20),
                          xaxis_tickangle=-20,
                          coloraxis_colorbar=dict(title="Churn Rate", tickformat='.0%'))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = df['risk_level'].value_counts().reindex(['High','Medium','Low']).fillna(0)
        fig2 = go.Figure(go.Pie(
            labels=[f"{k} Risk<br>({int(v):,})" for k, v in risk_counts.items()],
            values=risk_counts.values,
            hole=0.55,
            marker_colors=[RISK_COLORS[k] for k in risk_counts.index],
            textinfo='percent+label',
            textfont_size=11,
        ))
        fig2.update_layout(
            height=310, margin=dict(t=20,b=10,l=10,r=10),
            showlegend=False,
            annotations=[dict(text=f'{n_high+n_medium:,}<br>At-Risk', x=0.5, y=0.5,
                              font_size=14, showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Top 15 priority customers ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Top 15 Priority Customers (by Revenue at Risk)</div>',
                unsafe_allow_html=True)
    top15 = df[df['risk_level'] == 'High'].nlargest(15, 'revenue_at_risk')[
        ['loyalty_number','loyalty_card','segment_name','churn_probability',
         'clv','revenue_at_risk','potential_save','engagement_health_score']
    ].copy()
    top15.columns = ['Member ID','Tier','Segment','Churn Prob','CLV (CAD)',
                     'Revenue at Risk','Potential Save','Engagement Score']
    top15['Churn Prob'] = top15['Churn Prob'].apply(lambda x: f"{x:.1%}")
    top15['CLV (CAD)']  = top15['CLV (CAD)'].apply(lambda x: f"${x:,.0f}")
    top15['Revenue at Risk'] = top15['Revenue at Risk'].apply(lambda x: f"${x:,.0f}")
    top15['Potential Save']  = top15['Potential Save'].apply(lambda x: f"${x:,.0f}")
    top15['Engagement Score'] = top15['Engagement Score'].apply(lambda x: f"{x:.0f}")
    st.dataframe(top15, use_container_width=True, hide_index=True)

    # ── Churn definition comparison ───────────────────────────────────────────
    st.markdown('<div class="section-header">Churn Definition Comparison</div>', unsafe_allow_html=True)
    comp_data = {
        'Definition': ['Hard Churn (Formal Cancellation)', 'Activity Churn (Silent, no 2018 flights)',
                       'Combined (Adopted)'],
        'Rate': [f"{summary['churn_rate_hard']:.1%}", f"{summary['churn_rate_activity']:.1%}",
                 f"{summary['churn_rate_combined']:.1%}"],
        'Customers': [645, 2123, 2728],
        'Coverage': ['Formal only — understates problem',
                     'Silent quitters — better but may miss some',
                     '✅ Most complete business picture'],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHURN INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Churn Intelligence":
    st.title("📊 Churn Intelligence")
    st.caption("Model performance, feature importance, and churn probability distributions")

    # ── Model comparison cards ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    model_results = summary.get('model_results', {})
    cols = st.columns(len(model_results))
    for col, (mname, mres) in zip(cols, model_results.items()):
        is_best = mres['AUC'] == max(v['AUC'] for v in model_results.values())
        border  = "border: 2px solid #27ae60;" if is_best else ""
        badge   = " ★ Best" if is_best else ""
        col.markdown(f"""
        <div class="metric-card" style="{border}">
            <div style="font-size:0.9rem; font-weight:600; margin-bottom:8px">{mname}{badge}</div>
            <table style="width:100%;font-size:0.82rem;color:white">
                <tr><td>AUC</td><td style="text-align:right"><b>{mres['AUC']:.4f}</b></td></tr>
                <tr><td>Recall</td><td style="text-align:right">{mres['Recall']:.4f}</td></tr>
                <tr><td>Precision</td><td style="text-align:right">{mres['Precision']:.4f}</td></tr>
                <tr><td>F1</td><td style="text-align:right">{mres['F1']:.4f}</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    # CV scores if available
    cv = summary.get('cv_scores', {})
    if cv:
        st.markdown('<div class="insight-box">💡 5-Fold Cross-Validation AUC — '
                    + '  |  '.join(f"<b>{k}</b>: {v:.4f}" for k, v in cv.items())
                    + '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-header">Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
        top_fi = fi.head(15).sort_values('importance')
        fig = px.bar(
            top_fi, x='importance', y='feature', orientation='h',
            color='importance', color_continuous_scale='Reds',
            labels={'importance':'Importance Score','feature':'Feature'},
        )
        fig.update_layout(height=450, margin=dict(t=10,b=10),
                          coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Churn Probability Distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(
            df, x='churn_probability', nbins=60,
            color_discrete_sequence=['#e74c3c'],
            labels={'churn_probability':'Churn Probability','count':'Customers'},
        )
        fig2.add_vline(x=0.40, line_dash='dash', line_color='#f39c12',
                       annotation_text='Medium (0.40)', annotation_position='top right')
        fig2.add_vline(x=0.70, line_dash='dash', line_color='#e74c3c',
                       annotation_text='High (0.70)', annotation_position='top right')
        fig2.update_layout(height=230, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Churn Rate by Loyalty Tier</div>', unsafe_allow_html=True)
        tier_ch = df.groupby('loyalty_card').agg(
            churn_rate=('churned','mean'),
            count=('loyalty_number','count'),
            avg_clv=('clv','mean')
        ).reset_index().sort_values('churn_rate', ascending=False)
        fig3 = px.bar(
            tier_ch, x='loyalty_card', y='churn_rate',
            color='loyalty_card',
            color_discrete_sequence=px.colors.qualitative.Set2,
            text=tier_ch['churn_rate'].apply(lambda x: f'{x:.1%}'),
            labels={'churn_rate':'Churn Rate','loyalty_card':'Loyalty Tier'},
        )
        fig3.update_traces(textposition='outside')
        fig3.update_yaxes(tickformat='.0%')
        fig3.update_layout(height=230, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── CLV vs Churn scatter ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">CLV vs Churn Probability (Risk Matrix)</div>', unsafe_allow_html=True)
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig4 = px.scatter(
        sample, x='clv', y='churn_probability',
        color='segment_name', size='revenue_at_risk',
        hover_data=['loyalty_number','loyalty_card','engagement_health_score'],
        labels={'clv':'Customer Lifetime Value (CAD)',
                'churn_probability':'Churn Probability',
                'segment_name':'Segment'},
        color_discrete_sequence=SEG_PALETTE,
        opacity=0.6,
    )
    fig4.add_hline(y=0.70, line_dash='dash', line_color='#e74c3c',
                   annotation_text='High Risk Threshold')
    fig4.add_hline(y=0.40, line_dash='dash', line_color='#f39c12',
                   annotation_text='Medium Risk Threshold')
    fig4.update_layout(height=420, margin=dict(t=20,b=20))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Interpretation:</b> Top-right quadrant = High CLV + High Churn = '
                'highest business priority. These customers are valuable <i>and</i> leaving — '
                'the model catches them before they go silent.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SEGMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Segment Analysis":
    st.title("👥 Customer Segment Analysis")
    st.caption("Behaviorally distinct segments with specific retention strategies")

    # ── Segment summary cards ─────────────────────────────────────────────────
    seg_stats = df.groupby('segment_name').agg(
        count=('loyalty_number','count'),
        churn_rate=('churned','mean'),
        avg_clv=('clv','mean'),
        avg_engagement=('engagement_health_score','mean'),
        revenue_at_risk=('revenue_at_risk','sum'),
    ).reset_index().sort_values('churn_rate', ascending=False)

    cols = st.columns(len(seg_stats))
    for col, (_, row) in zip(cols, seg_stats.iterrows()):
        color = '#e74c3c' if row['churn_rate'] > 0.25 else '#f39c12' if row['churn_rate'] > 0.10 else '#27ae60'
        col.markdown(f"""
        <div class="metric-card" style="background:linear-gradient(135deg,{color},{color}aa)">
            <div style="font-size:0.85rem;font-weight:700">{row['segment_name']}</div>
            <div class="metric-value">{row['count']:,}</div>
            <div class="metric-label">{row['churn_rate']:.0%} churn · ${row['avg_clv']:,.0f} avg CLV</div>
            <div class="metric-delta">Engagement: {row['avg_engagement']:.0f}/100</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk vs Value scatter ──────────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-header">Risk vs Value Matrix</div>', unsafe_allow_html=True)
        fig = px.scatter(
            seg_stats, x='avg_clv', y='churn_rate',
            size='count', color='segment_name',
            text='segment_name',
            color_discrete_sequence=SEG_PALETTE,
            labels={'avg_clv':'Average CLV (CAD)','churn_rate':'Churn Rate',
                    'count':'Segment Size','segment_name':'Segment'},
            size_max=50,
        )
        fig.update_traces(textposition='top center')
        fig.update_yaxes(tickformat='.0%')
        fig.update_layout(height=380, margin=dict(t=20,b=10), showlegend=False)
        # Add quadrant lines at medians
        fig.add_hline(y=seg_stats['churn_rate'].median(), line_dash='dot',
                      line_color='gray', opacity=0.5)
        fig.add_vline(x=seg_stats['avg_clv'].median(), line_dash='dot',
                      line_color='gray', opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Revenue at Risk by Segment</div>', unsafe_allow_html=True)
        seg_rev = seg_stats.sort_values('revenue_at_risk', ascending=False)
        fig2 = go.Figure(go.Bar(
            x=seg_rev['segment_name'],
            y=seg_rev['revenue_at_risk'],
            marker_color=SEG_PALETTE[:len(seg_rev)],
            text=[f'${v/1e6:.1f}M' for v in seg_rev['revenue_at_risk']],
            textposition='outside',
        ))
        fig2.update_layout(
            height=380, margin=dict(t=20,b=10),
            yaxis_title='Revenue at Risk (CAD)',
            xaxis_tickangle=-15,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Behavioral heatmap ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Segment Behavioral Profiles (Normalized 0–1)</div>',
                unsafe_allow_html=True)
    hmap_feats = ['avg_flights_per_month','active_month_ratio','recency_months',
                  'redemption_rate','engagement_health_score','tenure_months',
                  'tier_numeric','clv_log','dollar_cost_per_flight','q4_flights']
    hmap_labels = ['Avg Flights/Month','Active Month Ratio','Recency (months)',
                   'Redemption Rate','Engagement Score','Tenure (months)',
                   'Loyalty Tier','CLV (log)','$ Cost/Flight','Q4 Flights']
    hmap_data = df.groupby('segment_name')[hmap_feats].mean()
    hmap_norm = (hmap_data - hmap_data.min()) / (hmap_data.max() - hmap_data.min() + 1e-6)
    hmap_norm.columns = hmap_labels

    fig3 = px.imshow(
        hmap_norm,
        color_continuous_scale='RdYlGn',
        text_auto='.2f',
        labels=dict(color='Normalized Score'),
        aspect='auto',
    )
    fig3.update_layout(height=300, margin=dict(t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Segment drill-down ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Segment Drill-Down</div>', unsafe_allow_html=True)
    selected_seg = st.selectbox("Select a segment to explore", sorted(segments))
    seg_df = df[df['segment_name'] == selected_seg]
    s = seg_df.agg({
        'loyalty_number':'count', 'churned':'mean', 'clv':'mean',
        'avg_flights_per_month':'mean', 'recency_months':'mean',
        'active_month_ratio':'mean', 'redemption_rate':'mean',
        'engagement_health_score':'mean', 'tenure_months':'mean',
    }).to_dict()

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Customers",        f"{int(s['loyalty_number']):,}")
    cc2.metric("Churn Rate",       f"{s['churned']:.1%}")
    cc3.metric("Avg CLV (CAD)",    f"${s['clv']:,.0f}")
    cc4.metric("Engagement Score", f"{s['engagement_health_score']:.0f}/100")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Flights/Month",  f"{s['avg_flights_per_month']:.2f}")
    c6.metric("Active Month Ratio", f"{s['active_month_ratio']:.1%}")
    c7.metric("Redemption Rate",    f"{s['redemption_rate']:.1%}")
    c8.metric("Avg Tenure",         f"{s['tenure_months']:.0f} months")

    d1, d2 = st.columns(2)
    with d1:
        fig4 = px.histogram(seg_df, x='churn_probability', nbins=40,
                            color_discrete_sequence=['#e74c3c'],
                            labels={'churn_probability':'Churn Probability','count':'Customers'},
                            title=f"Churn Probability — {selected_seg}")
        fig4.update_layout(height=280, margin=dict(t=30,b=10))
        st.plotly_chart(fig4, use_container_width=True)
    with d2:
        fig5 = px.histogram(seg_df, x='clv', nbins=40,
                            color_discrete_sequence=['#3498db'],
                            labels={'clv':'CLV (CAD)','count':'Customers'},
                            title=f"CLV Distribution — {selected_seg}")
        fig5.update_layout(height=280, margin=dict(t=30,b=10))
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GEOGRAPHY & DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Geography & Demographics":
    st.title("🗺️ Geography & Demographics")
    st.caption("Where churn happens, who churns, and when members were acquired")

    # ── Geographic ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Churn Rate by Province</div>', unsafe_allow_html=True)

    prov_col = 'province' if 'province' in df.columns else 'province_raw'
    prov_stats = df.groupby(prov_col).agg(
        churn_rate=('churned','mean'),
        count=(prov_col,'count'),
        avg_clv=('clv','mean'),
        revenue_at_risk=('revenue_at_risk','sum'),
        avg_engagement=('engagement_health_score','mean'),
    ).reset_index().rename(columns={prov_col:'province'}).sort_values('churn_rate', ascending=False)

    p1, p2 = st.columns(2)
    with p1:
        fig = px.bar(
            prov_stats, x='province', y='churn_rate',
            color='churn_rate',
            color_continuous_scale=['#27ae60','#f39c12','#e74c3c'],
            text=prov_stats['churn_rate'].apply(lambda x: f'{x:.1%}'),
            labels={'churn_rate':'Churn Rate','province':'Province'},
        )
        fig.add_hline(y=churn_rate, line_dash='dash', line_color='navy',
                      annotation_text=f'National avg {churn_rate:.1%}')
        fig.update_traces(textposition='outside')
        fig.update_yaxes(tickformat='.0%')
        fig.update_layout(height=380, margin=dict(t=10,b=10),
                          xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        fig2 = px.bar(
            prov_stats, x='province', y='revenue_at_risk',
            color='count',
            color_continuous_scale='Blues',
            text=prov_stats['revenue_at_risk'].apply(lambda x: f'${x/1e6:.1f}M'),
            labels={'revenue_at_risk':'Revenue at Risk (CAD)','province':'Province',
                    'count':'Customers'},
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=380, margin=dict(t=10,b=10),
                           xaxis_tickangle=-30,
                           coloraxis_colorbar=dict(title='Customers'))
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        prov_stats.rename(columns={
            'province':'Province','churn_rate':'Churn Rate','count':'Customers',
            'avg_clv':'Avg CLV','revenue_at_risk':'Revenue at Risk',
            'avg_engagement':'Avg Engagement'
        }).assign(**{
            'Churn Rate': prov_stats['churn_rate'].apply(lambda x: f'{x:.1%}'),
            'Avg CLV': prov_stats['avg_clv'].apply(lambda x: f'${x:,.0f}'),
            'Revenue at Risk': prov_stats['revenue_at_risk'].apply(lambda x: f'${x:,.0f}'),
            'Avg Engagement': prov_stats['avg_engagement'].apply(lambda x: f'{x:.0f}'),
        }),
        use_container_width=True, hide_index=True
    )

    # ── Cohort Analysis ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Enrollment Year Cohort Analysis</div>',
                unsafe_allow_html=True)
    cohort = df.groupby('enrollment_year').agg(
        churn_rate=('churned','mean'),
        count=('loyalty_number','count'),
        avg_clv=('clv','mean'),
        avg_engagement=('engagement_health_score','mean'),
    ).reset_index()

    co1, co2 = st.columns(2)
    with co1:
        bar_colors = ['#e74c3c' if r > 0.20 else '#f39c12' if r > 0.13 else '#27ae60'
                      for r in cohort['churn_rate']]
        fig3 = go.Figure(go.Bar(
            x=cohort['enrollment_year'].astype(str),
            y=cohort['churn_rate'],
            marker_color=bar_colors,
            text=[f'{r:.1%}' for r in cohort['churn_rate']],
            textposition='outside',
        ))
        fig3.add_hline(y=churn_rate, line_dash='dash', line_color='navy',
                       annotation_text=f'Overall avg {churn_rate:.1%}')
        fig3.update_yaxes(tickformat='.0%')
        fig3.update_layout(
            height=320, margin=dict(t=20,b=10),
            title_text='Churn Rate by Enrollment Year',
            xaxis_title='Enrollment Year', yaxis_title='Churn Rate',
        )
        st.plotly_chart(fig3, use_container_width=True)

    with co2:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=cohort['enrollment_year'].astype(str),
            y=cohort['count'],
            name='Customers',
            marker_color='#3498db', opacity=0.7,
            yaxis='y',
        ))
        fig4.add_trace(go.Scatter(
            x=cohort['enrollment_year'].astype(str),
            y=cohort['avg_engagement'],
            name='Avg Engagement',
            marker_color='#e74c3c',
            mode='lines+markers', line=dict(width=2),
            yaxis='y2',
        ))
        fig4.update_layout(
            height=320, margin=dict(t=20,b=10),
            title_text='Cohort Size & Engagement Score',
            xaxis_title='Enrollment Year',
            yaxis=dict(title='Number of Customers', side='left'),
            yaxis2=dict(title='Avg Engagement Score', side='right', overlaying='y'),
            legend=dict(x=0.7, y=1.1, orientation='h'),
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 <b>Key Finding:</b> Members who enrolled in 2017–2018 '
        'show the highest churn rates — they have the shortest tenure and the least time to '
        'build loyalty habits. First-year retention programs (milestone rewards at 3/6/12 months) '
        'would directly address this pattern.</div>',
        unsafe_allow_html=True
    )

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Demographic Churn Patterns</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)

    with d1:
        edu_ch = df.groupby('education')['churned'].agg(['mean','count']).reset_index()
        edu_order = ['High School or Below','College','Bachelor','Master','Doctor']
        edu_ch = edu_ch.set_index('education').reindex(
            [e for e in edu_order if e in edu_ch['education'].values]
        ).reset_index()
        fig5 = px.bar(
            edu_ch, x='education', y='mean',
            color='mean', color_continuous_scale=['#27ae60','#f39c12','#e74c3c'],
            text=edu_ch['mean'].apply(lambda x: f'{x:.1%}'),
            title='Churn by Education',
            labels={'mean':'Churn Rate','education':'Education'},
        )
        fig5.update_traces(textposition='outside')
        fig5.update_yaxes(tickformat='.0%')
        fig5.update_layout(height=320, margin=dict(t=40,b=10),
                           xaxis_tickangle=-25, coloraxis_showscale=False,
                           showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    with d2:
        gen_ch = df.groupby('gender')['churned'].agg(['mean','count']).reset_index()
        fig6 = px.bar(
            gen_ch, x='gender', y='mean',
            color='gender',
            color_discrete_sequence=['#e91e8c','#2196F3'],
            text=gen_ch['mean'].apply(lambda x: f'{x:.1%}'),
            title='Churn by Gender',
            labels={'mean':'Churn Rate','gender':'Gender'},
        )
        fig6.update_traces(textposition='outside')
        fig6.update_yaxes(tickformat='.0%')
        fig6.update_layout(height=320, margin=dict(t=40,b=10),
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)

    with d3:
        mar_ch = df.groupby('marital_status')['churned'].agg(['mean','count']).reset_index()
        fig7 = px.bar(
            mar_ch, x='marital_status', y='mean',
            color='marital_status',
            color_discrete_sequence=['#9b59b6','#1abc9c','#e67e22'],
            text=mar_ch['mean'].apply(lambda x: f'{x:.1%}'),
            title='Churn by Marital Status',
            labels={'mean':'Churn Rate','marital_status':'Marital Status'},
        )
        fig7.update_traces(textposition='outside')
        fig7.update_yaxes(tickformat='.0%')
        fig7.update_layout(height=320, margin=dict(t=40,b=10), showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RETENTION ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Retention Actions":
    st.title("🎯 Retention Actions")
    st.caption("Filterable action plan — filter, review, and download for CRM import")

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("At-Risk Customers", f"{len(ret):,}")
    m2.metric("High-Risk",         f"{(ret['risk_level']=='High').sum():,}")
    m3.metric("Total Revenue at Risk",
              f"${ret['revenue_at_risk'].sum():,.0f}")
    m4.metric("Potential Recoverable",
              f"${ret['potential_save'].sum():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        risk_filter = st.multiselect(
            "Risk Level",
            options=['High','Medium'],
            default=['High','Medium'],
        )
    with f2:
        seg_filter = st.multiselect(
            "Segment",
            options=sorted(ret['segment_name'].unique()),
            default=sorted(ret['segment_name'].unique()),
        )
    with f3:
        tier_filter = st.multiselect(
            "Loyalty Tier",
            options=sorted(ret['loyalty_card'].unique()),
            default=sorted(ret['loyalty_card'].unique()),
        )
    with f4:
        prov_opts = sorted(ret['province'].unique()) if 'province' in ret.columns else []
        if prov_opts:
            prov_filter = st.multiselect("Province", options=prov_opts, default=prov_opts)
        else:
            prov_filter = []

    # ── Filter data ───────────────────────────────────────────────────────────
    mask = (
        ret['risk_level'].isin(risk_filter) &
        ret['segment_name'].isin(seg_filter) &
        ret['loyalty_card'].isin(tier_filter)
    )
    if prov_filter and 'province' in ret.columns:
        mask &= ret['province'].isin(prov_filter)
    filtered = ret[mask].copy()

    st.markdown(f"**Showing {len(filtered):,} customers** — "
                f"Revenue at Risk: ${filtered['revenue_at_risk'].sum():,.0f}  |  "
                f"Potential Save: ${filtered['potential_save'].sum():,.0f}")

    # ── Display cols ──────────────────────────────────────────────────────────
    show_cols = ['loyalty_number','loyalty_card','segment_name','risk_level',
                 'churn_probability','clv','revenue_at_risk','potential_save',
                 'recommended_action','specific_offer','channel','timing']
    if 'province' in filtered.columns:
        show_cols.insert(2, 'province')

    display_df = filtered[show_cols].copy()
    display_df['churn_probability'] = display_df['churn_probability'].apply(lambda x: f'{x:.1%}')
    display_df['clv']               = display_df['clv'].apply(lambda x: f'${x:,.0f}')
    display_df['revenue_at_risk']   = display_df['revenue_at_risk'].apply(lambda x: f'${x:,.0f}')
    display_df['potential_save']    = display_df['potential_save'].apply(lambda x: f'${x:,.0f}')

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=380)

    # ── Download ───────────────────────────────────────────────────────────────
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered CSV (CRM Ready)",
        data=csv,
        file_name=f"retention_actions_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    # ── Retention playbooks ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Retention Playbooks by Segment</div>',
                unsafe_allow_html=True)

    playbooks = {
        'Active Champions': {
            'icon': '🏆',
            'tagline': 'Low churn, high CLV — protect them',
            'high': 'Loyalty Lock-In: 6,000 bonus miles + priority check-in for 6 months. '
                    'Channel: Email + App push. Timeline: within 3 days.',
            'medium': 'Engagement Boost: Double miles on next 3 flights. Email. 14 days.',
            'low': 'Quarterly appreciation: 1,500 bonus miles. Email.',
        },
        'Miles Hoarders': {
            'icon': '💰',
            'tagline': 'Frequent fliers with low CLV — grow their value',
            'high': 'Redemption Activation: 2× redemption value + 60-day flash sale on points. '
                    'Email + App push. Within 72 hours.',
            'medium': 'Redemption Reminder: 50% bonus on any redemption within 60 days. Email. 10 days.',
            'low': 'Points Expiry Warning + 500 bonus miles. Email. Monthly.',
        },
        'Premium Dormant': {
            'icon': '⚠️',
            'tagline': 'High CLV but disengaged — URGENT intervention',
            'high': 'VIP Win-Back: Complimentary companion ticket + 12-month status guarantee. '
                    'Personal phone call + premium email. Within 24 hours.',
            'medium': '10,000 bonus miles + 3 lounge access visits. Email. Within 3 days.',
            'low': '3,000 bonus miles loyalty recognition. Email. Monthly.',
        },
        'Seasonal Travelers': {
            'icon': '🌤️',
            'tagline': 'Low-value, seasonal — convert to year-round',
            'high': 'Off-Season Reactivation: 4,000 bonus miles + 30% off off-peak fare. '
                    'Email + SMS. Within 72 hours.',
            'medium': 'Seasonal Campaign: upcoming season preview + 2,000 bonus miles. Email. 14 days.',
            'low': 'Seasonal deals newsletter. Email. Monthly.',
        },
    }

    for seg_name, pb in playbooks.items():
        if seg_name in segments:
            with st.expander(f"{pb['icon']} {seg_name} — {pb['tagline']}"):
                st.markdown(f"🔴 **High Risk:** {pb['high']}")
                st.markdown(f"🟡 **Medium Risk:** {pb['medium']}")
                st.markdown(f"🟢 **Low Risk:** {pb['low']}")

    # Fallback for any new segment names not in playbooks
    for seg_name in segments:
        if seg_name not in playbooks:
            with st.expander(f"📋 {seg_name}"):
                st.markdown("**High Risk:** Priority retention offer — 5,000 bonus miles + 60-day waived fees. "
                             "Email + SMS. Within 48 hours.")
                st.markdown("**Medium Risk:** 2,500 bonus miles re-engagement. Email. 14 days.")
                st.markdown("**Low Risk:** 1,000 bonus miles appreciation. Email. Monthly.")
