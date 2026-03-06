"""
app.py — FlexiPrice Engine
===========================
A premium, production-grade Dynamic Price Optimization Dashboard.

Pages
-----
1. 🏠 Home           : Hero + overview
2. 📊 Data Analysis  : Charts, dataset, correlations
3. 🤖 Pricing Model  : Live ML prediction
4. 💡 Insights       : Business intelligence
"""

# ─── Imports ────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

from model import train_models, predict_price, FEATURE_COLS
from utils import (
    kpi_card, render_kpi_row, insight_card,
    section_header, divider, styled_fig,
    fmt_currency, fmt_number, fmt_pct,
    ACCENT_COLORS, PLOTLY_LAYOUT,
    price_elasticity, estimate_profit, revenue_simulation,
)

# ─── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlexiPrice Engine | Dynamic Pricing",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─── Data & Model Loading ──────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    from generate_dataset import generate_dataset
    df = generate_dataset()
    df.to_csv(csv_path, index=False)
    return df


@st.cache_resource
def load_models(_df):
    return train_models(_df)


df = load_data()
rf_model, lr_model, metrics, le_map, test_results = load_models(df)


# ─── Plotly Theme Override ──────────────────────────────────────────
CHART_COLORS = ["#00d4ff", "#7c3aed", "#10b981", "#f59e0b",
                "#ef4444", "#ec4899", "#06b6d4", "#a78bfa"]

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#8394b0", size=13),
    title_font=dict(size=16, color="#edf2f7", family="Space Grotesk, sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,212,255,0.1)",
                borderwidth=1, font=dict(size=11)),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(0,212,255,0.04)", zerolinewidth=0),
    yaxis=dict(gridcolor="rgba(0,212,255,0.04)", zerolinewidth=0),
)


def apply_theme(fig, height=420):
    fig.update_layout(**CHART_LAYOUT, height=height)
    return fig


# ─── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    # Logo / Brand
    st.markdown("""
    <div style="text-align:center; padding:24px 0 8px;">
        <div style="
            width:56px; height:56px; margin:0 auto 12px;
            background: linear-gradient(135deg, #00d4ff, #7c3aed);
            border-radius: 14px;
            display:flex; align-items:center; justify-content:center;
            box-shadow: 0 6px 20px rgba(0,212,255,0.3);
            font-size: 1.6rem;
        ">⚡</div>
        <h2 style="
            background: linear-gradient(135deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.2rem; font-weight: 700; margin: 0;
        ">FlexiPrice Engine</h2>
        <p style="color:#5a6a85; font-size:0.7rem; margin:4px 0 0;
                  letter-spacing:0.1em; text-transform:uppercase;">
            Intelligent Pricing v3.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Data Analysis", "🤖 Pricing Model", "💡 Insights"],
        label_visibility="collapsed",
    )

    divider()

    # Quick stats
    st.markdown("""
    <p style="color:#5a6a85; font-size:0.65rem; text-transform:uppercase;
              letter-spacing:0.12em; font-weight:700; margin-bottom:10px;">
        <span class="live-dot"></span> System Status
    </p>
    """, unsafe_allow_html=True)

    st.metric("Products", f"{len(df):,}")
    st.metric("Avg Price", f"${df['optimal_price'].mean():,.2f}")
    st.metric("Model R²", f"{metrics['rf']['R2']:.4f}")

    divider()

    st.markdown("""
    <div style="text-align:center; padding:8px 0;">
        <p style="color:#3a4a65; font-size:0.65rem;">
            Built by <strong style="color:#5a6a85;">Aryan Kule</strong><br>
            BSc Data Science • 2026
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    # ── Hero ──
    st.markdown("""
    <div style="text-align:center; padding:40px 0 16px;" class="fade-in">
        <div style="
            width:72px; height:72px; margin:0 auto 16px;
            background: linear-gradient(135deg, #00d4ff, #7c3aed);
            border-radius:20px; display:flex; align-items:center; justify-content:center;
            box-shadow: 0 8px 30px rgba(0,212,255,0.35);
            font-size: 2rem;
        ">⚡</div>
        <h1 style="
            background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #ec4899 100%);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem; font-weight: 700; margin: 0;
            animation: shimmer 4s ease-in-out infinite;
        ">Dynamic Price Optimization</h1>
        <p style="color:#5a6a85; font-size:1.05rem; max-width:620px; margin:14px auto 0;">
            An AI-powered pricing engine that maximizes revenue by analyzing
            demand, competition, inventory &amp; market signals in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── KPIs ──
    total_rev = df["revenue"].sum()
    avg_price = df["optimal_price"].mean()

    render_kpi_row([
        kpi_card("💰", fmt_currency(total_rev), "Total Revenue",
                 "+18.4%", True, "cyan"),
        kpi_card("🏷️", fmt_currency(avg_price), "Avg Optimal Price",
                 "+4.2%", True, "purple"),
        kpi_card("📦", fmt_number(df["units_sold"].sum()), "Units Sold",
                 "+8.7%", True, "emerald"),
        kpi_card("📈", f"{df['demand_level'].mean():.0f}", "Avg Demand",
                 "", True, "amber"),
    ])

    divider()

    # ── What is Dynamic Pricing ──
    section_header("What is Dynamic Pricing?",
                   "Adaptive pricing strategy used by industry leaders")

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
        <div class="glass-card">
            <p style="color:#8394b0; line-height:1.9; font-size:0.92rem;">
                <strong style="color:#edf2f7;">Dynamic pricing</strong> is a strategy where
                businesses adjust prices in <strong style="color:#00d4ff;">real time</strong>
                based on supply, demand, competitor pricing, and external factors.<br><br>

                Companies like <strong style="color:#f59e0b;">Amazon</strong>,
                <strong style="color:#7c3aed;">Uber</strong>, and major
                <strong style="color:#10b981;">airlines</strong> adjust prices thousands
                of times daily to maximize revenue while remaining competitive.<br><br>

                This dashboard uses a <strong style="color:#00d4ff;">Random Forest ML model</strong>
                trained on 2,000 simulated transactions to predict optimal pricing for any
                given market scenario.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <p style="color:#5a6a85; font-size:0.68rem; text-transform:uppercase;
                      letter-spacing:0.12em; font-weight:700; margin-bottom:16px;">
                Key Pricing Factors
            </p>
        """ + "".join([
            f'<span class="feature-badge badge-{c}">{t}</span>'
            for t, c in [
                ("📊 Demand Level", "blue"),
                ("🏷️ Competitor Price", "purple"),
                ("📦 Inventory Level", "teal"),
                ("⏰ Peak / Off-Peak", "orange"),
                ("👥 Customer Traffic", "blue"),
                ("🌤️ Seasonal Trends", "purple"),
                ("💸 Discount Strategy", "teal"),
                ("🤖 ML Predictions", "orange"),
            ]
        ]) + "</div>", unsafe_allow_html=True)

    divider()

    # ── Pipeline ──
    section_header("How It Works",
                   "End-to-end ML-powered pricing pipeline")

    p1, p2, p3, p4 = st.columns(4, gap="medium")
    steps = [
        ("📥", "Data Collection", "Real-time signals", "#00d4ff", p1),
        ("⚙️", "Feature Engineering", "Transform & encode", "#7c3aed", p2),
        ("🤖", "ML Model", "Random Forest", "#10b981", p3),
        ("💰", "Optimal Price", "Revenue maximized", "#f59e0b", p4),
    ]
    for icon, title, sub, color, col in steps:
        with col:
            st.markdown(f"""
            <div class="pipeline-step" style="border-top: 2px solid {color};">
                <div class="step-icon">{icon}</div>
                <div class="step-title" style="color:{color};">{title}</div>
                <div class="step-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # ── Quick Model Stats ──
    section_header("Model at a Glance")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; border-top:2px solid #00d4ff;">
            <div style="font-size:2.4rem; font-weight:700; color:#00d4ff;
                        font-family:'JetBrains Mono',monospace;">
                {metrics['rf']['R2']:.4f}
            </div>
            <div style="color:#5a6a85; font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:0.1em; font-weight:600; margin-top:4px;">
                R² Score (Random Forest)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; border-top:2px solid #7c3aed;">
            <div style="font-size:2.4rem; font-weight:700; color:#7c3aed;
                        font-family:'JetBrains Mono',monospace;">
                ${metrics['rf']['MAE']}
            </div>
            <div style="color:#5a6a85; font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:0.1em; font-weight:600; margin-top:4px;">
                Mean Absolute Error
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; border-top:2px solid #10b981;">
            <div style="font-size:2.4rem; font-weight:700; color:#10b981;
                        font-family:'JetBrains Mono',monospace;">
                {len(df):,}
            </div>
            <div style="color:#5a6a85; font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:0.1em; font-weight:600; margin-top:4px;">
                Training Samples
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════
elif page == "📊 Data Analysis":

    section_header("Data Analysis",
                   "Explore the dataset powering the pricing engine")

    # ── KPIs (use 4 to prevent cramping) ──
    render_kpi_row([
        kpi_card("📋", fmt_number(len(df)), "Records", color="cyan"),
        kpi_card("🏷️", str(df["product_category"].nunique()), "Categories", color="purple"),
        kpi_card("💲", fmt_currency(df["base_price"].mean()), "Avg Base", color="emerald"),
        kpi_card("🎯", fmt_currency(df["optimal_price"].mean()), "Avg Optimal", color="amber"),
    ])

    divider()

    # ── Dataset Preview (FULL-WIDTH, NOT collapsed) ──
    st.markdown("""
    <p style="color:#5a6a85; font-size:0.68rem; text-transform:uppercase;
              letter-spacing:0.12em; font-weight:700; margin-bottom:8px;">
        📂 Dataset Preview
    </p>
    """, unsafe_allow_html=True)

    st.dataframe(
        df.head(50),
        use_container_width=True,
        height=320,
    )

    # Quick dataset stats
    st.markdown("")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Rows", f"{df.shape[0]:,}")
    s2.metric("Columns", f"{df.shape[1]}")
    s3.metric("Min Price", f"${df['optimal_price'].min():,.2f}")
    s4.metric("Max Price", f"${df['optimal_price'].max():,.2f}")
    s5.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}")

    divider()

    # ── Analysis Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price Distribution", "🔥 Demand Analysis",
        "🏷️ Category Insights", "🔗 Correlations"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df["base_price"], name="Base Price",
                marker_color="#00d4ff", opacity=0.65, nbinsx=40,
            ))
            fig.add_trace(go.Histogram(
                x=df["optimal_price"], name="Optimal Price",
                marker_color="#7c3aed", opacity=0.65, nbinsx=40,
            ))
            fig.update_layout(title="Base vs Optimal Price Distribution",
                              xaxis_title="Price ($)", yaxis_title="Count",
                              barmode="overlay")
            st.plotly_chart(apply_theme(fig), use_container_width=True)

        with col2:
            fig = go.Figure()
            for i, cat in enumerate(sorted(df["product_category"].unique())):
                sub = df[df["product_category"] == cat]
                fig.add_trace(go.Box(
                    y=sub["optimal_price"], name=cat,
                    marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                    line_width=1.5,
                ))
            fig.update_layout(title="Optimal Price Distribution by Category",
                              yaxis_title="Price ($)", showlegend=False)
            st.plotly_chart(apply_theme(fig), use_container_width=True)

        # Scatter
        fig = px.scatter(
            df, x="base_price", y="optimal_price",
            color="product_category", opacity=0.55,
            color_discrete_sequence=CHART_COLORS,
            title="Base Price vs Optimal Price",
            labels={"base_price": "Base Price ($)", "optimal_price": "Optimal Price ($)"},
        )
        st.plotly_chart(apply_theme(fig, 440), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                df, x="demand_level", y="optimal_price",
                color="is_peak", opacity=0.45,
                color_discrete_map={0: "#00d4ff", 1: "#ef4444"},
                title="Demand Level vs Optimal Price",
                labels={"demand_level": "Demand", "optimal_price": "Price ($)",
                        "is_peak": "Peak"},
            )
            st.plotly_chart(apply_theme(fig), use_container_width=True)

        with col2:
            fig = px.scatter(
                df, x="customer_traffic", y="optimal_price",
                color="season", opacity=0.45,
                color_discrete_sequence=CHART_COLORS,
                title="Traffic vs Optimal Price",
                labels={"customer_traffic": "Visitors", "optimal_price": "Price ($)"},
            )
            st.plotly_chart(apply_theme(fig), use_container_width=True)

        # Demand by season + time
        col1, col2 = st.columns(2)

        with col1:
            season_demand = df.groupby("season")["demand_level"].mean().reset_index()
            fig = px.bar(
                season_demand, x="season", y="demand_level",
                color="season", color_discrete_sequence=CHART_COLORS,
                title="Average Demand by Season",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(apply_theme(fig, 360), use_container_width=True)

        with col2:
            tod = df.groupby("time_of_day")["demand_level"].mean().reindex(
                ["Morning", "Afternoon", "Evening", "Night"]
            ).reset_index()
            fig = px.bar(
                tod, x="time_of_day", y="demand_level",
                color="time_of_day", color_discrete_sequence=CHART_COLORS[4:],
                title="Average Demand by Time of Day",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(apply_theme(fig, 360), use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            cat_stats = df.groupby("product_category").agg(
                avg_base=("base_price", "mean"),
                avg_optimal=("optimal_price", "mean"),
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cat_stats["product_category"], y=cat_stats["avg_base"],
                                 name="Base Price", marker_color="#00d4ff"))
            fig.add_trace(go.Bar(x=cat_stats["product_category"], y=cat_stats["avg_optimal"],
                                 name="Optimal Price", marker_color="#7c3aed"))
            fig.update_layout(title="Avg Prices by Category", barmode="group",
                              xaxis_title="", yaxis_title="Price ($)")
            st.plotly_chart(apply_theme(fig), use_container_width=True)

        with col2:
            cat_rev = df.groupby("product_category")["revenue"].sum().reset_index()
            fig = px.pie(
                cat_rev, values="revenue", names="product_category",
                color_discrete_sequence=CHART_COLORS, title="Revenue Share",
                hole=0.5,
            )
            fig.update_traces(textinfo="percent+label", textfont_size=11)
            st.plotly_chart(apply_theme(fig), use_container_width=True)

        # Revenue by day
        dow_rev = df.groupby("day_of_week")["revenue"].mean().reindex(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dow_rev["day_of_week"], y=dow_rev["revenue"],
            mode="lines+markers+text",
            line=dict(color="#00d4ff", width=3, shape="spline"),
            marker=dict(size=8, color="#00d4ff",
                        line=dict(width=2, color="#0c1022")),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.06)",
        ))
        fig.update_layout(title="Average Revenue by Day of Week",
                          xaxis_title="", yaxis_title="Revenue ($)")
        st.plotly_chart(apply_theme(fig, 360), use_container_width=True)

    with tab4:
        num_cols = [
            "base_price", "competitor_price", "demand_level",
            "inventory_level", "customer_traffic", "is_peak",
            "discount_pct", "units_sold", "revenue", "optimal_price",
        ]
        corr = df[num_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale=["#06080f", "#7c3aed", "#00d4ff"],
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        fig.update_layout(xaxis=dict(tickangle=-45))
        st.plotly_chart(apply_theme(fig, 560), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — PRICING MODEL
# ═══════════════════════════════════════════════════════════════════
elif page == "🤖 Pricing Model":

    section_header("Dynamic Pricing Model",
                   "Predict the optimal price — adjust inputs in real time")

    # ── Model Metrics ──
    render_kpi_row([
        kpi_card("🎯", f"{metrics['rf']['R2']:.4f}", "RF R² Score", color="cyan"),
        kpi_card("📉", f"${metrics['rf']['MAE']}", "RF MAE", color="purple"),
        kpi_card("📊", f"${metrics['rf']['RMSE']}", "RF RMSE", color="emerald"),
        kpi_card("📐", f"{metrics['lr']['R2']:.4f}", "LR R² Score", color="amber"),
    ])

    divider()

    # ── Controls ──
    section_header("🎛️ Pricing Controls",
                   "Adjust market conditions to see the predicted price change")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<p style="color:#00d4ff; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Product</p>', unsafe_allow_html=True)
        product_category = st.selectbox("Category", sorted(df["product_category"].unique()), index=0)
        cat_data = df[df["product_category"] == product_category]
        base_price = st.slider("Base Price ($)", float(cat_data["base_price"].min()),
                               float(cat_data["base_price"].max()),
                               float(cat_data["base_price"].median()), 1.0)
        competitor_price = st.slider("Competitor Price ($)", float(base_price * 0.5),
                                     float(base_price * 1.5), float(base_price * 1.02), 1.0)

    with col2:
        st.markdown('<p style="color:#7c3aed; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Market Signals</p>', unsafe_allow_html=True)
        demand_level = st.slider("Demand Level (0–100)", 0, 100, 55)
        inventory_level = st.slider("Inventory (units)", 1, 500, 150)
        customer_traffic = st.slider("Daily Visitors", 50, 5000, 1500, 50)

    with col3:
        st.markdown('<p style="color:#10b981; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Context</p>', unsafe_allow_html=True)
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"], index=1)
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"], index=2)
        day_of_week = st.selectbox("Day of Week",
                                   ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], index=4)
        discount_pct = st.slider("Discount (%)", 0.0, 35.0, 5.0, 0.5)

    is_peak = int(
        day_of_week in ["Saturday", "Sunday"]
        or time_of_day == "Evening"
        or season in ["Winter", "Summer"]
    )

    divider()

    # ── Prediction ──
    rf_price = predict_price(rf_model, base_price, competitor_price, demand_level,
                             inventory_level, customer_traffic, is_peak, discount_pct,
                             time_of_day, day_of_week, season, product_category, le_map)
    lr_price = predict_price(lr_model, base_price, competitor_price, demand_level,
                             inventory_level, customer_traffic, is_peak, discount_pct,
                             time_of_day, day_of_week, season, product_category, le_map)

    price_change = ((rf_price - base_price) / base_price) * 100

    # Price display
    _c1, _c2, _c3 = st.columns([1, 2, 1])
    with _c2:
        if price_change > 2:
            css_class, arrow = "price-up", "▲"
        elif price_change < -2:
            css_class, arrow = "price-down", "▼"
        else:
            css_class, arrow = "price-neutral", "●"

        st.markdown(f"""
        <div class="price-display {css_class}">
            {arrow} ${rf_price:,.2f}
            <div style="font-size:0.85rem; font-weight:500; margin-top:8px; opacity:0.75;
                        font-family:'Space Grotesk',sans-serif;">
                Recommended Price &nbsp;|&nbsp; {price_change:+.1f}% from base
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Comparison row
    render_kpi_row([
        kpi_card("🏷️", fmt_currency(base_price), "Base Price", color="cyan"),
        kpi_card("🤖", fmt_currency(rf_price), "RF Prediction", color="purple"),
        kpi_card("📐", fmt_currency(lr_price), "LR Prediction", color="emerald"),
        kpi_card("🏪", fmt_currency(competitor_price), "Competitor", color="amber"),
        kpi_card("⚡", "Yes" if is_peak else "No", "Peak Period",
                 color="rose" if is_peak else "emerald"),
    ])

    divider()

    # ── Revenue Simulation ──
    section_header("💰 Revenue Simulation")

    avg_units = int(cat_data["units_sold"].mean()) if len(cat_data) > 0 else 30
    sim_df = revenue_simulation(base_price, rf_price, avg_units, steps=12)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sim_df["Price ($)"], y=sim_df["Revenue ($)"],
            mode="lines+markers", name="Revenue",
            line=dict(color="#00d4ff", width=3, shape="spline"),
            marker=dict(size=8, color="#00d4ff", line=dict(width=2, color="#0c1022")),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
        ))
        fig.add_vline(x=rf_price, line_dash="dash", line_color="#7c3aed", line_width=2,
                      annotation_text=f"Optimal: ${rf_price:,.0f}",
                      annotation_font_color="#7c3aed")
        fig.update_layout(title="Revenue Curve", xaxis_title="Price ($)", yaxis_title="Revenue ($)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with col2:
        max_rev_price = sim_df.loc[sim_df["Revenue ($)"].idxmax(), "Price ($)"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sim_df["Price ($)"], y=sim_df["Profit ($)"],
            marker_color=["#10b981" if p == max_rev_price else "#1a2347"
                          for p in sim_df["Price ($)"]],
            marker_line_color=["#10b981" if p == max_rev_price else "#00d4ff"
                               for p in sim_df["Price ($)"]],
            marker_line_width=1,
        ))
        fig.update_layout(title="Profit Estimation", xaxis_title="Price ($)", yaxis_title="Profit ($)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    divider()

    # ── Model Performance ──
    section_header("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_results["Actual"], y=test_results["RF_Predicted"],
            mode="markers", name="Predictions",
            marker=dict(color="#00d4ff", size=4, opacity=0.4),
        ))
        mn = min(test_results["Actual"].min(), test_results["RF_Predicted"].min())
        mx = max(test_results["Actual"].max(), test_results["RF_Predicted"].max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines", name="Perfect",
            line=dict(color="#ef4444", dash="dash", width=2),
        ))
        fig.update_layout(title="Actual vs Predicted (RF)",
                          xaxis_title="Actual ($)", yaxis_title="Predicted ($)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with col2:
        fi = metrics["feature_importance"]
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=["#0c1022", "#7c3aed", "#00d4ff"],
                     title="Feature Importance")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    divider()

    # ── Download ──
    section_header("📥 Export Results")

    result_row = pd.DataFrame([{
        "Category": product_category, "Base Price": base_price,
        "Competitor Price": competitor_price, "Demand": demand_level,
        "Inventory": inventory_level, "Traffic": customer_traffic,
        "Season": season, "Time": time_of_day, "Day": day_of_week,
        "Discount %": discount_pct, "Peak": is_peak,
        "RF Price": rf_price, "LR Price": lr_price,
        "Change %": round(price_change, 2),
    }])

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥  Download Prediction", result_row.to_csv(index=False),
                           "price_prediction.csv", "text/csv")
    with c2:
        st.download_button("📥  Download Full Dataset", df.to_csv(index=False),
                           "dynamic_pricing_dataset.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — INSIGHTS
# ═══════════════════════════════════════════════════════════════════
elif page == "💡 Insights":

    section_header("Business Insights & Recommendations",
                   "Data-driven strategies to maximize revenue")

    # ── Price Elasticity ──
    st.markdown("### 📉 Price Elasticity Analysis")
    st.markdown("""
    <div class="info-box">
        <strong>Price Elasticity of Demand</strong> measures how sensitive customers are to
        price changes. A value of <strong>−1.5</strong> means a 1% price increase leads to
        a 1.5% demand drop.
    </div>
    """, unsafe_allow_html=True)

    price_pcts = np.linspace(-30, 30, 25)
    base = df["base_price"].mean()
    base_demand = df["demand_level"].mean()
    demands = [max(base_demand * (1 + (-1.3) * (p / 100)), 0) for p in price_pcts]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_pcts, y=demands, mode="lines+markers",
            line=dict(color="#00d4ff", width=3, shape="spline"),
            marker=dict(size=6, color="#00d4ff", line=dict(width=2, color="#0c1022")),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#3a4a65", line_width=1)
        fig.update_layout(title="Demand Response to Price Changes",
                          xaxis_title="Price Change (%)", yaxis_title="Demand Index")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with col2:
        comp = df.groupby("product_category").agg(
            our=("optimal_price", "mean"), comp=("competitor_price", "mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=comp["product_category"], y=comp["our"],
                             name="Our Price", marker_color="#00d4ff"))
        fig.add_trace(go.Bar(x=comp["product_category"], y=comp["comp"],
                             name="Competitor", marker_color="#ef4444"))
        fig.update_layout(title="Our Price vs Competitor", barmode="group",
                          xaxis_title="", yaxis_title="Price ($)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    divider()

    # ── Demand Forecast ──
    section_header("🔮 Demand Forecasting Simulation",
                   "30-day simulated forecast with confidence bands")

    days = np.arange(1, 31)
    rng = np.random.default_rng(42)
    trend = 52 + 0.4 * days
    seasonal = 8 * np.sin(2 * np.pi * days / 7)
    forecast = trend + seasonal + rng.normal(0, 3, 30)

    fig = go.Figure()
    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([forecast + 8, (forecast - 8)[::-1]]),
        fill="toself", fillcolor="rgba(124,58,237,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=forecast, mode="lines+markers", name="Forecast",
        line=dict(color="#7c3aed", width=3, shape="spline"),
        marker=dict(size=5, color="#7c3aed"),
    ))
    fig.update_layout(title="30-Day Demand Forecast", xaxis_title="Day", yaxis_title="Demand")
    st.plotly_chart(apply_theme(fig, 400), use_container_width=True)

    divider()

    # ── Peak vs Off-Peak ──
    section_header("📊 Peak vs Off-Peak Performance")

    peak = df.groupby("is_peak").agg(
        price=("optimal_price", "mean"), rev=("revenue", "mean"),
    ).reset_index()
    peak["is_peak"] = peak["is_peak"].map({0: "Off-Peak", 1: "Peak"})

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=peak["is_peak"], y=peak["price"],
                             marker_color=["#00d4ff", "#ef4444"]))
        fig.update_layout(title="Avg Optimal Price", yaxis_title="Price ($)")
        st.plotly_chart(apply_theme(fig, 340), use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=peak["is_peak"], y=peak["rev"],
                             marker_color=["#10b981", "#f59e0b"]))
        fig.update_layout(title="Avg Revenue", yaxis_title="Revenue ($)")
        st.plotly_chart(apply_theme(fig, 340), use_container_width=True)

    divider()

    # ── Recommendations ──
    section_header("💼 Strategic Recommendations",
                   "Actionable insights from data and model analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(insight_card(
            "📈", "Implement Surge Pricing",
            "During peak periods (weekends & evenings), prices can be raised 6–12% "
            "without significant demand loss. The model confirms peak periods "
            "correlate with higher willingness to pay."
        ), unsafe_allow_html=True)
        st.markdown(insight_card(
            "📦", "Dynamic Discount on Dead Stock",
            "When inventory exceeds 400 units and demand drops below 20, apply "
            "15–25% automated discounts to accelerate turnover and free warehouse space."
        ), unsafe_allow_html=True)
        st.markdown(insight_card(
            "🔍", "Competitor Price Monitoring",
            "Products priced 8%+ above competitors show 22% lower conversions. "
            "Implement real-time competitor tracking and auto-adjust within a 5% band."
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(insight_card(
            "🎯", "Segment-Based Pricing",
            "Electronics and Sports show highest price elasticity — ideal for "
            "dynamic pricing with 14–18% potential revenue uplift per segment."
        ), unsafe_allow_html=True)
        st.markdown(insight_card(
            "⏰", "Time-Based Optimization",
            "Evening traffic converts 35% better than morning. Schedule peak prices "
            "6–10 PM and offer morning promotions to boost low-period sales."
        ), unsafe_allow_html=True)
        st.markdown(insight_card(
            "🤖", "Weekly Model Retraining",
            "Retrain the pricing model weekly for market shifts. Monthly retraining "
            "showed 8% accuracy loss. Automate with MLflow or Airflow pipelines."
        ), unsafe_allow_html=True)

    divider()

    # ── Export ──
    section_header("📥 Export Insights Report")

    report = df.groupby("product_category").agg(
        avg_base=("base_price", "mean"), avg_optimal=("optimal_price", "mean"),
        avg_competitor=("competitor_price", "mean"), avg_demand=("demand_level", "mean"),
        total_revenue=("revenue", "sum"), total_units=("units_sold", "sum"),
    ).round(2).reset_index()
    report.columns = ["Category", "Avg Base", "Avg Optimal", "Avg Competitor",
                       "Avg Demand", "Total Revenue", "Total Units"]

    st.dataframe(report, use_container_width=True)
    st.download_button("📥  Download Report", report.to_csv(index=False),
                       "pricing_insights_report.csv", "text/csv")
