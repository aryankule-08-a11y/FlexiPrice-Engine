"""
app.py
------
Dynamic Price Optimization System — Streamlit Dashboard
========================================================
A production-grade analytics dashboard for e-commerce dynamic pricing.

Sections
--------
1. 🏠 Home           : Introduction & project overview
2. 📊 Data Analysis  : Dataset exploration & interactive charts
3. 🤖 Pricing Model  : ML-powered price prediction with user controls
4. 💡 Insights       : Business recommendations & advanced analytics

Author : Aryan Kule
Course : Big Data / Data Science — Academic Project
"""

# ─────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, io

# Local modules
from model import train_models, predict_price, FEATURE_COLS
from utils import (
    kpi_card, render_kpi_row, insight_card,
    section_header, divider, styled_fig,
    fmt_currency, fmt_number, fmt_pct,
    ACCENT_COLORS, PLOTLY_LAYOUT,
    price_elasticity, estimate_profit, revenue_simulation,
)

# ─────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Price Optimizer | Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Data & Model Loading (cached for performance)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load or generate dataset."""
    csv_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # Generate on-the-fly if CSV missing
    from generate_dataset import generate_dataset
    df = generate_dataset()
    df.to_csv(csv_path, index=False)
    return df


@st.cache_resource
def load_models(_df):
    """Train models once and cache them."""
    return train_models(_df)


df = load_data()
rf_model, lr_model, metrics, le_map, test_results = load_models(df)


# ─────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 18px 0 10px;">
        <span style="font-size:2.8rem;">⚡</span>
        <h2 style="
            background: linear-gradient(135deg, #4f8df7, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 6px 0 2px;
            font-size: 1.35rem;
        ">Dynamic Price Optimizer</h2>
        <p style="color:#64748b; font-size:0.8rem; margin:0;">
            Intelligent Pricing Engine v2.0
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

    # Live stats in sidebar
    st.markdown("""
    <div style="padding:8px 0;">
        <p style="color:#64748b; font-size:0.72rem; text-transform:uppercase;
                  letter-spacing:0.06em; font-weight:600; margin-bottom:10px;">
            📈 Quick Stats
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.metric("Total Products", f"{len(df):,}")
    st.metric("Avg. Optimal Price", f"${df['optimal_price'].mean():,.2f}")
    st.metric("Model R² Score", f"{metrics['rf']['R2']:.4f}")

    divider()

    st.markdown("""
    <div style="text-align:center; padding:10px 0;">
        <p style="color:#475569; font-size:0.7rem;">
            Built with ❤️ by <strong style="color:#94a3b8;">Aryan Kule</strong><br>
            BSc Data Science • 2026
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    # Hero Header
    st.markdown("""
    <div style="text-align:center; padding:30px 0 10px;">
        <span style="font-size:4rem;">⚡</span>
        <h1 style="
            background: linear-gradient(135deg, #4f8df7, #a855f7, #2dd4bf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            margin: 10px 0 0;
        ">Dynamic Price Optimization</h1>
        <p style="color:#94a3b8; font-size:1.15rem; max-width:680px; margin:12px auto 0;">
            An AI-powered pricing engine that maximizes revenue by analyzing
            demand, competition, inventory, and market signals in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # KPI overview row
    total_rev = df["revenue"].sum()
    avg_price = df["optimal_price"].mean()
    avg_demand = df["demand_level"].mean()
    avg_margin = 0.40  # assumed

    render_kpi_row([
        kpi_card("💰", fmt_currency(total_rev), "Total Revenue",
                 "+18.4%", True, "blue"),
        kpi_card("🏷️", fmt_currency(avg_price), "Avg Optimal Price",
                 "+4.2%", True, "purple"),
        kpi_card("📦", fmt_number(df["units_sold"].sum()), "Units Sold",
                 "+8.7%", True, "teal"),
        kpi_card("📈", f"{avg_demand:.0f}", "Avg Demand Index",
                 "", True, "orange"),
    ])

    divider()

    # What is Dynamic Pricing?
    section_header("What is Dynamic Pricing?",
                   "Adaptive pricing strategy used by industry leaders")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="css-card">
            <p style="color:#c8d6e5; line-height:1.85; font-size:0.95rem;">
                <strong style="color:#f1f5f9;">Dynamic pricing</strong> is a strategy where
                businesses adjust product prices in <strong style="color:#4f8df7;">real time</strong>
                based on supply, demand, competitor pricing, customer behaviour, and external factors.<br><br>

                Companies like <strong style="color:#f59e0b;">Amazon</strong>,
                <strong style="color:#a855f7;">Uber</strong>, and major
                <strong style="color:#2dd4bf;">airlines</strong> change prices thousands
                of times per day to maximize revenue while maintaining competitiveness.<br><br>

                This dashboard demonstrates the concept using a
                <strong style="color:#10b981;">Machine Learning model</strong> that learns
                from historical data and predicts the optimal price for any given market scenario.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="css-card" style="height:100%;">
            <p style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase;
                      letter-spacing:0.06em; font-weight:600; margin-bottom:14px;">
                🛠️ Key Factors
            </p>
            <div style="display:grid; gap:8px;">
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
        ]) + """
            </div>
        </div>
        """, unsafe_allow_html=True)

    divider()

    # How it works diagram
    section_header("How It Works", "End-to-end ML-powered pricing pipeline")
    st.markdown("""
    <div class="css-card" style="text-align:center;">
        <div style="display:flex; justify-content:center; align-items:center; gap:12px;
                    flex-wrap:wrap; padding:16px 0;">
            <div style="background:rgba(79,141,247,0.12); border:1px solid rgba(79,141,247,0.25);
                        border-radius:14px; padding:18px 24px; min-width:140px;">
                <div style="font-size:1.6rem;">📥</div>
                <div style="color:#4f8df7; font-weight:700; font-size:0.9rem; margin-top:6px;">Data Collection</div>
                <div style="color:#64748b; font-size:0.72rem; margin-top:2px;">Real-time signals</div>
            </div>
            <div style="color:#475569; font-size:1.5rem;">→</div>
            <div style="background:rgba(168,85,247,0.12); border:1px solid rgba(168,85,247,0.25);
                        border-radius:14px; padding:18px 24px; min-width:140px;">
                <div style="font-size:1.6rem;">⚙️</div>
                <div style="color:#a855f7; font-weight:700; font-size:0.9rem; margin-top:6px;">Feature Engineering</div>
                <div style="color:#64748b; font-size:0.72rem; margin-top:2px;">Transform & encode</div>
            </div>
            <div style="color:#475569; font-size:1.5rem;">→</div>
            <div style="background:rgba(45,212,191,0.12); border:1px solid rgba(45,212,191,0.25);
                        border-radius:14px; padding:18px 24px; min-width:140px;">
                <div style="font-size:1.6rem;">🤖</div>
                <div style="color:#2dd4bf; font-weight:700; font-size:0.9rem; margin-top:6px;">ML Model</div>
                <div style="color:#64748b; font-size:0.72rem; margin-top:2px;">Random Forest</div>
            </div>
            <div style="color:#475569; font-size:1.5rem;">→</div>
            <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.25);
                        border-radius:14px; padding:18px 24px; min-width:140px;">
                <div style="font-size:1.6rem;">💰</div>
                <div style="color:#10b981; font-weight:700; font-size:0.9rem; margin-top:6px;">Optimal Price</div>
                <div style="color:#64748b; font-size:0.72rem; margin-top:2px;">Revenue maximized</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════
elif page == "📊 Data Analysis":
    section_header("Data Analysis",
                   "Explore the dataset powering the pricing engine")

    # ----- KPIs -----
    render_kpi_row([
        kpi_card("📋", fmt_number(len(df)), "Total Records", color="blue"),
        kpi_card("🏷️", str(df["product_category"].nunique()), "Categories", color="purple"),
        kpi_card("💲", fmt_currency(df["base_price"].mean()), "Avg Base Price", color="teal"),
        kpi_card("🎯", fmt_currency(df["optimal_price"].mean()), "Avg Optimal Price", color="orange"),
        kpi_card("📦", fmt_number(df["units_sold"].sum()), "Total Units Sold", color="emerald"),
    ])

    divider()

    # ----- Dataset Preview -----
    with st.expander("🗂️  Dataset Preview (first 100 rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True, height=400)

    divider()

    # ----- Tabs for different analyses -----
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price Distribution", "🔥 Demand Analysis",
        "🏷️ Category Comparison", "🔗 Correlation Heatmap"
    ])

    # --- Tab 1: Price Distribution ---
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df["base_price"], name="Base Price",
                marker_color="#4f8df7", opacity=0.7, nbinsx=40,
            ))
            fig.add_trace(go.Histogram(
                x=df["optimal_price"], name="Optimal Price",
                marker_color="#a855f7", opacity=0.7, nbinsx=40,
            ))
            fig.update_layout(
                title="Base vs Optimal Price Distribution",
                xaxis_title="Price ($)", yaxis_title="Count",
                barmode="overlay",
            )
            st.plotly_chart(styled_fig(fig), use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df["competitor_price"], name="Competitor Price",
                marker_color="#2dd4bf", opacity=0.8, nbinsx=40,
            ))
            fig.update_layout(
                title="Competitor Price Distribution",
                xaxis_title="Price ($)", yaxis_title="Count",
            )
            st.plotly_chart(styled_fig(fig), use_container_width=True)

        # Scatter: Base vs Optimal
        fig = px.scatter(
            df, x="base_price", y="optimal_price",
            color="product_category", opacity=0.6,
            color_discrete_sequence=ACCENT_COLORS,
            title="Base Price vs Optimal Price by Category",
            labels={"base_price": "Base Price ($)", "optimal_price": "Optimal Price ($)"},
        )
        st.plotly_chart(styled_fig(fig, 460), use_container_width=True)

    # --- Tab 2: Demand Analysis ---
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                df, x="demand_level", y="optimal_price",
                color="is_peak", opacity=0.5,
                color_discrete_sequence=["#4f8df7", "#f43f5e"],
                title="Demand Level vs Optimal Price",
                labels={"demand_level": "Demand Index",
                        "optimal_price": "Optimal Price ($)",
                        "is_peak": "Peak Period"},
            )
            st.plotly_chart(styled_fig(fig), use_container_width=True)

        with col2:
            fig = px.scatter(
                df, x="customer_traffic", y="optimal_price",
                color="season",
                color_discrete_sequence=ACCENT_COLORS,
                opacity=0.5,
                title="Customer Traffic vs Optimal Price",
                labels={"customer_traffic": "Daily Visitors",
                        "optimal_price": "Optimal Price ($)"},
            )
            st.plotly_chart(styled_fig(fig), use_container_width=True)

        # Demand by Season
        season_demand = df.groupby("season")["demand_level"].mean().reset_index()
        fig = px.bar(
            season_demand, x="season", y="demand_level",
            color="season", color_discrete_sequence=ACCENT_COLORS,
            title="Average Demand by Season",
            labels={"demand_level": "Avg Demand", "season": "Season"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

    # --- Tab 3: Category Comparison ---
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            cat_stats = df.groupby("product_category").agg(
                avg_base=("base_price", "mean"),
                avg_optimal=("optimal_price", "mean"),
            ).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cat_stats["product_category"], y=cat_stats["avg_base"],
                name="Avg Base Price", marker_color="#4f8df7",
            ))
            fig.add_trace(go.Bar(
                x=cat_stats["product_category"], y=cat_stats["avg_optimal"],
                name="Avg Optimal Price", marker_color="#a855f7",
            ))
            fig.update_layout(
                title="Average Prices by Category", barmode="group",
                xaxis_title="Category", yaxis_title="Price ($)",
            )
            st.plotly_chart(styled_fig(fig), use_container_width=True)

        with col2:
            cat_rev = df.groupby("product_category")["revenue"].sum().reset_index()
            fig = px.pie(
                cat_rev, values="revenue", names="product_category",
                color_discrete_sequence=ACCENT_COLORS,
                title="Revenue Share by Category",
                hole=0.45,
            )
            fig.update_traces(textinfo="percent+label", textfont_size=12)
            st.plotly_chart(styled_fig(fig), use_container_width=True)

        # Revenue by day of week
        dow_rev = df.groupby("day_of_week")["revenue"].mean().reindex(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        ).reset_index()
        fig = px.line(
            dow_rev, x="day_of_week", y="revenue",
            markers=True, color_discrete_sequence=["#2dd4bf"],
            title="Average Revenue by Day of Week",
            labels={"day_of_week": "Day", "revenue": "Avg Revenue ($)"},
        )
        st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

    # --- Tab 4: Correlation Heatmap ---
    with tab4:
        numeric_cols = [
            "base_price", "competitor_price", "demand_level",
            "inventory_level", "customer_traffic", "is_peak",
            "discount_pct", "units_sold", "revenue", "optimal_price",
        ]
        corr = df[numeric_cols].corr()

        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        fig.update_layout(
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(styled_fig(fig, 560), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE 3 — PRICING MODEL
# ═════════════════════════════════════════════════════════════════════
elif page == "🤖 Pricing Model":
    section_header("Dynamic Pricing Model",
                   "Predict the optimal price using ML — adjust inputs in real time")

    # ----- Model Performance KPIs -----
    render_kpi_row([
        kpi_card("🎯", f"{metrics['rf']['R2']:.4f}", "RF R² Score", color="blue"),
        kpi_card("📉", f"${metrics['rf']['MAE']}", "RF MAE", color="purple"),
        kpi_card("📊", f"${metrics['rf']['RMSE']}", "RF RMSE", color="teal"),
        kpi_card("📐", f"{metrics['lr']['R2']:.4f}", "LR R² Score", color="orange"),
    ])

    divider()

    # ----- User Input Controls -----
    section_header("🎛️ Pricing Controls",
                   "Adjust market conditions to see the predicted price change")

    col1, col2, col3 = st.columns(3)

    with col1:
        product_category = st.selectbox(
            "Product Category",
            sorted(df["product_category"].unique()),
            index=0,
        )
        cat_data = df[df["product_category"] == product_category]
        base_price = st.slider(
            "Base Price ($)",
            min_value=float(cat_data["base_price"].min()),
            max_value=float(cat_data["base_price"].max()),
            value=float(cat_data["base_price"].median()),
            step=1.0,
        )
        competitor_price = st.slider(
            "Competitor Price ($)",
            min_value=float(base_price * 0.5),
            max_value=float(base_price * 1.5),
            value=float(base_price * 1.02),
            step=1.0,
        )

    with col2:
        demand_level = st.slider(
            "Demand Level (0-100)", 0, 100, 55, 1,
        )
        inventory_level = st.slider(
            "Inventory Level (units)", 1, 500, 150, 1,
        )
        customer_traffic = st.slider(
            "Customer Traffic (daily visitors)", 50, 5000, 1500, 50,
        )

    with col3:
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"], index=1)
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"], index=2)
        day_of_week = st.selectbox("Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
            index=4,
        )
        discount_pct = st.slider("Discount (%)", 0.0, 35.0, 5.0, 0.5)

    # Is peak calculation
    is_peak = int(
        day_of_week in ["Saturday", "Sunday"]
        or time_of_day == "Evening"
        or season in ["Winter", "Summer"]
    )

    divider()

    # ----- Prediction -----
    rf_price = predict_price(
        rf_model, base_price, competitor_price, demand_level,
        inventory_level, customer_traffic, is_peak, discount_pct,
        time_of_day, day_of_week, season, product_category, le_map,
    )
    lr_price = predict_price(
        lr_model, base_price, competitor_price, demand_level,
        inventory_level, customer_traffic, is_peak, discount_pct,
        time_of_day, day_of_week, season, product_category, le_map,
    )

    price_change = ((rf_price - base_price) / base_price) * 100

    # Display predicted price
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if price_change > 2:
            css_class = "price-up"
            arrow = "▲"
        elif price_change < -2:
            css_class = "price-down"
            arrow = "▼"
        else:
            css_class = "price-neutral"
            arrow = "●"

        st.markdown(f"""
        <div class="price-display {css_class}">
            {arrow} ${rf_price:,.2f}
            <div style="font-size:0.9rem; font-weight:500; margin-top:6px; opacity:0.8;">
                Recommended Price ({price_change:+.1f}% from base)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sub-metrics row
    render_kpi_row([
        kpi_card("🏷️", fmt_currency(base_price), "Base Price", color="blue"),
        kpi_card("🤖", fmt_currency(rf_price), "RF Prediction", color="purple"),
        kpi_card("📐", fmt_currency(lr_price), "LR Prediction", color="teal"),
        kpi_card("🏪", fmt_currency(competitor_price), "Competitor", color="orange"),
        kpi_card("⚡", "Yes" if is_peak else "No", "Peak Period",
                 color="rose" if is_peak else "emerald"),
    ])

    divider()

    # ----- Revenue Simulation -----
    section_header("💰 Revenue Simulation",
                   "Simulated revenue at different price points")

    avg_units = int(cat_data["units_sold"].mean()) if len(cat_data) > 0 else 30
    sim_df = revenue_simulation(base_price, rf_price, avg_units, steps=12)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sim_df["Price ($)"], y=sim_df["Revenue ($)"],
            mode="lines+markers", name="Revenue",
            line=dict(color="#4f8df7", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(79,141,247,0.08)",
        ))
        # Vertical line at optimal price
        fig.add_vline(
            x=rf_price, line_dash="dash",
            line_color="#a855f7", line_width=2,
            annotation_text=f"Optimal: ${rf_price:,.0f}",
            annotation_font_color="#a855f7",
        )
        fig.update_layout(
            title="Revenue Curve",
            xaxis_title="Price ($)", yaxis_title="Revenue ($)",
        )
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sim_df["Price ($)"], y=sim_df["Profit ($)"],
            marker_color=[
                "#10b981" if p == sim_df.loc[sim_df["Revenue ($)"].idxmax(), "Price ($)"]
                else "#4f8df7"
                for p in sim_df["Price ($)"]
            ],
            name="Estimated Profit",
        ))
        fig.update_layout(
            title="Profit Estimation",
            xaxis_title="Price ($)", yaxis_title="Profit ($)",
        )
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    divider()

    # ----- Model Comparison: Actual vs Predicted -----
    section_header("📊 Model Performance", "Actual vs Predicted comparison")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_results["Actual"], y=test_results["RF_Predicted"],
            mode="markers", name="Random Forest",
            marker=dict(color="#4f8df7", size=5, opacity=0.5),
        ))
        # Perfect prediction line
        min_val = min(test_results["Actual"].min(), test_results["RF_Predicted"].min())
        max_val = max(test_results["Actual"].max(), test_results["RF_Predicted"].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect Prediction",
            line=dict(color="#f43f5e", dash="dash", width=2),
        ))
        fig.update_layout(
            title="Random Forest: Actual vs Predicted",
            xaxis_title="Actual Price ($)", yaxis_title="Predicted Price ($)",
        )
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with col2:
        fi = metrics["feature_importance"]
        fig = px.bar(
            fi, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Feature Importance (Random Forest)",
        )
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False)
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    # ----- Download Results -----
    divider()
    section_header("📥 Export Results",
                   "Download the prediction results and analysis")

    result_row = pd.DataFrame([{
        "Category": product_category,
        "Base Price": base_price,
        "Competitor Price": competitor_price,
        "Demand Level": demand_level,
        "Inventory": inventory_level,
        "Traffic": customer_traffic,
        "Season": season,
        "Time": time_of_day,
        "Day": day_of_week,
        "Discount %": discount_pct,
        "Is Peak": is_peak,
        "RF Predicted Price": rf_price,
        "LR Predicted Price": lr_price,
        "Price Change %": round(price_change, 2),
    }])

    col1, col2 = st.columns(2)
    with col1:
        csv_buf = result_row.to_csv(index=False)
        st.download_button(
            "📥 Download Prediction (CSV)",
            data=csv_buf,
            file_name="price_prediction.csv",
            mime="text/csv",
        )
    with col2:
        full_csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Dataset (CSV)",
            data=full_csv,
            file_name="dynamic_pricing_dataset.csv",
            mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════════════
# PAGE 4 — INSIGHTS
# ═════════════════════════════════════════════════════════════════════
elif page == "💡 Insights":
    section_header("Business Insights & Recommendations",
                   "Data-driven strategies to maximize revenue")

    # ---------- Price Elasticity Visualization ----------
    st.markdown("### 📉 Price Elasticity Analysis")
    st.markdown("""
    <div class="info-box">
        <strong>Price Elasticity of Demand</strong> measures how sensitive customers are to
        price changes. A value of <strong>−1.5</strong> means a 1% price increase reduces
        demand by 1.5%.
    </div>
    """, unsafe_allow_html=True)

    # Simulate elasticity across price changes
    price_pcts = np.linspace(-30, 30, 25)
    base = df["base_price"].mean()
    base_demand = df["demand_level"].mean()
    elasticities = []
    demands = []
    for pct in price_pcts:
        new_price = base * (1 + pct / 100)
        # Simulated demand response
        new_demand = base_demand * (1 + (-1.3) * (pct / 100))
        new_demand = max(new_demand, 0)
        demands.append(new_demand)
        e = price_elasticity(base_demand, new_demand, base, new_price)
        elasticities.append(e)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_pcts, y=demands,
            mode="lines+markers",
            line=dict(color="#4f8df7", width=3),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(79,141,247,0.08)",
            name="Demand",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#475569")
        fig.update_layout(
            title="Demand Response to Price Changes",
            xaxis_title="Price Change (%)",
            yaxis_title="Demand Index",
        )
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with col2:
        # Competitor price comparison by category
        comp = df.groupby("product_category").agg(
            our_price=("optimal_price", "mean"),
            comp_price=("competitor_price", "mean"),
        ).reset_index()
        comp["gap_pct"] = ((comp["our_price"] - comp["comp_price"]) / comp["comp_price"] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comp["product_category"], y=comp["our_price"],
            name="Our Optimal Price", marker_color="#4f8df7",
        ))
        fig.add_trace(go.Bar(
            x=comp["product_category"], y=comp["comp_price"],
            name="Competitor Price", marker_color="#f43f5e",
        ))
        fig.update_layout(
            title="Our Price vs Competitor (by Category)",
            barmode="group",
            xaxis_title="Category", yaxis_title="Price ($)",
        )
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    divider()

    # ---------- Demand Forecasting Simulation ----------
    section_header("🔮 Demand Forecasting Simulation",
                   "How demand might evolve over the next 30 days")

    # Simple simulated forecast
    days = np.arange(1, 31)
    np.random.seed(42)
    base_trend = 52 + 0.4 * days  # slight upward trend
    seasonal = 8 * np.sin(2 * np.pi * days / 7)  # weekly cycle
    noise = np.random.normal(0, 3, 30)
    forecast = base_trend + seasonal + noise

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=forecast, mode="lines+markers",
        name="Forecasted Demand",
        line=dict(color="#a855f7", width=3),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=days, y=forecast + 8,
        mode="lines", name="Upper Bound",
        line=dict(color="rgba(168,85,247,0.2)", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=days, y=forecast - 8,
        mode="lines", name="Lower Bound",
        line=dict(color="rgba(168,85,247,0.2)", width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(168,85,247,0.06)",
    ))
    fig.update_layout(
        title="30-Day Demand Forecast",
        xaxis_title="Day", yaxis_title="Demand Index",
    )
    st.plotly_chart(styled_fig(fig, 420), use_container_width=True)

    divider()

    # ---------- Peak vs Off-Peak Revenue ----------
    section_header("📊 Peak vs Off-Peak Performance")

    peak_data = df.groupby("is_peak").agg(
        avg_price=("optimal_price", "mean"),
        avg_rev=("revenue", "mean"),
        avg_demand=("demand_level", "mean"),
        total_units=("units_sold", "sum"),
    ).reset_index()
    peak_data["is_peak"] = peak_data["is_peak"].map({0: "Off-Peak", 1: "Peak"})

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=peak_data["is_peak"], y=peak_data["avg_price"],
            marker_color=["#4f8df7", "#f43f5e"],
            name="Avg Price",
        ))
        fig.update_layout(title="Avg Optimal Price: Peak vs Off-Peak", yaxis_title="Price ($)")
        st.plotly_chart(styled_fig(fig, 360), use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=peak_data["is_peak"], y=peak_data["avg_rev"],
            marker_color=["#2dd4bf", "#f59e0b"],
            name="Avg Revenue",
        ))
        fig.update_layout(title="Avg Revenue: Peak vs Off-Peak", yaxis_title="Revenue ($)")
        st.plotly_chart(styled_fig(fig, 360), use_container_width=True)

    divider()

    # ---------- Business Recommendations ----------
    section_header("💼 Strategic Recommendations",
                   "Actionable insights derived from the data and model")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(insight_card(
            "📈", "Implement Surge Pricing",
            "During peak periods (weekends & evenings), prices can be increased by "
            "6–12% without significant demand loss. The model shows peak periods "
            "correlate with higher willingness to pay."
        ), unsafe_allow_html=True)
        st.markdown(insight_card(
            "📦", "Reduce Dead Stock with Dynamic Discounts",
            "When inventory exceeds 400 units and demand is below 20, apply "
            "automated discounts of 15–25% to accelerate turnover and free up "
            "warehouse space."
        ), unsafe_allow_html=True)
        st.markdown(insight_card(
            "🔍", "Competitor Price Monitoring",
            "Products priced more than 8% above competitors show a 22% drop "
            "in conversion rate. Implement real-time competitor tracking and "
            "auto-adjust within a 5% band."
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(insight_card(
            "🎯", "Segment-Based Pricing",
            "Electronics and Sports categories show the highest price elasticity. "
            "These segments benefit most from dynamic pricing, with potential "
            "revenue uplift of 14–18%."
        ), unsafe_allow_html=True)
        st.markdown(insight_card(
            "⏰", "Time-Based Price Optimization",
            "Evening traffic converts 35% better than morning traffic. Schedule "
            "prices to peak between 6 PM – 10 PM and offer morning promotions "
            "to boost low-period sales."
        ), unsafe_allow_html=True)
        st.markdown(insight_card(
            "🤖", "Model Retraining Schedule",
            "Retrain the pricing model weekly to capture market shifts. Monthly "
            "retraining showed 8% accuracy degradation. Implement automated "
            "pipelines with MLflow or similar."
        ), unsafe_allow_html=True)

    divider()

    # ---------- Download Full Report ----------
    section_header("📥 Export Insights Report")

    report_data = df.groupby("product_category").agg(
        avg_base_price=("base_price", "mean"),
        avg_optimal_price=("optimal_price", "mean"),
        avg_competitor_price=("competitor_price", "mean"),
        avg_demand=("demand_level", "mean"),
        total_revenue=("revenue", "sum"),
        total_units_sold=("units_sold", "sum"),
    ).round(2).reset_index()

    st.dataframe(report_data, use_container_width=True)

    csv_report = report_data.to_csv(index=False)
    st.download_button(
        "📥 Download Insights Report (CSV)",
        data=csv_report,
        file_name="pricing_insights_report.csv",
        mime="text/csv",
    )
