"""
utils.py
--------
Utility / helper functions for the Dynamic Price Optimizer dashboard.
Includes KPI card generators, formatting helpers, chart templates,
and business-logic calculators.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Union


# ── Formatting Helpers ──────────────────────────────────────────────

def fmt_currency(value: float) -> str:
    """Return a compact USD string that fits KPI cards."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"${value / 1_000:.0f}K"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.2f}"


def fmt_number(value: float) -> str:
    """Format a large number with K/M suffixes."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    """Format as percentage string."""
    return f"{value:+.1f}%"


# ── KPI Card (HTML) ─────────────────────────────────────────────────

def kpi_card(icon: str, value: str, label: str,
             delta: str = "", delta_positive: bool = True,
             color: str = "blue") -> str:
    """
    Return an HTML snippet for a premium KPI card.

    Parameters
    ----------
    icon   : Emoji or icon string
    value  : Main metric value (pre-formatted)
    label  : Short label below the value
    delta  : Optional delta string (e.g. "+12.4 %")
    delta_positive : Controls green/red colouring
    color  : One of blue | purple | teal | orange | rose | emerald
    """
    delta_class = "positive" if delta_positive else "negative"
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card {color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """


def render_kpi_row(cards: list[str]):
    """Render a row of KPI cards inside equal Streamlit columns."""
    cols = st.columns(len(cards))
    for col, html in zip(cols, cards):
        col.markdown(html, unsafe_allow_html=True)


# ── Insight Card (HTML) ─────────────────────────────────────────────

def insight_card(icon: str, title: str, text: str) -> str:
    """Return HTML for an insight/recommendation card."""
    return f"""
    <div class="insight-card">
        <div class="insight-icon">{icon}</div>
        <div class="insight-title">{title}</div>
        <div class="insight-text">{text}</div>
    </div>
    """


# ── Chart Theme ─────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    title_font=dict(size=18, color="#f1f5f9"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(148,163,184,0.15)",
        borderwidth=1,
        font=dict(size=12),
    ),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
)

ACCENT_COLORS = [
    "#4f8df7", "#a855f7", "#2dd4bf", "#f59e0b",
    "#f43f5e", "#10b981", "#6366f1", "#ec4899",
]


def styled_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply the standard dashboard theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig


# ── Business Calculators ────────────────────────────────────────────

def price_elasticity(demand_old: float, demand_new: float,
                     price_old: float, price_new: float) -> float:
    """
    Point price elasticity of demand.
    A value of -1.5 means a 1 % price increase leads to 1.5 % demand drop.
    """
    if price_old == 0 or demand_old == 0:
        return 0.0
    pct_demand = (demand_new - demand_old) / demand_old
    pct_price  = (price_new - price_old)  / price_old
    if pct_price == 0:
        return 0.0
    return round(pct_demand / pct_price, 3)


def estimate_profit(revenue: float, cost_ratio: float = 0.6) -> float:
    """Simple profit estimate (revenue minus estimated costs)."""
    return revenue * (1 - cost_ratio)


def revenue_simulation(base_price: float, optimal_price: float,
                       units_sold: int, steps: int = 10) -> pd.DataFrame:
    """
    Simulate revenue across a range of price points from
    80 % of base_price to 120 % of base_price.
    """
    prices = np.linspace(base_price * 0.7, base_price * 1.4, steps)
    # Simple demand model: units drop linearly as price rises
    elasticity = -1.2
    base_demand = units_sold
    demands = base_demand * (1 + elasticity * ((prices - base_price) / base_price))
    demands = np.clip(demands, 0, None)
    revenues = prices * demands
    profits  = revenues * 0.4  # 40 % margin

    return pd.DataFrame({
        "Price ($)": np.round(prices, 2),
        "Est. Demand": np.round(demands, 0).astype(int),
        "Revenue ($)": np.round(revenues, 2),
        "Profit ($)": np.round(profits, 2),
    })


# ── Section header helper ──────────────────────────────────────────

def section_header(title: str, subtitle: str = ""):
    """Render a gradient header + optional subtitle."""
    st.markdown(f'<h2 class="gradient-header">{title}</h2>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="gradient-subheader">{subtitle}</p>', unsafe_allow_html=True)


def divider():
    """Render a subtle gradient divider."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
