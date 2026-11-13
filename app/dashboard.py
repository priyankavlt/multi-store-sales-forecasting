"""Streamlit dashboard: sales history, model quality and forecast explorer.

Run with ``streamlit run app/dashboard.py`` after the pipeline has produced
features, a model and a forecast.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from salesforecast.config import load_config

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL, GRID, INK2 = "#c3c2b7", "#e1e0d9", "#52514e"
MONTH0 = pd.Timestamp("2013-01-01")

st.set_page_config(page_title="Multi-store Sales Forecasting", layout="wide")
cfg = load_config()


def month_label(block: int) -> str:
    return (MONTH0 + pd.DateOffset(months=int(block))).strftime("%b %Y")


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13),
        hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=NEUTRAL)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    p = cfg.processed_dir
    return {
        name: pd.read_parquet(p / f"{name}.parquet")
        for name in ("history", "shops", "items", "categories")
    }


@st.cache_data
def load_forecast() -> pd.DataFrame | None:
    path = cfg.predictions_dir / "forecast.csv"
    return pd.read_csv(path) if path.exists() else None


def load_report(name: str) -> dict | None:
    path = cfg.reports_dir / name
    return json.loads(path.read_text()) if path.exists() else None


if not (cfg.processed_dir / "history.parquet").exists():
    st.error("No processed data found. Run `make pipeline` (or `salesforecast all`) first.")
    st.stop()

tables = load_tables()
history, shops, items, cats = (
    tables["history"],
    tables["shops"],
    tables["items"],
    tables["categories"],
)
forecast = load_forecast()
metrics = load_report("metrics.json")
summary = load_report("forecast_summary.json")

st.title("Multi-store Sales Forecasting")
st.caption(
    "Monthly unit sales for every shop/item pair across a 60-store electronics & media retail chain."
)

tab_overview, tab_model, tab_explore = st.tabs(
    ["Overview", "Model performance", "Forecast explorer"]
)

# ----------------------------------------------------------------------------- overview
with tab_overview:
    monthly = history.groupby("date_block_num", as_index=False)[["item_cnt", "revenue"]].sum()
    last = int(monthly["date_block_num"].max())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Units sold, last month",
        f"{monthly.iloc[-1]['item_cnt']:,.0f}",
        f"{month_label(last)}",
        delta_color="off",
    )
    c2.metric(
        "Active shops, last month",
        f"{history.loc[history.date_block_num == last, 'shop_id'].nunique()}",
    )
    c3.metric(
        "Items sold, last month",
        f"{history.loc[history.date_block_num == last, 'item_id'].nunique():,}",
    )
    if summary:
        c4.metric(
            "Forecast units, next month*",
            f"{summary['total_units']:,.0f}",
            f"{month_label(summary['forecast_month'])}",
            delta_color="off",
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[month_label(m) for m in monthly["date_block_num"]],
            y=monthly["item_cnt"],
            mode="lines+markers",
            line=dict(color=BLUE, width=2),
            marker=dict(size=8),
            name="Units sold",
            hovertemplate="%{x}<br>%{y:,.0f} units<extra></extra>",
        )
    )
    fig.update_layout(title="Total units sold per month", hovermode="x unified", showlegend=False)
    st.plotly_chart(style(fig), width="stretch")
    if summary:
        st.caption(
            "*The forecast covers only the shop/item pairs in the forecast set, and each pair is clipped to 20 units, "
            "so it is not directly comparable with the chain-wide history above."
        )

    left, right = st.columns(2)
    recent = history[history.date_block_num > last - 12]
    by_shop = (
        recent.groupby("shop_id", as_index=False)["revenue"]
        .sum()
        .merge(shops, on="shop_id")
        .nlargest(15, "revenue")
        .sort_values("revenue")
    )
    fig = go.Figure(
        go.Bar(
            x=by_shop["revenue"] / 1e6,
            y=by_shop["shop_name"],
            orientation="h",
            marker_color=BLUE,
            hovertemplate="%{y}<br>%{x:,.1f}M revenue<extra></extra>",
        )
    )
    fig.update_layout(title="Top 15 shops by revenue, last 12 months (millions)")
    left.plotly_chart(style(fig, 480), width="stretch")

    by_family = (
        recent.merge(items[["item_id", "item_category_id"]], on="item_id")
        .merge(cats, on="item_category_id")
        .groupby("cat_family_en", as_index=False)["item_cnt"]
        .sum()
        .nlargest(12, "item_cnt")
        .sort_values("item_cnt")
    )
    fig = go.Figure(
        go.Bar(
            x=by_family["item_cnt"],
            y=by_family["cat_family_en"],
            orientation="h",
            marker_color=BLUE,
            hovertemplate="%{y}<br>%{x:,.0f} units<extra></extra>",
        )
    )
    fig.update_layout(title="Units by category family, last 12 months")
    right.plotly_chart(style(fig, 480), width="stretch")

# ----------------------------------------------------------------------------- model
with tab_model:
    if not metrics:
        st.info("No trained model yet. Run `salesforecast train`.")
    else:
        m = metrics["metrics"]
        lgbm = m["LightGBM"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Hold-out RMSE (LightGBM)", f"{lgbm['rmse']:.4f}")
        c2.metric("Improvement vs best baseline", f"{metrics['lift_vs_best_baseline']:.1%}")
        c3.metric("Hold-out month", month_label(metrics["validation_month"]))

        table = pd.DataFrame(m).T[["rmse", "mae", "bias", "r2"]].sort_values("rmse")
        st.subheader("Hold-out month comparison")
        st.dataframe(table.style.format("{:.4f}"), width="stretch")

        left, right = st.columns(2)
        imp = pd.DataFrame(metrics["top_features"]).sort_values("gain_share")
        fig = go.Figure(
            go.Bar(
                x=imp["gain_share"] * 100,
                y=imp["feature"],
                orientation="h",
                marker_color=BLUE,
                hovertemplate="%{y}<br>%{x:.1f}% of gain<extra></extra>",
            )
        )
        fig.update_layout(title="Top features (share of split gain, %)")
        left.plotly_chart(style(fig, 560), width="stretch")

        seg = pd.DataFrame(metrics["rmse_by_category_family"]).sort_values("rmse")
        fig = go.Figure(
            go.Bar(
                x=seg["rmse"],
                y=seg["segment"],
                orientation="h",
                marker_color=BLUE,
                customdata=seg["rows"],
                hovertemplate="%{y}<br>RMSE %{x:.3f}<br>%{customdata:,} rows<extra></extra>",
            )
        )
        fig.update_layout(title="Hold-out RMSE by category family")
        right.plotly_chart(style(fig, 560), width="stretch")

        backtest = load_report("backtest.json")
        if backtest:
            bt = pd.DataFrame(backtest["months"]).set_index("month")
            fig = go.Figure()
            for col, color in zip(bt.columns, [BLUE, ORANGE, AQUA], strict=False):
                fig.add_trace(
                    go.Scatter(
                        x=[month_label(i) for i in bt.index],
                        y=bt[col],
                        name=col,
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=8),
                        hovertemplate=f"{col}<br>%{{x}}: RMSE %{{y:.4f}}<extra></extra>",
                    )
                )
            fig.update_layout(title="Rolling-origin backtest (RMSE)", hovermode="x unified")
            fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(style(fig), width="stretch")

        drift = load_report("drift.json")
        if drift:
            st.subheader("Feature drift: forecast month vs last observed month")
            t = drift["thresholds"]
            st.caption(
                f"{drift['n_alert']} alert, {drift['n_warning']} warning. "
                f"Row-level features use the Population Stability Index "
                f"(warning ≥ {t['psi_warning']}, alert ≥ {t['psi_alert']}); group-level aggregates "
                f"use the relative change of their mean (warning ≥ {t['mean_shift_warning']:.0%}, "
                f"alert ≥ {t['mean_shift_alert']:.0%})."
            )
            left, right = st.columns(2)
            left.markdown("**Row-level features (PSI)**")
            left.dataframe(
                pd.DataFrame(drift["features"]).head(15), width="stretch", hide_index=True
            )
            right.markdown("**Group-level aggregates (mean shift)**")
            right.dataframe(
                pd.DataFrame(drift["group_level_features"]).head(15),
                width="stretch",
                hide_index=True,
            )

# ----------------------------------------------------------------------------- explorer
with tab_explore:
    if forecast is None:
        st.info("No forecast yet. Run `salesforecast predict`.")
    else:
        fc = forecast.merge(
            items[["item_id", "item_name", "item_category_id"]], on="item_id", how="left"
        ).merge(cats[["item_category_id", "item_category_name"]], on="item_category_id", how="left")
        shop_opts = shops[shops.shop_id.isin(fc.shop_id.unique())].sort_values("shop_name")
        shop_name = st.selectbox("Shop", shop_opts["shop_name"])
        shop_id = int(shop_opts.loc[shop_opts.shop_name == shop_name, "shop_id"].iloc[0])
        shop_fc = fc[fc.shop_id == shop_id].sort_values("item_cnt_month", ascending=False)

        c1, c2 = st.columns([2, 3])
        c1.metric("Forecast units for this shop", f"{shop_fc['item_cnt_month'].sum():,.0f}")
        top = shop_fc.head(25)
        c1.dataframe(
            top[["item_id", "item_name", "item_category_name", "item_cnt_month"]].rename(
                columns={"item_cnt_month": "forecast_units", "item_category_name": "category"}
            ),
            hide_index=True,
            width="stretch",
            height=520,
        )

        item_label = c2.selectbox(
            "Item (top 25 for this shop)",
            [f"{r.item_id} · {r.item_name}" for r in top.itertuples()],
        )
        item_id = int(item_label.split(" · ")[0])
        hist = history[(history.shop_id == shop_id) & (history.item_id == item_id)]
        months = list(
            range(int(history.date_block_num.min()), int(history.date_block_num.max()) + 1)
        )
        series = hist.set_index("date_block_num")["item_cnt"].reindex(months, fill_value=0)
        fc_value = float(shop_fc.loc[shop_fc.item_id == item_id, "item_cnt_month"].iloc[0])
        fc_month = int(summary["forecast_month"]) if summary else months[-1] + 1

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[month_label(m) for m in months],
                y=series.values,
                name="Actual units",
                mode="lines+markers",
                line=dict(color=BLUE, width=2),
                marker=dict(size=7),
                hovertemplate="%{x}<br>%{y:,.0f} units<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[month_label(months[-1]), month_label(fc_month)],
                y=[series.values[-1], fc_value],
                name="Forecast",
                mode="lines+markers",
                line=dict(color=ORANGE, width=2, dash="dot"),
                marker=dict(size=9),
                hovertemplate="%{x}<br>%{y:,.2f} units<extra></extra>",
            )
        )
        fig.update_layout(title="Monthly units in this shop", hovermode="x unified")
        c2.plotly_chart(style(fig, 470), width="stretch")
        c2.caption("Actuals shown unclipped; the model target is clipped to 0-20 units per month.")
