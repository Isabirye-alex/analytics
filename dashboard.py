"""
dashboard.py

Streamlit dashboard for the Customer Analytics & Churn Intelligence pipeline.

Audience: Business stakeholders (non-technical)
Purpose:  Translate model outputs into actionable business insights

Run:
    streamlit run dashboard.py
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ingest import DataIngestor
from clean import DataCleaner
from features import FeatureEngineering
from pipeline.analytics.customer_behavior import CustomerBehavior
from pipeline.ml.dataset_builder import DatasetBuilder
from pipeline.ml.model import ChurnModel


# Page Configuration


st.set_page_config(
    page_title="Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Styling


st.markdown(
    """
    <style>
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 2px;
            border-left: 4px solid #4361ee;
        }
        .insight-box {
            background-color: #fff3cd;
            border-radius: 8px;
            padding: 2px 2px;
            border-left: 4px solid #ffc107;
            margin-bottom: 8px;
        }
        .warning-box {
            background-color: #f8d7da;
            border-radius: 8px;
            padding: 12px 16px;
            border-left: 4px solid #dc3545;
            margin-bottom: 2px;
        }
        .success-box {
            background-color: #d1e7dd;
            border-radius: 8px;
            padding: 12px 16px;
            border-left: 4px solid #198754;
            margin-bottom: 2px;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1 { color: #1a1a2e; }
        h2 { color: #16213e; }
    </style>
""",
    unsafe_allow_html=True,
)


# Data Loading — Cached so pipeline only runs once per session


@st.cache_data(show_spinner="Running analytics pipeline...")
def load_pipeline(file_path: str) -> dict:
    """
    Run the full pipeline and return all outputs.
    Cached — only reruns if file_path changes.
    """

    # Ingest
    raw_df = DataIngestor(file_path).fetch_data_from_csv()
    if isinstance(raw_df, Exception):
        raise raw_df

    # Clean
    clean_result = DataCleaner(raw_df).run_pipeline()
    cleaned_df = clean_result["cleaned_dataset"]

    # Feature engineering
    feature_df = FeatureEngineering(cleaned_df).run_pipeline()

    # Analytics
    behavior = CustomerBehavior(feature_df).run_pipeline()
    rfm = behavior["rfm"]
    clv = behavior["clv"]
    cohort = behavior["cohort"]
    pareto = behavior["pareto"]

    # ML dataset + model
    dataset = DatasetBuilder(rfm, clv, feature_df).build()
    model = ChurnModel(dataset)
    results = model.run_pipeline(threshold=0.4)

    cohort_serialisable = cohort.copy()
    cohort_serialisable.index = cohort_serialisable.index.astype(str)
    cohort_serialisable.columns = cohort_serialisable.columns.astype(str)

    return {
        "feature_df": feature_df.copy(),
        "rfm": rfm.copy(),
        "clv": clv.copy(),
        "cohort": cohort_serialisable,
        "pareto": pareto.copy(),
        "dataset": dataset.copy(),
        "churn_scores": results["churn_scores"],
        "feature_importance": results["feature_importance"],
        "metrics": results["metrics"],
        "cv_results": results["cv_results"],
    }


# Sidebar


with st.sidebar:
    st.title("Customer Analytics")
    st.markdown("---")

    file_path = st.text_input("Data file path", value="sales.csv")

    st.markdown("---")
    st.markdown("Navigation")

    page = st.radio(
        label="Go to",
        options=[
            "Overview",
            "Churn Intelligence",
            "Customer Segments",
            "Revenue Analysis",
            "Retention Heatmap",
            "Customer Lifetime Value",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Pipeline runs once per session. Refresh the page to reload data.")

# Load Data

try:
    data = load_pipeline(file_path)
except Exception as e:
    st.error(f"Pipeline failed: {e}")
    st.stop()

# Unpack
rfm = data["rfm"]
clv = data["clv"]
cohort = data["cohort"]
pareto = data["pareto"]
dataset = data["dataset"]
churn_scores = data["churn_scores"]
importance = data["feature_importance"]
metrics = data["metrics"]
cv_results = data["cv_results"]
feature_df = data["feature_df"]


# Helper — KPI Card Row


def kpi_row(items: list):
    """
    Render a row of KPI metric cards.
    items: list of (label, value, delta, delta_color) tuples
    """
    cols = st.columns(len(items))
    for col, (label, value, delta, color) in zip(cols, items):
        col.metric(label=label, value=value, delta=delta, delta_color=color)


# PAGE: Overview


if page == "Overview":

    st.title("Customer Analytics Dashboard")
    st.markdown("A summary of your customer base, revenue health, and churn risk.")
    st.markdown("---")

    # Top KPIs
    total_customers = rfm["CustomerNo"].nunique()
    total_revenue = feature_df["TotalRevenue"].sum()
    churners = churn_scores["ChurnPrediction"].sum()
    churn_rate = churners / len(churn_scores)
    avg_clv = clv["CLV"].mean()
    top_segment = rfm["Segment"].value_counts().idxmax()

    kpi_row(
        [
            ("Total Customers", f"{total_customers:,}", None, "off"),
            ("Total Revenue", f"${total_revenue:,.0f}", None, "off"),
            (
                "At-Risk Customers",
                f"{churners:,}",
                f"{churn_rate:.0%} of base",
                "inverse",
            ),
            ("Avg Customer Value", f"${avg_clv:,.0f}", None, "off"),
        ]
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Revenue over time
    with col1:
        st.subheader("Monthly Revenue Trend")
        monthly = feature_df.groupby("YearMonth")["TotalRevenue"].sum().reset_index()
        monthly["YearMonth"] = monthly["YearMonth"].astype(str)

        fig = px.line(
            monthly,
            x="YearMonth",
            y="TotalRevenue",
            markers=True,
            labels={"YearMonth": "Month", "TotalRevenue": "Revenue ($)"},
        )
        fig.update_traces(line_color="#4361ee", line_width=2)
        fig.update_layout(
            plot_bgcolor="white",
            yaxis_tickprefix="$",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Segment distribution
    with col2:
        st.subheader("Customer Segments")
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Business insight
    st.markdown(
        f"""
    <div class="insight-box">
         <strong>Key Takeaway:</strong>
        Your largest customer segment is <strong>{top_segment}</strong>.
        {churners:,} customers ({churn_rate:.0%} of your base) are predicted to leave.
        Prioritise retention spend on high-value customers first — see the
        <em>Churn Intelligence</em> tab for the specific list.
    </div>
    """,
        unsafe_allow_html=True,
    )


# PAGE: Churn Intelligence


elif page == "Churn Intelligence":

    st.title("Churn Intelligence")
    st.markdown(
        "Which customers are leaving, how confident are we, and who should you contact first?"
    )
    st.markdown("---")

    # Model confidence KPIs
    report = metrics["classification_report"]
    recall = report["1.0"]["recall"]
    precision = report["1.0"]["precision"]
    f1 = report["1.0"]["f1-score"]
    cv_f1 = cv_results["test_f1"]

    kpi_row(
        [
            (
                "Churners Detected",
                f"{recall:.0%}",
                "of all customers who will leave",
                "off",
            ),
            (
                "Alert Accuracy",
                f"{precision:.0%}",
                "of alerts are genuine at-risk",
                "off",
            ),
            (
                "Model Reliability",
                f"{cv_f1:.0%}",
                "validated across 5 test runs",
                "off",
            ),
            (
                "Decision Threshold",
                f"{metrics['threshold']}",
                "probability cutoff",
                "off",
            ),
        ]
    )

    st.markdown("---")

    col1, col2 = st.columns([1.5, 1])

    # Churn probability distribution
    with col1:
        st.subheader("Churn Risk Distribution")
        st.caption("Each bar shows how many customers fall into that risk band.")

        fig = px.histogram(
            churn_scores,
            x="ChurnProbability",
            nbins=20,
            color_discrete_sequence=["#e63946"],
            labels={"ChurnProbability": "Churn Probability", "count": "Customers"},
        )
        fig.add_vline(
            x=metrics["threshold"],
            line_dash="dash",
            line_color="#4361ee",
            annotation_text=f"Alert threshold ({metrics['threshold']})",
            annotation_position="top right",
        )
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance — plain English
    with col2:
        st.subheader("What Drives Churn?")
        st.caption("How much each factor contributes to the prediction.")

        imp = importance.copy()
        imp["Label"] = (
            imp["Feature"]
            .map(
                {
                    "Lifespan": "How long they've been a customer",
                    "RevenueTrend": "Recent spend trend",
                    "AvgGapDays": "Average gap between purchases",
                    "Frequency": "How often they buy",
                    "CLV": "Total lifetime value",
                    "ChurnThreshold": "Personal churn threshold",
                }
            )
            .fillna(imp["Feature"])
        )

        fig = px.bar(
            imp.sort_values("Importance"),
            x="Importance",
            y="Label",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            labels={"Importance": "Influence", "Label": ""},
        )
        fig.update_layout(
            plot_bgcolor="white",
            coloraxis_showscale=False,
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # High-value at-risk table
    st.subheader("🚨 Priority Retention List")
    st.caption("High-value customers predicted to leave. Contact these first.")

    clv_threshold = dataset["CLV"].quantile(0.75)
    at_risk = (
        dataset[
            (dataset["CLV"] > clv_threshold)
            & (dataset["RevenueTrend"] < 0)
            & (dataset["Churn"] == 1)
        ]
        .merge(churn_scores, on="CustomerNo", how="inner")[
            ["CustomerNo", "CLV", "RevenueTrend", "Frequency", "ChurnProbability"]
        ]
        .sort_values("ChurnProbability", ascending=False)
        .head(20)
    )

    at_risk_display = at_risk.copy()
    at_risk_display["CLV"] = at_risk_display["CLV"].map("${:,.0f}".format)
    at_risk_display["RevenueTrend"] = at_risk_display["RevenueTrend"].map(
        "${:,.0f}".format
    )
    at_risk_display["ChurnProbability"] = at_risk_display["ChurnProbability"].map(
        "{:.0%}".format
    )
    at_risk_display.columns = [
        "Customer ID",
        "Lifetime Value",
        "Revenue Trend (90d)",
        "Purchase Frequency",
        "Churn Risk",
    ]

    st.dataframe(at_risk_display, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
    <div class="warning-box">
         <strong>Action Required:</strong>
        These {len(at_risk)} customers represent your highest-value churn risk.
        Their recent spend is declining and the model is highly confident they are leaving.
        A targeted retention offer to this group has the highest possible return on investment.
    </div>
    """,
        unsafe_allow_html=True,
    )


# PAGE: Customer Segments


elif page == "Customer Segments":

    st.title("Customer Segments")
    st.markdown("How your customers are grouped by purchasing behaviour.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segment Size")
        seg = rfm["Segment"].value_counts().reset_index()
        seg.columns = ["Segment", "Customers"]

        fig = px.bar(
            seg.sort_values("Customers", ascending=True),
            x="Customers",
            y="Segment",
            orientation="h",
            color="Customers",
            color_continuous_scale="Blues",
            labels={"Customers": "Number of Customers", "Segment": ""},
        )
        fig.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average Revenue by Segment")
        seg_rev = rfm.groupby("Segment")["Monetary"].mean().reset_index()
        seg_rev.columns = ["Segment", "Avg Revenue"]
        seg_rev = seg_rev.sort_values("Avg Revenue", ascending=True)

        fig = px.bar(
            seg_rev,
            x="Avg Revenue",
            y="Segment",
            orientation="h",
            color="Avg Revenue",
            color_continuous_scale="Greens",
            labels={"Avg Revenue": "Average Revenue ($)", "Segment": ""},
        )
        fig.update_layout(
            plot_bgcolor="white",
            coloraxis_showscale=False,
            xaxis_tickprefix="$",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Segment Detail")

    seg_summary = (
        rfm.groupby("Segment")
        .agg(
            Customers=("CustomerNo", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Revenue=("Monetary", "mean"),
        )
        .reset_index()
        .sort_values("Avg_Revenue", ascending=False)
    )

    seg_summary["Avg_Recency"] = seg_summary["Avg_Recency"].map("{:.0f} days".format)
    seg_summary["Avg_Frequency"] = seg_summary["Avg_Frequency"].map(
        "{:.1f} orders".format
    )
    seg_summary["Avg_Revenue"] = seg_summary["Avg_Revenue"].map("${:,.0f}".format)
    seg_summary.columns = [
        "Segment",
        "Customers",
        "Avg Days Since Purchase",
        "Avg Orders",
        "Avg Revenue",
    ]

    st.dataframe(seg_summary, use_container_width=True, hide_index=True)


# PAGE: Revenue Analysis


elif page == "Revenue Analysis":

    st.title("Revenue Analysis")
    st.markdown("Where your revenue comes from and how concentrated it is.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pareto — Revenue Concentration")
        st.caption("Cumulative revenue vs cumulative customers")

        df = pareto.copy()

        fig = go.Figure()

        # Bars = Revenue
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["CumRevenue"],
                name="Revenue",
                marker_color="#4361ee",
                opacity=0.7,
            )
        )

        # Line = Cumulative %
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["CumRevenuePct"],
                name="Cumulative %",
                mode="lines+markers", 
                line=dict(color="#e63946", width=2),
                yaxis="y2",  
            )
        )

        # 80% reference line
        fig.add_hline(y=80, line_dash="dash", line_color="orange", yref="y2")

        # Layout with secondary axis enabled
        fig.update_layout(
            title="Pareto Chart — Revenue Concentration",
            template="plotly_white",
            yaxis=dict(title="Revenue"),
            yaxis2=dict(
                title="Cumulative %",
                side="right",
                range=[0, 110],
            ),
            legend=dict(orientation="h"),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by Country")

        country_rev = (
            feature_df.groupby("Country")["TotalRevenue"]
            .sum()
            .reset_index()
            .sort_values("TotalRevenue", ascending=False)
            .head(10)
        )

        fig = px.bar(
            country_rev.sort_values("TotalRevenue"),
            x="TotalRevenue",
            y="Country",
            orientation="h",
            color="TotalRevenue",
            color_continuous_scale="Blues",
            labels={"TotalRevenue": "Total Revenue ($)", "Country": ""},
        )
        fig.update_layout(
            plot_bgcolor="white",
            coloraxis_showscale=False,
            xaxis_tickprefix="$",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Revenue-Generating Products")

    top_products = (
        feature_df.groupby("ProductName")["TotalRevenue"]
        .sum()
        .reset_index()
        .sort_values("TotalRevenue", ascending=False)
        .head(10)
    )
    top_products["TotalRevenue"] = top_products["TotalRevenue"].map("${:,.0f}".format)
    top_products.columns = ["Product", "Total Revenue"]

    st.dataframe(top_products, use_container_width=True, hide_index=True)


# PAGE: Retention Heatmap


elif page == "Retention Heatmap":

    st.title("Retention Heatmap")
    st.markdown(
        "Shows what percentage of customers from each starting month "
        "are still purchasing in later months."
    )
    st.markdown("---")

    st.markdown(
        """
    <div class="insight-box">
        💡 <strong>How to read this:</strong>
        Each row is a group of customers who made their first purchase in that month.
        Each column shows what percentage of them came back in a later month.
        Dark colour = strong retention. Light colour = customers stopped buying.
    </div>
    """,
        unsafe_allow_html=True,
    )

    cohort_plot = cohort.copy()
    cohort_plot.index = cohort_plot.index.astype(str)
    cohort_plot.columns = cohort_plot.columns.astype(str)

    fig = px.imshow(
        cohort_plot,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(
            x="Months Since First Purchase",
            y="First Purchase Month",
            color="Retention %",
        ),
        text_auto=".0%",
    )
    fig.update_layout(
        xaxis_title="Months Since First Purchase",
        yaxis_title="First Purchase Month (Cohort)",
        coloraxis_colorbar=dict(tickformat=".0%"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Retention drop insight
    if cohort_plot.shape[1] > 1:
        month1_col = cohort_plot.iloc[:, 1] if cohort_plot.shape[1] > 1 else None
        if month1_col is not None:
            avg_m1_retention = month1_col.mean()
            st.markdown(
                f"""
            <div class="{'warning-box' if avg_m1_retention < 0.3 else 'success-box'}">
                {'⚠️' if avg_m1_retention < 0.3 else '✅'}
                <strong>Month 1 Retention:</strong>
                On average, <strong>{avg_m1_retention:.0%}</strong> of new customers
                return for a second purchase.
                {'This is low — consider an onboarding campaign targeting first-time buyers.' 
                 if avg_m1_retention < 0.3 else 
                 'This is healthy retention for a second purchase.'}
            </div>
            """,
                unsafe_allow_html=True,
            )


# PAGE: CLV Table

elif page == "Customer Lifetime Value":

    st.title("Customer Lifetime Value")
    st.markdown(
        "Which customers are worth the most over their entire relationship with you.",
        
    )
    st.markdown("---")

    # KPIs
    top_10_pct = clv["CLV"].quantile(0.9)
    bottom_50_pct = clv["CLV"].quantile(0.5)
    top_10_revenue = clv[clv["CLV"] >= top_10_pct]["CLV"].sum()
    total_clv = clv["CLV"].sum()

    kpi_row(
        [
            ("Total Customer Value", f"${total_clv:,.0f}", None, "off"),
            (
                "Top 10% Customer Value",
                f"${top_10_revenue:,.0f}",
                f"{top_10_revenue/total_clv:.0%} of total",
                "off",
            ),
            ("Median Customer Value", f"${bottom_50_pct:,.0f}", None, "off"),
            ("Top 10% Threshold", f"${top_10_pct:,.0f}", "CLV to be top tier", "off"),
        ]
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CLV Distribution")
        st.caption(
            "Most customers cluster at the lower end — a small group drives most value."
        )

        fig = px.histogram(
            clv,
            x="CLV",
            nbins=30,
            color_discrete_sequence=["#4361ee"],
            labels={"CLV": "Customer Lifetime Value ($)", "count": "Customers"},
        )
        fig.update_layout(plot_bgcolor="white", xaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("CLV vs Purchase Frequency")
        st.caption("Higher frequency customers tend to have higher lifetime value.")

        fig = px.scatter(
            clv,
            x="Frequency",
            y="CLV",
            opacity=0.5,
            color_discrete_sequence=["#4361ee"],
            labels={"Frequency": "Purchase Frequency", "CLV": "Lifetime Value ($)"},
        )
        fig.update_layout(plot_bgcolor="white", yaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 20 Customers by Lifetime Value")

    top_customers = clv.sort_values("CLV", ascending=False).head(20).copy()

    # Merge churn risk
    top_customers = top_customers.merge(
        churn_scores[["CustomerNo", "ChurnProbability", "ChurnPrediction"]],
        on="CustomerNo",
        how="left",
    )

    top_customers["CLV"] = top_customers["CLV"].map("${:,.0f}".format)
    top_customers["ChurnProbability"] = top_customers["ChurnProbability"].map(
        "{:.0%}".format
    )
    top_customers["ChurnPrediction"] = top_customers["ChurnPrediction"].map(
        {0: "✅ Retained", 1: "⚠️ At Risk"}
    )

    display_cols = [
        "CustomerNo",
        "CLV",
        "Frequency",
        "Lifespan",
        "ChurnProbability",
        "ChurnPrediction",
    ]
    display_cols = [c for c in display_cols if c in top_customers.columns]

    top_customers = top_customers[display_cols]
    top_customers.columns = [
        c.replace("CustomerNo", "Customer ID")
        .replace("ChurnProbability", "Churn Risk")
        .replace("ChurnPrediction", "Status")
        .replace("Lifespan", "Months Active")
        for c in top_customers.columns
    ]

    st.dataframe(top_customers, use_container_width=True, hide_index=True)

    at_risk_top = (top_customers["Status"] == "⚠️ At Risk").sum()
    if at_risk_top > 0:
        st.markdown(
            f"""
        <div class="warning-box">
            <strong>{at_risk_top} of your top 20 customers are predicted to churn.</strong>
            These accounts represent significant revenue risk.
            Cross-reference with the Priority Retention List in Churn Intelligence.
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class="success-box">
            <strong>All top 20 customers are predicted to be retained.</strong>
            Focus retention efforts on the mid-tier customers in the Churn Intelligence tab.
        </div>
        """,
            unsafe_allow_html=True,
        )
