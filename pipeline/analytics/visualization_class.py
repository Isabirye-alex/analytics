import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualization:
    """
    Pure visualization class (NO side effects).
    Returns matplotlib figures only.
    """


    # Retention Heatmap

    def plot_retention(self, retention_df):
        if retention_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(retention_df, annot=True, cmap="viridis", fmt=".0%", ax=ax)

        ax.set_title("Customer Retention Heatmap Over Time")
        ax.set_xlabel("Cohort Index (Months Since First Purchase)")
        ax.set_ylabel("Cohort Month")

        fig.tight_layout()
        return fig


    # Pareto Curve

    def plot_pareto(self, pareto_df):
        if pareto_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(20, 8))

        ax.plot(
            pareto_df["CumCustomerPct"],
            pareto_df["CumRevenuePct"],
            marker="o",
            label="Cumulative Revenue",
        )

        ax.axhline(0.8, linestyle="--", label="80% Revenue")

        top_cutoff = pareto_df[pareto_df["CumRevenuePct"] <= 0.8].shape[0] / len(
            pareto_df
        )

        ax.axvline(top_cutoff, linestyle="--", label="Top Customers")

        ax.set_title("Pareto Analysis of Customers")
        ax.set_xlabel("Cumulative Customer Percentage")
        ax.set_ylabel("Cumulative Revenue Percentage")

        ax.legend()
        fig.tight_layout()

        return fig
