from matplotlib.path import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os


class DataVisualization:
    """
    Handles visualization of customer analytics results, including:
    - Retention heatmaps
    - Pareto (80/20) customer-revenue analysis
    """

    def __init__(self, figures_dir="pipeline/figures"):
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        HERE = Path(__file__). # Adjust based on where this file is
        self.figures_dir = HERE / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Figures will be saved to: {self.figures_dir}")

    # -------------------------------
    # Retention Heatmap
    # -------------------------------
    def plot_retention(self, retention_df):
        """
        Creates a retention heatmap to visualize customer retention over time.

        Parameters
        ----------
        retention_df : pandas.DataFrame
            Cohort retention matrix where rows are cohorts and columns are periods.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        if retention_df.empty:
            self.logger.warning("Retention DataFrame is empty. Skipping heatmap.")
            return None, None

        self.logger.info("Plotting retention heatmap")

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(retention_df, annot=True, cmap="viridis", fmt=".0%", ax=ax)
        ax.set_title("Customer Retention Heatmap Over Time")
        ax.set_xlabel("Cohort Index (Months Since First Purchase)")
        ax.set_ylabel("Cohort Month")

        save_path = os.path.join(self.figures_dir, "retention_analysis.jpg")
        fig.savefig(save_path, dpi=300)
        self.logger.info(f"Retention heatmap saved to {save_path}")

        return fig, ax

    # -------------------------------
    # Pareto Curve
    # -------------------------------
    def plot_pareto(self, pareto_df):
        """
        Creates a Pareto curve to visualize the cumulative contribution
        of customers to total revenue.

        Parameters
        ----------
        pareto_df : pandas.DataFrame
            DataFrame containing cumulative revenue and cumulative customer percentage.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        if pareto_df.empty:
            self.logger.warning("Pareto DataFrame is empty. Skipping Pareto plot.")
            return None, None

        self.logger.info("Plotting Pareto curve")

        fig, ax = plt.subplots(figsize=(20, 8))

        # Plot cumulative customer % vs cumulative revenue %
        ax.plot(
            pareto_df["CumCustomerPct"],
            pareto_df["CumRevenuePct"],
            marker="o",
            label="Cumulative Revenue",
        )

        # Highlight 80% revenue line
        ax.axhline(0.8, color="red", linestyle="--", label="80% Revenue")

        # Highlight corresponding top customers
        top_customers_cutoff = pareto_df[pareto_df["CumRevenuePct"] <= 0.8].shape[
            0
        ] / len(pareto_df)
        ax.axvline(
            top_customers_cutoff, color="blue", linestyle="--", label="Top Customers"
        )

        ax.set_title("Pareto Analysis of Customers")
        ax.set_xlabel("Cumulative Customer Percentage")
        ax.set_ylabel("Cumulative Revenue Percentage")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(self.figures_dir, "pareto_curve.jpg")
        fig.savefig(save_path, dpi=300)
        self.logger.info(f"Pareto curve saved to {save_path}")

        return fig, ax
