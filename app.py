import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shiny.express import input, ui, render
from sklearn.preprocessing import LabelEncoder


# -----------------------------
#  MODEL FUNCTION
# -----------------------------
def model(borough, month, amount_of_people=0):
    le = LabelEncoder()
    le.fit(['BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND'])

    beta_0 = 3.6832530510462056
    beta = np.array([-0.06246448, -0.07376973, -0.02036989])
    means = np.array([-8.63506797e-18, -4.68760833e-17, 2.00000000e+00])
    std_devs = np.array([0.70710678, 0.70710678, 1.41421356])

    if not (0 <= amount_of_people <= 10):
        raise ValueError("Household population input must be strictly between 0 and 10 people.")
    if not (1 <= month <= 12):
        raise ValueError("Month input must be an integer between 1 and 12.")

    if amount_of_people == 0:
        return 0.0

    borough_encoded = le.transform([borough.upper().strip()])[0]

    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    raw_features = np.array([sin_month, cos_month, float(borough_encoded)])
    standardized_features = (raw_features - means) / std_devs

    predicted_log_y = beta_0 + np.dot(beta, standardized_features)
    base_cost_per_person = np.expm1(predicted_log_y)
    base_cost_per_person = max(0.0, base_cost_per_person)

    return base_cost_per_person * amount_of_people


# -----------------------------
#  MONTH MAP
# -----------------------------
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


# -----------------------------
#  SHINY UI
# -----------------------------
with ui.nav_panel("Energy Cost Estimator"):

    # INPUTS
    with ui.layout_columns():
        ui.input_text("ppn", "Amount of people")
        ui.input_select("borough", "Borough", {
            "Bronx": "Bronx",
            "Queens": "Queens",
            "Manhattan": "Manhattan",
            "Brooklyn": "Brooklyn",
            "Staten Island": "Staten Island",
        })
        ui.input_select("month", "Month", {m: m for m in MONTH_MAP})

    # VALUE BOXES
    with ui.layout_columns():

        @render.ui
        def monthly_cost_value():
            try:
                people = int(input.ppn())
            except:
                return ui.value_box(title="Monthly Cost", value="")

            month_num = MONTH_MAP[input.month()]
            cost = model(input.borough(), month_num, people)

            return ui.value_box(
                title="Monthly Cost",
                value=f"${cost:,.2f}"
            )

        @render.ui
        def estimate_cost_value():
            try:
                people = int(input.ppn())
            except:
                return ui.value_box(title="Estimated Energy Usage", value="")

            cost_rate = 0.16402
            month_num = MONTH_MAP[input.month()]
            cost = model(input.borough(), month_num, people)
            usage = cost / cost_rate

            return ui.value_box(
                title="Estimated Energy Usage",
                value=f"{usage:,.0f} kWh"
            )

    # -----------------------------
    #  MONTHLY USAGE PLOT
    # -----------------------------
    with ui.card():
        ui.card_header("Estimated Monthly Energy Usage")

        @render.plot
        def monthly_usage_plot():
            # --- Convert people input safely ---
            try:
                people = int(input.ppn())
            except Exception:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "Enter a valid number of people", ha="center")
                ax.axis("off")
                return fig

            borough = input.borough()
            month_num = MONTH_MAP[input.month()]

            # --- Compute estimated monthly usage ---
            cost_rate = 0.16402  # dollars per kWh
            monthly_cost = model(borough, month_num, people)   # ← FIXED
            monthly_usage_kwh = monthly_cost / cost_rate

            # --- Create daily usage curve ---
            days = np.arange(1, 31)
            rng = np.random.default_rng(seed=month_num + people)

            daily_usage = (monthly_usage_kwh / 30) * (1 + rng.normal(0, 0.05, size=30))

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(days, daily_usage, marker="o", linewidth=2, color="tab:blue")

            ax.set_title(f"Estimated Energy Usage for {input.month()} ({borough})")
            ax.set_xlabel("Day of Month")
            ax.set_ylabel("kWh per Day")
            ax.grid(True, alpha=0.3)

            return fig

