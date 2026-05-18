import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from functools import partial
from shiny.express import input, ui, render
from shiny.ui import page_navbar
from sklearn.preprocessing import LabelEncoder

def model(amount_of_people, borough, month):

    intercept = 2.555903559111899
    b1 = 0.032631  #associated with borough encoded
    b2 = -0.072184 #associated with cos_month
    b3 = -0.131015 #associated with sin_month
    b4 = -0.053463 #associated with month_num
    b5 = 0.386717  #associated with KWH per person

    # 1. FIX: Wrap borough in a list [borough] so fit_transform gets a 1D array
    le = LabelEncoder()
    # Pre-populate categories so encoding works reliably for a single input
    le.fit(["Bronx", "Queens", "Manhattan", "Brooklyn", "Staten Island"])
    borough_encoded = le.transform([borough])[0]

    # 2. FIX: Convert the month string to a number using a dictionary instead of .dt.month
    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    month_num = month_mapping[month]

    # processing for cos_month
    cos_month = np.cos(2 * np.pi * month_num / 12)

    # processing for sin_month
    sin_month = np.sin(2 * np.pi * month_num / 12)
    
    cost_per_person = intercept + (b1 * borough_encoded) + (b2 * cos_month) + (b3 * sin_month) + (b4 * month_num)
    
    # Avoid errors if input.ppn() is empty or not a number yet
    try:
        people_count = int(amount_of_people)
    except (ValueError, TypeError):
        people_count = 0
        
    return cost_per_person * people_count

with ui.nav_panel("Energy Cost Estimator"):
    with ui.layout_columns():
        ui.input_text("ppn","Amount of people")
        ui.input_select("borough","Borough",{"Bronx":"Bronx","Queens":"Queens","Manhattan":"Manhattan","Brooklyn":"Brooklyn","Staten Island": "Staten Island"})
        ui.input_select("month",
                        "Month",
                        {"January":"January",
                         "February":"February",
                         "March":"March",
                         "April":"April",
                         "May":"May",
                         "June":"June",
                         "July":"July",
                         "August":"August",
                         "September":"September",
                         "October":"October",
                         "November":"November",
                         "December":"December"

                         }
                        )
    with ui.layout_columns():
        @render.ui
        def monthly_cost_value():
            calculated_cost = model(input.ppn(), input.borough(), input.month())
            return ui.value_box(
                    title="Monthly Cost",
                    value=f"${calculated_cost:,.2f}"
                    )

        @render.ui
        def estimate_cost_value():
            ppn = input.ppn()
            return ui.value_box(
                    title="Estimated Energy Usage",
                    value=ppn
                    )

    with ui.card():
        ui.card_header("Energy Usage Plot")

        @render.plot
        def energy_plot():
           
            y  = np.random.default_rng().integers(low=0, high=100, size=100) 
            x = np.arange(100)

            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title("Electric Usage This Month")
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature")
            return fig


