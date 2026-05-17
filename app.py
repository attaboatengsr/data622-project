import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from functools import partial
from shiny.express import input, ui, render
from shiny.ui import page_navbar


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
            return ui.value_box(
                    title="Monthly Cost",
                    value=input.borough()
                    )

        @render.ui
        def estimate_cost_value():
            return ui.value_box(
                    title="Estimated Energy Cost",
                    value=input.ppn()
                    )
        #ui.value_box(title="Monthly Cost", value=input.borough())
        #ui.value_box(title="Energy Usage", value="350 kWh", theme="bg-primary")

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
            ax.set_ylabel("kWh")
            return fig


