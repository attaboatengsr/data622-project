import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from functools import partial
from shiny import render
from shiny.express import input, ui
from shiny.ui import page_navbar

#ui.page_opts(
#    title="Energy Cost Estimator",
#    page_fn=partial(page_navbar, id="page")
#)

with ui.nav_panel("Energy Cost Estimator"):
    with ui.layout_columns():
        ui.value_box(title="Monthly Cost", value="$120")
        ui.value_box(title="Energy Usage", value="350 kWh", theme="bg-primary")

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

        ui.input_file("file_upload", "Choose CSV File", accept=[".csv"])
        @render.text
        def show_info():
            return input.file_upload()

