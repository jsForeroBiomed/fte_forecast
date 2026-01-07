import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from shiny import App, ui, render


PURPLE = "#6F2C91"
LIGHT_PURPLE = "#B48AD6"
DARK_PURPLE = "#4B1E6D"
GRAY = "#6E6E73"
LIGHT_GRAY = "#E5E5EA"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "font.weight": "medium",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "medium",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": GRAY,
    "axes.labelcolor": GRAY,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "grid.color": LIGHT_GRAY,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6
})


TRAIN_DATA_PATH = "data/3_internal_train_data.csv"
VAL_DATA_PATH = "data/3_internal_validation_data.csv"
TEST_DATA_PATH = "data/.test/test_data.csv"

FINAL_MODEL_LABEL = "ARIMA(0,1,0)"
DATA_FREQUENCY_LABEL = "Monthly"

TRAIN_RESULTS_PATH = "results/train_results.csv"
VAL_RESULTS_PATH = "results/validation_results.csv"


train_df = pd.read_csv(TRAIN_DATA_PATH)
val_df = pd.read_csv(VAL_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

df = pd.concat([train_df, val_df, test_df])

for d in [train_df, val_df, test_df, df]:
    d["Date"] = pd.to_datetime(d["Date"])
    d.sort_values("Date", inplace=True)
    d.reset_index(drop=True, inplace=True)

train_results = pd.read_csv(TRAIN_RESULTS_PATH)
val_results = pd.read_csv(VAL_RESULTS_PATH)

train_results["Model"] = (
    train_results["order_p"].astype(str) + "," +
    train_results["order_d"].astype(str) + "," +
    train_results["order_q"].astype(str)
)
val_results["Model"] = (
    val_results["order_p"].astype(str) + "," +
    val_results["order_d"].astype(str) + "," +
    val_results["order_q"].astype(str)
)

model_comparison = (
    pd.merge(train_results, val_results, on="Model", how="outer")
    .sort_values("rmse")
)

model_comparison.rename(columns={"RMSE_insample": "RMSE train", "rmse": "RMSE validation"}, inplace=True)

rolling_val = pd.read_csv("data/4_rolling_validation_df.csv")
rolling_test = pd.read_csv("data/4_rolling_test_df.csv")
backtesting_metrics = pd.read_csv("results/backtest_metrics.csv")

rolling_val["Date"] = pd.to_datetime(rolling_val["Date"])
rolling_test["Date"] = pd.to_datetime(rolling_test["Date"])

backtesting_metrics["dataset"] = (
    backtesting_metrics["dataset"].str.lower().str.strip()
)

last_date = test_df["Date"].max()
last_value = float(test_df.loc[test_df["Date"] == last_date, "Total_FTE"].iloc[0])


app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&display=swap"
        ),
        ui.tags.style("""
    body {
        font-family: 'Montserrat', sans-serif;
        font-weight: 500;
    }

    h2 {
        font-weight: 700;
    }

    h3, h4 {
        font-weight: 600;
    }

    p, label, .card-body {
        font-weight: 500;
    }



    .nav-tabs .nav-link {
        color: #6E6E73;
        font-weight: 500;
        border: none;
        background-color: transparent;
    }

    .nav-tabs .nav-link:hover {
        color: #6F2C91;              
        background-color: #F3EDF8;
        border: none;
    }

    .nav-tabs .nav-link.active {
        color: #6F2C91;           
        background-color: #FFFFFF;
        border: none;
        border-bottom: 3px solid #6F2C91;
        font-weight: 600;
    }

     .nav-tabs {
         border-bottom: 1px solid #E5E5EA;
     }

     
     table {
         width: 100%;
         border-collapse: collapse;
     }

     table th {
         text-align: left;
         padding: 8px 12px;
         border-bottom: 2px solid #6F2C91;
         font-weight: 600;
         color: #6F2C91;
     }

     table td {
         text-align: left;
         padding: 8px 12px;
         border-bottom: 1px solid #E5E5EA;
         font-weight: 500;
     }

     table tr:hover {
         background-color: #F3EDF8;
     }
 """)
     ),

    ui.h2("FTE Forecast Dashboard"),

    ui.navset_tab(

        ui.nav_panel(
            "Overview",
            ui.layout_columns(
                ui.card(
                    ui.h4("KPIs"),
                    ui.p(f"Last available date: {last_date.date()}"),
                    ui.p(f"Last observed FTE: {last_value:,.0f}"),
                    ui.p(f"Data frequency: {DATA_FREQUENCY_LABEL}"),
                    ui.p(f"Final model: {FINAL_MODEL_LABEL}")
                ),
                ui.card(
                    ui.h4("Historical FTE"),
                    ui.output_plot("plot_full_series", height="380px")
                ),
                col_widths=[4, 8]
            )
        ),

        ui.nav_panel(
            "Model comparison",
            ui.layout_columns(
                ui.card(
                    ui.h4("Train vs Validation metrics"),
                    ui.output_table("comparison_table")
                ),
                ui.card(
                    ui.h4("RMSE by model"),
                    ui.output_plot("rmse_bar_plot", height="350px")
                ),
                col_widths=[6, 6]
            )
        ),

        ui.nav_panel(
            "Backtesting",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_radio_buttons(
                        "bt_dataset",
                        "Dataset",
                        choices={"validation": "Validation", "test": "Test"},
                        selected="validation"
                    )
                ),
                ui.card(
                    ui.h4("Rolling backtest"),
                    ui.output_plot("backtest_plot", height="380px")
                ),
                ui.card(
                    ui.h4("Backtest metrics"),
                    ui.output_table("backtest_metrics_table")
                )
            )
        ),

        ui.nav_panel(
            "Forecast",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "fc_horizon",
                        "Forecast horizon",
                        choices={"6": "6 months", "12": "12 months", "24": "24 months"},
                        selected="12"
                    ),
                    ui.input_select(
                        "fc_ci",
                        "Confidence level",
                        choices={"0.8": "80%", "0.95": "95%"},
                        selected="0.95"
                    )
                ),
                ui.card(
                    ui.h4("Forecast with uncertainty"),
                    ui.output_plot("forecast_plot", height="400px")
                ),
                ui.card(
                    ui.output_text("forecast_range_summary")
                )
            )
        )
    )
)


def server(input, output, session):

    @output
    @render.plot
    
    def plot_full_series():
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            df["Date"],
            df["Total_FTE"],
            color=DARK_PURPLE,
            linewidth=2,
            label="Historical FTE"
        )

        train_split = train_df["Date"].max()
        ax.axvline(
            train_split,
            linestyle=":",
            linewidth=1.5,
            color=GRAY,
            label="Train / Validation split"
        )

        if len(test_df) > 0:
            val_split = val_df["Date"].max()
            ax.axvline(
                val_split,
                linestyle=":",
                linewidth=1.5,
                color=GRAY,
                label="Validation / Test split"
            )

        ax.set_title("Historical FTE (monthly)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total FTE")
        ax.legend()
        ax.grid(True)

        return fig


    @output
    @render.table
    def comparison_table():
        result = model_comparison[
            ["Model", "RMSE train", "RMSE validation"]
        ].copy()
        # Asegurar que los valores numéricos están bien formateados
        result["RMSE train"] = pd.to_numeric(result["RMSE train"], errors="coerce").round(2)
        result["RMSE validation"] = pd.to_numeric(result["RMSE validation"], errors="coerce").round(2)
        # Reemplazar NaN con 0 para evitar errores de JavaScript
        result = result.fillna(0)
        return result

    @output
    @render.plot
    def rmse_bar_plot():
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(model_comparison))
        ax.bar(x, model_comparison["RMSE train"], color=LIGHT_PURPLE, label="Train")
        ax.bar(x, model_comparison["RMSE validation"], bottom=model_comparison["RMSE train"],
               color=PURPLE, label="Validation")
        ax.set_xticks(x)
        ax.set_xticklabels(model_comparison["Model"], rotation=45)
        ax.legend()
        
        return fig

    @output
    @render.plot
    def backtest_plot():
        data = rolling_val if input.bt_dataset() == "validation" else rolling_test
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["Date"], data["Actual"], color=DARK_PURPLE, label="Actual")
        ax.plot(data["Date"], data["Predicted"], color=PURPLE, linestyle="--", label="Forecast")
        ax.legend()
        ax.set_xlabel("Year")
        ax.set_ylabel("Total FTE")
        ax.grid(True)
        return fig

    @output
    @render.table
    def backtest_metrics_table():
        filtered = backtesting_metrics[
            backtesting_metrics["dataset"] == input.bt_dataset()
        ]
        
        if len(filtered) == 0:
            # Retornar DataFrame vacío con las columnas correctas si no hay datos
            return pd.DataFrame(columns=["model", "RMSE", "MAE"])
        
        result = filtered[["model", "RMSE", "MAE"]].copy()
        # Asegurar que los valores numéricos están bien formateados
        result["RMSE"] = pd.to_numeric(result["RMSE"], errors="coerce").round(1)
        result["MAE"] = pd.to_numeric(result["MAE"], errors="coerce").round(1)
        # Reemplazar NaN con 0 y asegurar que model sea string
        result["model"] = result["model"].astype(str)
        result = result.fillna(0)
        return result

    @output
    @render.plot

    def forecast_plot():
        horizon = int(input.fc_horizon())
        ci = float(input.fc_ci())

        model = ARIMA(df["Total_FTE"], order=(0, 1, 0)).fit()
        res = model.get_forecast(horizon)

        mean = res.predicted_mean
        ci_df = res.conf_int(alpha=1 - ci)

        last_date = df["Date"].max()
        forecast_dates = pd.date_range(
            start=last_date,
            periods=horizon + 1,
            freq="MS"
        )

        fig, ax = plt.subplots(figsize=(10, 4))

        # Historical
        ax.plot(
            df["Date"],
            df["Total_FTE"],
            color=DARK_PURPLE,
            linewidth=2,
            label="Historical"
        )

        # Connect last observed point with forecast
        forecast_x = forecast_dates
        forecast_y = np.concatenate(
            [[df["Total_FTE"].iloc[-1]], mean.values]
        )

        ax.plot(
            forecast_x,
            forecast_y,
            color=PURPLE,
            linewidth=2.5,
            label="Forecast"
        )

        # Confidence interval (starts from first forecast point)
        ax.fill_between(
            forecast_dates[1:],
            ci_df.iloc[:, 0],
            ci_df.iloc[:, 1],
            color=LIGHT_PURPLE,
            alpha=0.35,
            label=f"{int(ci*100)}% confidence interval"
        )

        ax.axvline(
            last_date,
            linestyle=":",
            color=GRAY
        )

        ax.set_title("FTE forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total FTE")
        ax.legend()
        ax.grid(True)

        return fig


    @output
    @render.text
    def forecast_range_summary():
        horizon = int(input.fc_horizon())
        ci = float(input.fc_ci())
        model = ARIMA(df["Total_FTE"], order=(0,1,0)).fit()
        res = model.get_forecast(horizon)
        ci_df = res.conf_int(alpha=1-ci)
        return (
            f"{int(ci*100)}% interval at {horizon} months:\n"
            f"{ci_df.iloc[-1,0]:,.0f} – {ci_df.iloc[-1,1]:,.0f} FTE"
        )

app = App(app_ui, server)
