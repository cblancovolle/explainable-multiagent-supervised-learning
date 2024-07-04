import numpy as np
import matplotlib.pyplot as plt
from mas.head import HeadAgent
from mas.context import GaussianContextAgent
import threading
import time
import dearpygui.dearpygui as dpg
from sklearn.metrics import mean_squared_error


class GUI:
    def __init__(self) -> None:
        np.random.seed(42)

        # internal state
        self.mas = HeadAgent(
            R=np.array([0.8]),
            imprecise_th=0.01,
            bad_th=0.3,
            alpha=0.1,
            min_vol=0.1,
            context_cls=GaussianContextAgent,
            context_kwargs={"alpha": 0.3},
            step_callback=[
                self.update_agent_plot,
                self.update_targets_plot,
                self.delay,
            ],
        )

        # initialize window and widgets
        self.sleep_time = 0.01
        self.epochs = 1
        self.n = 100
        self.p_min = -3.14
        self.p_max = 3.14
        self.X_visualization = []
        self.y_visualization = []
        self.X = np.linspace(self.p_min, self.p_max, self.n)
        self.noise = np.random.normal(size=self.X.shape)
        self.noise_coeff = 0.1
        self.agent_lines = {}
        # y = X**2 + noise_coeff * noise
        self.y = np.sin(self.X) + self.noise_coeff * self.noise

        dpg.create_context()
        dpg.create_viewport()
        dpg.setup_dearpygui()

        with dpg.window(label="Debugging", tag="main"):
            dpg.add_button(
                label="Run Training",
                tag="train",
                callback=self.run_training,
            )

            # dpg.add_button(label="Build targets", ref="build_targets")
            with dpg.plot(label="plot", width=-1, height=-1):
                # optionally create legend
                dpg.add_plot_legend()

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="y_axis")

                dpg.fit_axis_data(f"x_axis")
                dpg.fit_axis_data(f"y_axis")
                dpg.add_scatter_series(
                    self.X_visualization,
                    self.y_visualization,
                    label="ref",
                    tag="ref",
                    parent="y_axis",
                )
                dpg.add_scatter_series(
                    [], [], label="ref_seen", tag="ref_seen", parent="y_axis"
                )

        # coloring target training points
        with dpg.theme() as item_theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(
                    dpg.mvStyleVar_Alpha, (255, 0, 0, 60), category=dpg.mvThemeCat_Plots
                )
        dpg.bind_item_theme("ref", item_theme)

        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        while dpg.is_dearpygui_running():
            # update plot
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def update_targets_plot(self, *args, **kwargs):
        self.X_visualization.append(kwargs["X"])
        self.y_visualization.append(kwargs["y"])
        dpg.set_value("ref", (self.X_visualization, self.y_visualization))
        dpg.fit_axis_data(f"x_axis")
        dpg.fit_axis_data(f"y_axis")

    def update_agent_plot(self, *args, **kwargs):
        n_points_to_plot = 10

        context_agents = self.mas.context_agents
        for idx, a in context_agents.items():
            # for idx, a in enumerate(context_agents.values()):
            l, h = a.validity.low[0], a.validity.high[0]
            X_viz = np.linspace(l, h, n_points_to_plot)
            y_viz = a.predict(X_viz)
            line_id = f"context_{idx}"
            if not dpg.does_item_exist(line_id):
                self.agent_lines[idx] = dpg.add_line_series(
                    X_viz, y_viz, parent="y_axis", label=line_id, tag=line_id
                )
            else:
                dpg.set_value(line_id, [X_viz, y_viz])
        # clear deleted agent's lines
        for l in self.agent_lines:
            if not l in context_agents:
                dpg.delete_item(f"context_{l}")

    def delay(self, *args, **kwargs):
        time.sleep(self.sleep_time)

    def _run_training(self):
        self.mas.fit(self.X, self.y, self.epochs, verbose=True)
        self.y = np.sin(self.X) * 2 + self.noise * self.noise_coeff

    def run_training(self, sender, app_data):
        thread = threading.Thread(target=self._run_training)
        thread.daemon = True
        thread.start()


if __name__ == "__main__":
    gui = GUI()
