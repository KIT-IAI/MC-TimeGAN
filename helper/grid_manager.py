"""grid_manager.py"""

import os
import shutil
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandapower as pp
import pandapower.plotting as plot
from pandapower.control import ConstControl
from pandapower.timeseries.output_writer import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.timeseries.data_sources.frame_data import DFData

class GridManager:
    def __init__(self):
        pass

    def create_grid(self, name: str, lv: float, hv: float, l_lv: float, inputfeed: dict, mode: str = 'default') -> pp.auxiliary.pandapowerNet:
        """Create simple grid topologies that accommodate the provided load and PV profiles."""
        assert mode in ['default', 'row', 'parallel']
        grid = pp.create_empty_network(name=name)
        bus_hv, bus_lv1, bus_lv2 = pp.create_bus(grid, hv), pp.create_bus(grid, lv), pp.create_bus(grid, lv)
        pp.create_line(grid, bus_lv1, bus_lv2, l_lv, std_type='NAYY 4x150 SE')
        pp.create_ext_grid(grid, bus_hv)
        pp.create_transformer(grid, bus_hv, bus_lv1, std_type='0.25 MVA 10/0.4 kV')

        for _ in range(1, max(len(value) for value in inputfeed.values())):
            pp.create_bus(grid, lv)
            if mode == 'row':
                pp.create_line(grid, grid['bus'].index[-2], grid['bus'].index[-1], l_lv, std_type='NAYY 4x150 SE')
            elif mode == 'parallel':
                pp.create_line(grid, bus_lv1, grid['bus'].index[-1], l_lv, std_type='NAYY 4x150 SE')

        if mode == 'default':
            for key, items in inputfeed.items():
                for i, _ in enumerate(items):
                    if key == 'load':
                        pp.create_load(grid, len(grid['bus'])-1, p_mw=1, name='load_' + str(i))
                    elif key == 'sgen':
                        pp.create_sgen(grid, len(grid['bus'])-1, p_mw=1, name='sgen_' + str(i))
        else:
            for key, items in inputfeed.items():
                for i, _ in enumerate(items):
                    if key == 'load':
                        pp.create_load(grid, i+2, p_mw=1, name='load_' + str(i+2))
                    elif key == 'sgen':
                        pp.create_sgen(grid, i+2, p_mw=1, name='sgen_' + str(i+2))

        return grid

    def correct_bus_names(self, grid: pp.auxiliary.pandapowerNet) -> pp.auxiliary.pandapowerNet:
        for i, (_, name) in enumerate(grid.bus[['name']].itertuples()):
            if not name.endswith(str(i)):
                grid.bus.loc[i, ['name']] = name[:name.rfind(' ')] + ' ' + str(i)
        return grid

    def fix_bus_names(self, names: list) -> list:
        return [name[:name.rfind(' ')] + ' ' + str(i) if not name.endswith(str(i)) else name for i, name in enumerate(names)]

    def plot_annotated_grid(self, grid: pp.auxiliary.pandapowerNet) -> None:
        grid.bus_geodata.drop(grid.bus_geodata.index, inplace=True)
        fig, ax = plt.subplots()
        plot.simple_plot(grid, ext_grid_size=3, trafo_size=3, plot_loads=True, plot_sgens=True, load_size=3, sgen_size=3, ax=ax, show_plot=False)
        for i, x, y, _ in grid.bus_geodata.itertuples():
            ax.annotate(' ' + str(i) if grid.bus['name'][i] is None else ' ' + grid.bus['name'][i][grid.bus['name'][i].rfind(' ') + 1:], (x, y))
        plt.show()

    def datasource(self, inputfeed: dict) -> tuple:
        df, time_steps = None, None
        for _, items in inputfeed.items():
            for profile, name in items:
                new_df = pd.DataFrame(profile.to_numpy(), index=profile.index, columns=[name])
                df = pd.concat([df, new_df], axis=1) if df is not None else new_df
                time_steps = profile.index
        return DFData(df), time_steps

    def create_controllers(self, grid: pp.auxiliary.pandapowerNet, element: str, variable: str, element_index: list, data_source: DFData, profile_name: str) -> None:
        """grid | element e.g., load/sgen/... | variable e.g., p_mw/... | element_index e.g., [0] index in DF load/sgen/... | data_soure e.g., DataSource | profile_name e.g., 'load_test' name of column in DataSource"""
        ConstControl(grid, element=element, variable=variable, element_index=element_index, data_source=data_source, profile_name=profile_name)

    def create_output_writer(self, grid: pp.auxiliary.pandapowerNet, time_steps: list, output_dir: str, keys: list) -> pd.DataFrame:
        ow = OutputWriter(grid, time_steps, output_path=output_dir, output_file_type='.xlsx', log_variables=list())
        for key in keys:
            ow.log_variable('res_' + key, 'p_mw')
        ow.log_variable('res_bus', 'vm_pu')
        ow.log_variable('res_trafo', 'loading_percent')
        ow.log_variable('res_line', 'loading_percent')
        return ow

    def plot_results(self, data: pd.DataFrame, ylabel: str, title: str):
        fig, ax = plt.subplots(figsize=(15, 3))
        ax.plot(data, label=data.columns.values)
        if not ylabel.find('Voltage'):
            ax.axhline(0.9, color='r')
            ax.axhline(1.1, color='r')
        ax.legend()
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_xlim(0, data.index[-1])
        ax.set_xlabel('Time (15 min)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()
        plt.show()

    def time_series(self, inputfeed: dict, output_dir: str, grid: pp.auxiliary.pandapowerNet) -> None:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        grid.trafo.tap_pos = 1
        ds, time_steps = self.datasource(inputfeed)

        for category, items in inputfeed.items():
            for i, (_, name) in enumerate(items):
                self.create_controllers(grid=grid,
                                        element=category,
                                        variable='p_mw',
                                        element_index=[i],
                                        data_source=ds,
                                        profile_name=name)

        ow = self.create_output_writer(grid, time_steps, output_dir, inputfeed.keys())
        run_timeseries(grid, time_steps)

    def display_compact_dataframes(self, dataframes):
        compact_views = {}
        for key, dfs in dataframes.items():
            compact_list = []
            for df in dfs:
                compact_df = pd.concat([df.head(5), df.tail(5)])
                compact_list.append(compact_df)
            compact_views[key] = compact_list
        return compact_views

    def read_and_plot_xlsx_files(self, output_dir):
        subfolders = ['res_bus', 'res_line', 'res_load', 'res_sgen', 'res_trafo']
        dataframes_dict = {}  # Dictionary to store DataFrames
        fig, axs = plt.subplots(len(subfolders), 1, figsize=(5, 3 * len(subfolders)))

        for idx, folder in enumerate(subfolders):
            folder_path = os.path.join(output_dir, folder)
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.xlsx'):
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_excel(file_path, index_col=0)

                    # Store the DataFrame in the dictionary
                    if folder not in dataframes_dict:
                        dataframes_dict[folder] = []
                    dataframes_dict[folder].append(df)

                    ax = axs[idx] if len(subfolders) > 1 else axs
                    df.plot(ax=ax, title=folder, legend=True)

                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel('Value')
                    ax.grid(True)

        plt.tight_layout()
        plt.show()
        return dataframes_dict

    def extract_feeder(self, input_grid: pp.auxiliary.pandapowerNet, busses: list, keys: list) -> pp.auxiliary.pandapowerNet:
        grid = copy.deepcopy(input_grid)
        output_grid = pp.create_empty_network()
        for key in keys:
            if key == 'bus':
                grid[key] = grid[key].drop(busses)
                grid[key] = grid[key].reset_index()

                output_grid[key] = copy.deepcopy(grid[key])
            elif key == 'ext_grid':
                output_grid[key] = copy.deepcopy(grid[key])
            elif key == 'line':
                for bool_list in [bus != grid[key]['from_bus'] for bus in busses]:
                    grid[key]['from_bus'] = grid[key][['from_bus']].where(bool_list, np.nan, axis=0)
                    grid[key] = grid[key].dropna(subset='from_bus')
                grid[key] = grid[key].reset_index(drop=True)
                grid[key]['from_bus'] = grid[key]['from_bus'].replace(grid['bus']['index'].tolist(), grid['bus'].index.tolist())

                grid[key]['from_bus'] = grid[key]['from_bus'].astype(int)

                for bool_list in [bus != grid[key]['to_bus'] for bus in busses]:
                    grid[key]['to_bus'] = grid[key][['to_bus']].where(bool_list, np.nan, axis=0)
                    grid[key] = grid[key].dropna(subset='to_bus')
                grid[key] = grid[key].reset_index(drop=True)
                grid[key]['to_bus'] = grid[key]['to_bus'].replace(grid['bus']['index'].tolist(), grid['bus'].index.tolist())

                grid[key]['to_bus'] = grid[key]['to_bus'].astype(int)

                output_grid[key] = copy.deepcopy(grid[key])
            elif key == 'trafo':
                grid[key]['lv_bus'] = grid[key]['lv_bus'].replace(grid['bus']['index'].tolist(), grid['bus'].index.tolist())
                grid[key]['lv_bus'] = grid[key]['lv_bus'].astype(int)

                output_grid[key] = copy.deepcopy(grid[key])
            elif key == 'bus_geodata':
                grid[key] = grid[key].drop(busses)
                grid[key] = grid[key].reset_index(drop=True)

                output_grid[key] = copy.deepcopy(grid[key])
            else:
                for bool_list in [bus != grid[key]['bus'] for bus in busses]:
                    grid[key]['bus'] = grid[key][['bus']].where(bool_list, np.nan, axis=0)
                    grid[key] = grid[key].dropna(subset='bus')
                grid[key] = grid[key].reset_index(drop=True)
                grid[key]['bus'] = grid[key]['bus'].replace(grid['bus']['index'].tolist(), grid['bus'].index.tolist())
                grid[key]['bus'] = grid[key]['bus'].astype(int)

                output_grid[key] = copy.deepcopy(grid[key])

        output_grid['bus'] = output_grid['bus'].drop(['index'], axis=1)

        self.plot_annotated_grid(output_grid)

        return output_grid

    def load_data_from_directory(self, directory: str, data_type: str, file_name: str) -> pd.DataFrame:
        """
        Loads data from an Excel file located within a specified directory and modifies column names.

        Parameters:
        directory (str): The base directory where the data is stored.
        data_type (str): The type of data, used to construct the file path and prefix column names.
        file_name (str): The name of the Excel file (without extension) to be loaded.

        Returns:
        pd.DataFrame: A DataFrame with renamed columns.
        """
        path = f"{directory}/{data_type}/{file_name}.xlsx"
        print(path)
        df_loaded = pd.read_excel(path, index_col=0)
        new_column_names = [f"{data_type}_{i}" for i, _ in enumerate(df_loaded.columns)]
        return df_loaded.set_axis(new_column_names, axis=1)

    def export_data_to_directory(self, *data: tuple, path: str) -> None:
        """
        Exports multiple DataFrames to a specified directory as CSV files.

        Parameters:
        path (str): The directory where the CSV files will be saved. Must be specified.
        *data (tuple): Tuples containing DataFrames and their corresponding file names.

        Raises:
        Exception: If no path is provided.
        """
        if path is None:
            raise Exception('A path must be provided!')

        # Ensure the directory exists, or create it if it does not
        if not os.path.exists(path):
            os.makedirs(path)

        # Export the data as CSV files
        for dataset, name in data:
            path_csv = os.path.join(path, name)
            dataset.to_csv(path_csv, index=False)
            print(f'Export completed for {name} at {path_csv}')