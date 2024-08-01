"""label_processing.py"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
import simbench as sb
import seaborn as sns
import sklearn.preprocessing
from itertools import chain

class LabelProcessor:
    def __init__(self):
        pass

    def plot_profiles_and_labels(self, df_profiles: pd.DataFrame, df_labels: pd.DataFrame) -> None:
        """Visualize profile and corresponding labels in a joint plot."""
        assert df_profiles.shape == df_labels.shape, f"Profiles shape: {df_profiles.shape}, Labels shape: {df_labels.shape}"

        for (_, data), (_, labels) in zip(df_profiles.items(), df_labels.items()):
            fig, ax = plt.subplots()
            ax.plot(data, label='Profile')
            ax_twin = ax.twinx()
            ax_twin.plot(labels, 'r.', label='Labels')
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_twin.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2)
            sns.move_legend(ax, 'lower center', bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
            ax.set_xlim(-0.1, len(data)+0.1)
            ax.set_xlabel('Time (15 min)')
            ax.set_ylabel('Active Power (MW)')
            ax_twin.set_ylabel('Label')
            ax.grid(True)
            plt.show()

    def binary_labels(self, df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """Create binary ordinal labels {0,1}."""
        if threshold is not None:
            assert 0 < threshold < 1

        temp_dict = {}
        for column, data in df.items():
            zero_series = pd.Series([0 for _ in range(len(data))], dtype=int)
            labels = zero_series.mask(data > threshold * data.max(), 1) if threshold is not None else data.mask(data != 0, 1).astype(int)
            temp_dict[column] = labels.to_list()

        return pd.DataFrame(temp_dict)

    def three_sigma_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ordinal labels according to three sigma intervals above the mean value."""
        temp_dict = {}
        for column, data in df.items():
            mean = data.mean()
            sigma = data.std()
            labels = data.mask(data <= mean, 0)
            labels = labels.mask((data > mean) & (data <= mean + sigma), 1)
            labels = labels.mask((data > mean + sigma) & (data <= mean + 2 * sigma), 2)
            labels = labels.mask(data > mean + 2 * sigma, 3).astype(int)
            temp_dict[column] = labels.to_list()

        return pd.DataFrame(temp_dict)

    def five_sigma_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create labels according to three sigma intervals above and two sigma intervals below the mean value."""
        temp_dict = {}
        for column, data in df.items():
            mean = data.mean()
            sigma = data.std()
            labels = data.mask(data <= mean - sigma, 0)
            labels = labels.mask((data > mean - sigma) & (data <= mean), 1)
            labels = labels.mask((data > mean) & (data <= mean + sigma), 2)
            labels = labels.mask((data > mean + sigma) & (data <= mean + 2 * sigma), 3)
            labels = labels.mask(data > mean + 2 * sigma, 4).astype(int)
            temp_dict[column] = labels.to_list()

        return pd.DataFrame(temp_dict)

    def discretize(self, df: pd.DataFrame, bin_by: str, factor: int = 1, mean_offset: bool = True) -> pd.DataFrame:
        """Discretize time series data into equal-sized bins and retrieve bin values as ordinal labels."""
        assert bin_by in ['mean', 'std']

        temp_dict = {}
        for column, data in df.items():
            bin_value = data.mean() if bin_by == 'mean' else data.std()
            if mean_offset:
                bins = [data.mean() + factor * bin_value * i for i in range(-int(np.ceil((data.mean() - data.min()) / (factor * bin_value))),
                                                                           int(np.ceil((data.max() - data.mean()) / (factor * bin_value))) + 1)]
            else:
                bins = [factor * bin_value * i for i in range(int(np.ceil((data.max() - data.min()) / (factor * bin_value))) + 2)]

            transformer = sklearn.preprocessing.FunctionTransformer(pd.cut, kw_args={'bins': bins,
                                                                                     'labels': [i for i in range(len(bins) - 1)],
                                                                                     'retbins': False,
                                                                                     'include_lowest': True})
            temp_dict[column] = transformer.fit_transform(data).astype(int)

        return pd.DataFrame(temp_dict)

    def season_labels(self, months: tuple, no_dp: int = 96 * 7 * 4) -> pd.DataFrame:
        """Create ordinal labels for considered months. Assumes equal length of months."""
        assert len(months) == 2 and months[0] < months[-1]

        return pd.DataFrame({'season': [month for month in range(months[0], months[-1]) for _ in range(no_dp)]})

    def season_magnitude_labels(self, months: tuple, no_dp: int = 96 * 7 * 4) -> pd.DataFrame:
        """Create ordinal labels according to the expected active power magnitude of PV generation, depending on the incoming sun. Assumes equal length of months."""
        assert len(months) == 2 and months[0] < months[-1]

        return pd.DataFrame({'season': [list(chain(range(1, 6), range(6, -1, -1)))[i] for i in range(months[0], months[-1]) for _ in range(no_dp)]})

    def peak_spreading_approach(self, df: pd.DataFrame, no_neighbors: int, target_label_list: list = None) -> pd.DataFrame:
        """Set label values in adjacency of a target label to label's value. By default, the highest ordinal label value serves as target label."""
        if target_label_list is not None:
            assert len(target_label_list) == df.shape[-1]

        temp_dict = {}
        for i, (column, data) in enumerate(df.items()):
            ordinal_labels = np.unique(data.to_numpy())
            hist, bin_edges = np.histogram(data.to_numpy(), len(ordinal_labels))
            target_label = max(ordinal_labels) if target_label_list is None else target_label_list[i]

            data_ = data.copy()
            for window in data.eq(target_label).rolling(no_neighbors, min_periods=1, closed='both'):
                if window.any():
                    data_[window.index] = target_label

            hist_, _ = np.histogram(data_.to_numpy(), bin_edges)
            print(pd.DataFrame([hist, hist_], index=['Before', 'After'], columns=ordinal_labels).rename_axis(column).to_markdown(), '\n')
            temp_dict[column] = data_.astype(int).to_list()

        return pd.DataFrame(temp_dict)

    def peak_elevation_approach(self, df: pd.DataFrame, no_neighbors: int, threshold: int) -> pd.DataFrame:
        """Modify label values based on criteria that evaluates each individual label's adjacency and compares it with the label class."""
        temp_dict = {}
        for column, data in df.items():
            temp_list = []
            for i, window in enumerate(data.rolling(2 * no_neighbors + 1, min_periods=1, center=True, closed='both')):
                temp_list.append([window[i], window.drop(i).mean()])
            labels_stats = pd.DataFrame(temp_list, columns=['label', 'mean'])
            ordinal_labels = np.unique(data.to_numpy())
            hist, _ = np.histogram(data.to_numpy(), len(ordinal_labels))
            neighbor_label_means = {unique_label: labels_stats['mean'].where(labels_stats['label'].eq(unique_label)).mean() for unique_label in ordinal_labels}
            labels_stats = pd.concat([labels_stats, pd.Series([neighbor_label_means[label] for label in labels_stats['label']], name='label_mean')], axis=1)
            result_list = []
            for _, label, mean, label_mean in labels_stats.itertuples():
                if mean / label_mean > threshold:
                    result_list.extend([label + 1] if label != max(ordinal_labels) else [label])
                else:
                    result_list.extend([label])
            ordinal_labels_ = np.unique(result_list)
            hist_, _ = np.histogram(result_list, len(ordinal_labels_))
            print(pd.DataFrame([hist, hist_], index=['Before', 'After'], columns=ordinal_labels if len(ordinal_labels) > len(ordinal_labels_) else ordinal_labels_).rename_axis(column).to_markdown(), '\n')
            temp_dict[column] = result_list
        return pd.DataFrame(temp_dict)

class TimeSeriesSimulator:
    def __init__(self, sb_code, days):
        """Initialize the simulator with a SimBench code and number of days for the simulation."""
        self.sb_code = sb_code
        self.days = days
        self.net = self.load_net(sb_code)
        self.time_steps = range(96 * days)  # 15-minute intervals for the specified number of days
        self.label_processor = LabelProcessor()

    def load_net(self, sb_code):
        """Load the SimBench network based on the provided code."""
        net = sb.get_simbench_net(sb_code)
        assert not sb.profiles_are_missing(net), "Profiles are missing in the loaded network."
        return net

    def apply_absolute_values(self, absolute_values_dict, case_or_time_step):
        """Assign absolute values to the data in the network."""
        for elm_param, data in absolute_values_dict.items():
            if data.shape[1]:
                elm, param = elm_param
                self.net[elm].loc[:, param] = data.loc[case_or_time_step]

    def prepare_profiles(self):
        """Prepare and retrieve the absolute profiles."""
        profiles = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True)
        return profiles

    def run_power_flow(self, profiles):
        """Run power flow simulations and store the results."""
        results = pd.DataFrame([], index=self.time_steps, columns=['Load Sum', 'min_vm_pu', 'max_vm_pu'])
        self.net.trafo.tap_pos = 1  # Set trafo tap position to avoid voltage violations
        
        for time_step in self.time_steps:
            self.apply_absolute_values(profiles, time_step)
            pp.runpp(self.net, numba=False)
            results.loc[time_step] = [
                self.net.res_load.p_mw.sum(),
                self.net.res_bus.vm_pu.min(),
                self.net.res_bus.vm_pu.max()
            ]
        return results

    def plot_profiles(self, results, profiles):
        """Visualize the load sum result and grid profiles."""
        load_profile_names = pd.Series(self.net.profiles['load'].columns)
        load_profile_names = list(load_profile_names.loc[load_profile_names.str.contains('H0') & load_profile_names.str.contains('pload')])
        load_profiles = self.net.profiles['load'].loc[self.time_steps, load_profile_names]

        fig, ax1 = plt.subplots(figsize=(15, 5))
        ax2 = ax1.twinx()
        ax1.plot(load_profiles, label=load_profile_names)
        ax2.plot(results, 'k', label=results.columns)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc=2)
        sns.move_legend(ax1, 'lower center', bbox_to_anchor=(.5, 1), ncol=6, title=None, frameon=False)
        ax1.set_title('Profiles of Study Cases', pad=30)
        ax1.set_ylabel('Active Power Sum (MW)')
        ax2.set_ylabel('Active Power per Peak Power')
        ax1.set_xlabel('Time (15 min)')
        ax1.set_xlim(0, load_profiles.index[-1])
        ax1.grid(True)
        plt.show()  

        fig, ax = plt.subplots()
        ax.plot(results.loc[:, ['min_vm_pu', 'max_vm_pu']], label=['min_vm_pu', 'max_vm_pu'])
        ax.axhline(0.9, color='r')
        ax.axhline(1.1, color='r')
        ax.legend()
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
        ax.set_xlim(0, results.index[-1])
        ax.set_title('Voltage Extrema')
        ax.set_xlabel('Time (15 min)')
        ax.set_ylabel('Voltage (pu)')
        ax.grid(True)
        plt.show()  

    def plot_profile(self, data, title):
        """Visualize the provided profile."""
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(data, label=data.columns)
        ax.legend()
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_title(title)
        ax.set_xlabel('Time (15 min)')
        ax.set_ylabel('Active Power (MW)')
        ax.set_xlim(0, data.index[-1])
        ax.grid(True)
        plt.show()

    def collect_data_at_pv_buses(self, profiles):
        """Collect data at PV buses and print status updates."""
        print("Collecting data at PV buses...")
        index_list, bus_id_list = [], []
        for series in [bus == self.net.load['bus'] for bus in self.net.sgen['bus']]:
            index_list.extend(list(series.loc[series.tolist()].index))
            bus_id_list.extend(list(self.net.load.loc[list(series.loc[series.tolist()].index), 'bus']))

        sgen_active_power_data = profiles[('sgen', 'p_mw')].set_axis(
            [f"{name[name.find(' ') + 1:]} @Bus {bus_id}" for name, bus_id in zip(self.net.sgen['name'].tolist(), self.net.sgen['bus'].tolist())], 
            axis=1
        )

        # Only use the columns for which we have matching bus IDs
        valid_columns = min(len(bus_id_list), profiles[('load', 'p_mw')].shape[1])
        load_active_power_data = profiles[('load', 'p_mw')].iloc[:, :valid_columns].set_axis(
            [f"{name[name.find(' ') + 1:]} @Bus {bus_id}" for name, bus_id in zip(self.net.load['name'].tolist()[:valid_columns], bus_id_list[:valid_columns])], 
            axis=1
        )
        
        print("Data collection completed.")
        return sgen_active_power_data, load_active_power_data

    def run_analysis(self):
        """Run the analysis process."""
        self.profiles = self.prepare_profiles()
        results = self.run_power_flow(self.profiles)
        self.plot_profiles(results, self.profiles)
        self.sgen_active_power_data, self.load_active_power_data = self.collect_data_at_pv_buses(self.profiles)

        self.plot_profile(self.sgen_active_power_data.loc[self.time_steps], 'PV profiles')
        self.plot_profile(self.load_active_power_data.loc[self.time_steps], 'Load profiles at buses with PV')
        self.plot_profile(self.sgen_active_power_data, 'PV profiles (whole year)')
        self.plot_profile(self.load_active_power_data, 'Load profiles at buses with PV (whole year)')

    def run_label_prepare_process(self):
        """Run the label preparation process."""
        print("Binary labels for load active power")
        binary_labels_load = self.label_processor.binary_labels(self.load_active_power_data.iloc[self.time_steps, [0]], 0.7)
        self.label_processor.plot_profiles_and_labels(self.load_active_power_data.iloc[self.time_steps, [0]], binary_labels_load)
        
        print("Binary labels for sgen active power")
        binary_labels_sgen = self.label_processor.binary_labels(self.sgen_active_power_data.iloc[self.time_steps, [0]])
        self.label_processor.plot_profiles_and_labels(self.sgen_active_power_data.iloc[self.time_steps, [0]], binary_labels_sgen)

        print("Three sigma labels for load active power")
        three_sigma_labels = self.label_processor.three_sigma_labels(self.load_active_power_data.iloc[self.time_steps, [0]])
        self.label_processor.plot_profiles_and_labels(self.load_active_power_data.iloc[self.time_steps, [0]], three_sigma_labels)

        print("Five sigma labels for load active power")
        five_sigma_labels = self.label_processor.five_sigma_labels(self.load_active_power_data.iloc[self.time_steps, [0]])
        self.label_processor.plot_profiles_and_labels(self.load_active_power_data.iloc[self.time_steps, [0]], five_sigma_labels)

        print("Discretized labels for load active power")
        discretized_labels_load = self.label_processor.discretize(self.load_active_power_data.iloc[self.time_steps, [0]], bin_by='std')
        self.label_processor.plot_profiles_and_labels(self.load_active_power_data.iloc[self.time_steps, [0]], discretized_labels_load)
        print("Discretized labels for sgen active power")
        discretized_labels_sgen = self.label_processor.discretize(self.sgen_active_power_data.iloc[self.time_steps, [0]], bin_by='std')
        self.label_processor.plot_profiles_and_labels(self.sgen_active_power_data.iloc[self.time_steps, [0]], discretized_labels_sgen)
        print("Season labels for sgen active power (Months)") 
        season_labels = self.label_processor.season_labels((0, 4))
        self.label_processor.plot_profiles_and_labels(self.sgen_active_power_data.iloc[:4 * (96 * 7 * 4), [0]], season_labels)
        print("Season magnitude labels for sgen active power")
        season_magnitude_labels = self.label_processor.season_magnitude_labels((0, 12), int(self.sgen_active_power_data.shape[0] / 12))
        
        # Ensure season_magnitude_labels has the correct shape
        season_magnitude_labels = season_magnitude_labels.iloc[:self.sgen_active_power_data.shape[0]]
        season_magnitude_labels.columns = ['season']
        
        self.label_processor.plot_profiles_and_labels(self.sgen_active_power_data.iloc[:, [0]], season_magnitude_labels)

    def run_peak_spread_label(self):
        """Run the peak spread label process."""
        five_sigma_labels = self.label_processor.five_sigma_labels(self.load_active_power_data.iloc[self.time_steps, [0]])
        peak_spread_labels = self.label_processor.peak_spreading_approach(df=five_sigma_labels, 
                                                                          no_neighbors=2,
                                                                          target_label_list=None)
        self.label_processor.plot_profiles_and_labels(self.load_active_power_data.iloc[self.time_steps, [0]], peak_spread_labels)

    def run_elevated_label(self):
        """Run the elevated label process."""
        five_sigma_labels = self.label_processor.five_sigma_labels(self.load_active_power_data.iloc[self.time_steps, [0]])
        peak_elevated_labels = self.label_processor.peak_elevation_approach(df=five_sigma_labels, 
                                                                            no_neighbors=20, 
                                                                            threshold=1.25)
        self.label_processor.plot_profiles_and_labels(self.load_active_power_data.iloc[self.time_steps, [0]], peak_elevated_labels)