"""evaluation_processing.py"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import signal
from itertools import chain
from sklearn.utils import shuffle
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
import os  # Add this import for saving plots

class Evaluation:
    def __init__(self):
        pass

    def count_datapoints(self, df: pd.DataFrame, threshold: float) -> int:
        temp_dict = dict()
        for column, data in df.items():
            temp_list = [value >= threshold for _, value in data.items()]
            unique, counts = np.unique(temp_list, return_counts=True)
            temp_dict[column] = [counts[0], len(data) - counts[0]]
        print(pd.DataFrame(temp_dict, index=['No Overshoot', 'Overshoot']).to_markdown())

    def prepare_data(self, *inputs: tuple, horizon: int, shuffle_stack: bool = True, random_state: int = None) -> np.ndarray:
        """Conduct preprocessing, i.e, scale data, slice data into sequences and shuffle data stack.
        Consistent shuffling between multiple data stacks must be performed separetaly."""
        if len(inputs) > 2:
            raise Exception('Only one input (data) or two inputs (data and labels) are allowed')
        return_list = list()
        for data, bool_scale in inputs:
            #Create Minimum-Maximum Scaler
            if bool_scale:
                scaler = MinMaxScaler().fit(data)
                data = scaler.transform(data)
            #Create a list holding the sequences defined by sliding window of width = horizon and stack to a 3-dimensional array (batch, horizon, feature)
            data_stack = np.stack([data[i:i + horizon] for i in range(len(data) - horizon)])
            if shuffle_stack:
                data_stack = shuffle(data_stack, random_state=random_state)
            return_list.extend([data_stack, scaler.data_max_, scaler.data_min_]) if bool_scale else return_list.extend([data_stack])
        return return_list if len(return_list) > 1 else return_list.pop()

    def low_dimensional_representation(self, data_original: np.ndarray, data_synthetic: np.ndarray, technique: str, random_state: int = 58) -> np.ndarray:
        assert technique in ['pca', 'tsne']
        sample_no = min([1000, len(data_original)])
        np.random.seed(random_state)
        idx = np.random.permutation(len(data_original))[:sample_no]
        data_original, data_synthetic = np.asarray(data_original)[idx], np.asarray(data_synthetic)[idx]
        no, seq_len, dim = data_original.shape
        prep_data = np.array([np.reshape(np.mean(data_original[i, :, :], 1), [1, seq_len]).flatten().tolist() for i in range(sample_no)])
        prep_data_hat = np.array([np.reshape(np.mean(data_synthetic[i, :, :], 1), [1, seq_len]).flatten().tolist() for i in range(sample_no)])
        if technique == 'pca':
            pca = PCA(n_components=2, random_state=random_state)
            pca.fit(prep_data)
            return pca.transform(prep_data), pca.transform(prep_data_hat)
        elif technique == 'tsne':
            prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=random_state)
            tsne_result = tsne.fit_transform(prep_data_final)
            return tsne_result[:sample_no], tsne_result[sample_no:]

    def distribution_estimate(self, data_original: np.ndarray, data_synthetic: np.ndarray, technique: str) -> None:
        assert technique in ['pca', 'tsne']
        sample_no = min([1000, len(data_original)])
        np.random.seed(42)
        idx = np.random.permutation(len(data_original))[:sample_no]
        data_original, data_synthetic = np.asarray(data_original)[idx], np.asarray(data_synthetic)[idx]
        no, seq_len, dim = data_original.shape
        prep_data = np.array([np.reshape(np.mean(data_original[i, :, :], 1), [1, seq_len]).flatten().tolist() for i in range(sample_no)])
        prep_data_hat = np.array([np.reshape(np.mean(data_synthetic[i, :, :], 1), [1, seq_len]).flatten().tolist() for i in range(sample_no)])
        colors = ['red' for i in range(sample_no)] + ['blue' for i in range(sample_no)]
        if technique == 'pca':
            pca = PCA(n_components=2)
            pca.fit(prep_data)
            pca_results = pca.transform(prep_data)
            pca_hat_results = pca.transform(prep_data_hat)
            fig, ax = plt.subplots()
            ax.scatter(pca_results[:, 0], pca_results[:, 1], c=colors[:sample_no], alpha=0.2, label='Original')
            ax.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c=colors[sample_no:], alpha=0.2, label='MC-TimeGAN')
            ax.legend()
            sns.move_legend(ax, 'lower center', bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
            ax.set_xlabel('x-pca')
            ax.set_ylabel('y-pca')
            plt.show()
        elif technique == 'tsne':
            prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(prep_data_final)
            fig, ax = plt.subplots()
            ax.scatter(tsne_results[:sample_no, 0], tsne_results[:sample_no, 1], c=colors[:sample_no], alpha=0.2, label='Original')
            ax.scatter(tsne_results[sample_no:, 0], tsne_results[sample_no:, 1], c=colors[sample_no:], alpha=0.2, label='MC-TimeGAN')
            ax.legend()
            sns.move_legend(ax, 'lower center', bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
            ax.set_xlabel('x-tsne')
            ax.set_ylabel('y-tsne')
            plt.show()

    def numeric_evaluation(self, do: pd.DataFrame, ds: pd.DataFrame) -> None:
        funcs = [np.mean, np.max, np.min]
        temp_list = list()
        for func in funcs:
            temp_list += [do.apply(func).to_list()]
            temp_list += [ds.apply(func).to_list()]
            delta = (ds.apply(func) - do.apply(func)) / do.apply(func) * 100
            temp_list += [delta.to_list()]
        index = ['Mean_O', 'Mean_S', 'Delta (%)', 'Max_O', 'Max_S', 'Delta (%)', 'Min_O', 'Min_S', 'Delta (%)']
        print(pd.DataFrame(temp_list, index=index, columns=do.columns.to_list()).to_markdown(tablefmt='github'))

    def plot_res(self, df_o: pd.DataFrame, df_s: pd.DataFrame, limits: list, ylabel: str, saving: tuple = ('', False)) -> None:
        for (_, do), (_, ds) in zip(df_o.items(), df_s.items()):
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.plot(do, label='Original')
            ax.plot(ds, alpha=0.75, label='MC-TimeGAN')
            [ax.axhline(limit, color='r', alpha=0.5, lw=0.75) for limit in limits]
            ax.set_xlim(-0.1, len(do) + 0.1)
            ax.set_xticks(np.arange(0, len(do) + 1, 96))
            ax.set_xticklabels(np.arange(0, (len(do) + 1) / 96, 1, dtype='int'))
            ax.set_xlabel('Time (Days)')
            ax.set_ylabel(ylabel)
            ax.legend()
            sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
            ax.grid(True)
            if not len(limits) > 1:
                ax.set_ylim(0, 1.05 * max(do.max(), ds.max()) if max(do.max(), ds.max()) > limits[0] else 1.05 * limits[0])
            else:
                ax.set_ylim(0.995 * min(do.min(), ds.min()) if min(do.min(), ds.min()) < limits[-1] else 0.995 * limits[-1],
                            1.005 * max(do.max(), ds.max()) if max(do.max(), ds.max()) > limits[0] else 1.005 * limits[0])
            if saving[-1]:
                plt.savefig(saving[0], bbox_inches='tight', pad_inches=0)
            plt.show()

    def plot_cumsum(self, *dfs: pd.DataFrame, yname: str) -> None:
        assert len(dfs) < 3
        colors = list(mcolors.TABLEAU_COLORS.keys())
        markers = list(mmarkers.MarkerStyle.markers)[2:]
        indices = np.random.choice(dfs[0].shape[-1], len(colors)) if dfs[0].shape[-1] > len(colors) else range(dfs[0].shape[-1])
        fig, ax = plt.subplots(figsize=(15, 4))
        for i, df in enumerate(dfs):
            for j, index in enumerate(indices):
                ax.plot(df.iloc[:, [index]].cumsum(), color=colors[j], ls='-' if i == 0 else '--', marker='' if i == 0 else markers[j], markevery=0.05, label=df.columns[j])
        ax.legend()
        sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
        ax.set_xlim(-0.1, len(df))
        ax.set_xlabel('Time (15 min)')
        ax.set_ylabel('Active ' + yname + ' Power (MW)')
        ax.grid(True)

    def plot_acf_and_pacf(self, *dfs: pd.DataFrame, acf_lags: int = None, pacf_lags: int = None) -> None:
        fig, ax = plt.subplots(dfs[0].shape[-1], 2, figsize=(12, 3 * dfs[0].shape[-1]), layout='constrained')
        for i, df in enumerate(dfs):
            for j, ((_, data)) in enumerate(df.items()):
                ax[j, 0].sharey(ax[j, 1])
                ax[j, 0].plot(range(len(data)) if acf_lags is None else range(acf_lags + 1), acf(data, nlags=len(data) if acf_lags is None else acf_lags), alpha=1 if i == 0 else 0.75)
                ax[j, 1].stem(range(96 + 1) if pacf_lags is None else range(pacf_lags + 1), pacf(data, nlags=96 if pacf_lags is None else pacf_lags, method='ols'), linefmt='C0-' if i == 0 else 'C1-', markerfmt='o' if i == 0 else 'x')
                ax[j, 0].set_xlim(-0.1, len(data) + 0.1 if acf_lags is None else acf_lags + 0.1)
                ax[j, 1].set_xlim(-0.1, 96 + 0.1 if pacf_lags is None else pacf_lags + 0.1)
                [col.set_xlabel('Lags (15min)') for col in ax[j, :]]
                ax[j, 0].set_ylabel('ACF')
                ax[j, 1].set_ylabel('pACF')
                [col.grid(True) for col in ax[j, :]]
        plt.show()

    def prepare_violin(self, *dfs: pd.DataFrame, names: list = ['Data', 'Profile', 'Type']) -> pd.DataFrame:
        assert len(dfs) == 2, "Exactly 2 dataframes required"
        assert len(names) == 3, "Names list must contain 3 elements"
        data_dict = {names[0]: [], names[1]: [], names[2]: []}
        for i, (column, data) in enumerate(chain(dfs[0].items(), dfs[1].items())):
            data_dict[names[0]].extend(data.tolist())
            data_dict[names[1]].extend([column] * len(data))
            data_type = 'Original' if i < dfs[0].shape[1] else 'MC-TimeGAN'
            data_dict[names[2]].extend([data_type] * len(data))
        return pd.DataFrame(data_dict)

    def plot_violin(self, res_line_original: pd.DataFrame, res_line_synthetic: pd.DataFrame) -> None:
        df_violin = self.prepare_violin(res_line_original, res_line_synthetic)
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.violinplot(data=df_violin, x='Profile', y='Data', hue='Type', split=True, gap=0.1, inner='quart')
        sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
        ax.set_title('Violin Plot', pad=30)
        plt.show()

    def plot_profile_violin_and_cumsum(self, load_original: pd.DataFrame, load_synthetic: pd.DataFrame) -> None:
        fig, ax = plt.subplots(load_original.shape[1], 2, figsize=(15, 4 * load_original.shape[1]))
        for i, _ in enumerate(load_original.items()):
            df_violin = self.prepare_violin(load_original.iloc[:, [i]], load_synthetic.iloc[:, [i]])
            sns.violinplot(data=df_violin, x='Data', y='Profile', hue='Type', split=True, gap=0.1, inner='quart', ax=ax[i, 0])
            ax[i, 0].set_xlabel('Active power in MW')
            ax[i, 0].set_yticks(ax[i, 0].get_yticks())
            ax[i, 0].set_yticklabels(ax[i, 0].get_yticklabels(), rotation='vertical', verticalalignment='center')
            ax[i, 1].plot(load_original.iloc[:, [i]].cumsum(), label='Original')
            ax[i, 1].plot(load_synthetic.iloc[:, [i]].cumsum(), ls='--', marker='o', markevery=0.05, label='MC-TimeGAN')
            ax[i, 1].set_xlabel('Time in quarter hours')
            ax[i, 1].set_ylabel('Active power in MW')
            ax[i, 1].legend()
        plt.show()

    def plot_acf_comparison(self, load_original: pd.DataFrame, load_synthetic: pd.DataFrame) -> None:
        fig, ax = plt.subplots(load_original.shape[1], 2, figsize=(12, 3 * load_original.shape[1]), layout='constrained')
        for i, ((_, do), (_, ds)) in enumerate(zip(load_original.items(), load_synthetic.items())):
            ax[i, 0].sharey(ax[i, 1])
            ax[i, 0].plot(range(len(do)), acf(do, nlags=len(do)), range(len(ds)), acf(ds, nlags=len(ds)), alpha=0.75)
            ax[i, 1].plot(range(len(do)), acf(do, nlags=len(do)), range(len(ds)), acf(ds, nlags=len(ds)), alpha=0.75)
            ax[i, 0].set_xlim(-0.1, len(do) + 0.1)
            ax[i, 1].set_xlim(-0.1, 150 + 0.1)
            for col in ax[i, :]:
                col.set_xlabel('Lags (15min)')
                col.set_ylabel('ACF')
                col.grid(True)
        plt.show()

    def plot_psd(self, load_original: pd.DataFrame, load_synthetic: pd.DataFrame, column_index: int = 4, fs: float = 1e-5, nperseg: int = 1024) -> None:
        f_synthetic, psd_synthetic = signal.welch(load_synthetic.to_numpy()[:, column_index], fs=fs, nperseg=nperseg)
        f_original, psd_original = signal.welch(load_original.to_numpy()[:, column_index], fs=fs, nperseg=nperseg)
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.semilogy(f_original, psd_original, label='Original')
        ax.semilogy(f_synthetic, psd_synthetic, alpha=0.75, label='MC-TimeGAN')
        ax.legend()
        sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
        ax.set_xlim(0, f_original[-1])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (V**2/Hz)')
        ax.grid(True)
        plt.show()

    def plot_trafo_line_voltage(self, trafo, line, voltage, saving=(False, ''), fontsize=12):
        # Ensure LaTeX is used for text rendering
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams.update({
            'font.size': fontsize,
            'axes.titlesize': fontsize,
            'axes.labelsize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'legend.fontsize': fontsize,
            'figure.titlesize': fontsize,
            'pgf.rcfonts': False
        })
        # Configure plot dimensions according to IEEE standard sizing
        fig_width = 5.5
        fig_height = 3.375 * 1.70  # Height adjusted for three subplots
        fig, ax = plt.subplots(3, 1, figsize=(fig_width, fig_height))  # Adjusted figure size for better spacing
    
        for i, grid_state_metric in enumerate([trafo, line, voltage]):
            ax[i].plot(grid_state_metric[0],  label='Original', linewidth=1.2)  # Use separate marker and linestyle
            ax[i].plot(grid_state_metric[-1], label='MC-TimeGAN', linewidth=1.2)  # Use separate marker and linestyle
            if i == 0:
                ax[i].axhline(100, color='r', alpha=0.5, lw=1.2)
                ax[i].set_yticks([0, 25, 50, 75, 100, 125])
            elif i == 1:
                ax[i].axhline(100, color='r', alpha=0.5, lw=1.2)
                ax[i].set_yticks([0, 20, 40, 60, 80, 100])
            else:
                ax[i].axhline(0.95, color='r', alpha=0.5, lw=1.2)
                ax[i].axhline(1.05, color='r', alpha=0.5, lw=1.2)
                ax[i].set_yticks([0.95, 0.97, 1.00, 1.02, 1.05])
            ax[i].set_xlim(-0.1, len(grid_state_metric[0]) - 1)
            ylabel = [r'Transformer Load (\%)', r'Line Load (\%)', r'Bus Voltage (pu)'][i]
            ax[i].set_ylabel(ylabel, fontsize=fontsize)
            ax[i].set_xticks(np.arange(0, len(grid_state_metric[0]) + 1, 96))
            ax[i].set_xticklabels(np.arange(0, (len(grid_state_metric[0]) + 1) / 96, 1, dtype=int), fontsize=fontsize)
            ax[i].grid(True)
    
        ax[0].legend(fontsize=fontsize)
        sns.move_legend(ax[0], "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
        ax[-1].set_xlabel(r'Time (Days)', fontsize=fontsize)
    
        if saving[0]:
            pgf_folder = os.path.join('helper', 'output', 'single_feeder_grid', 'figures')
            os.makedirs(pgf_folder, exist_ok=True)
            pgf_file_name = os.path.join(pgf_folder, saving[-1] + '.pgf')
            plt.savefig(pgf_file_name, bbox_inches='tight', pad_inches=0, format='pgf')
            pdf_file_name = os.path.join(pgf_folder, saving[-1] + '.pdf')
            plt.savefig(pdf_file_name, bbox_inches='tight', pad_inches=0)
            print(pdf_file_name)
            print(pgf_file_name)
        plt.show()