import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import logging
import matplotlib.ticker as mticker


class Visualizer:
    # --- 1. 核心样式定义 (参考你的 exp_6.1_macro_combined_01.png) ---
    # 强制定义线宽、颜色、标记和图层顺序(zorder)
    # Zorder 越大，画在越上面
    MODEL_STYLES = {
        "Non-Private Optimal": {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': 3.0, 'zorder': 100},
        "POETIC": {'color': '#d62728', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5, 'zorder': 90,
                   'markersize': 8},  # Red
        "POETIC-slow": {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'linewidth': 2.5, 'zorder': 80,
                        'markersize': 8},  # Orange
        "POETIC-No-Adaptive": {'color': '#2ca02c', 'linestyle': ':', 'marker': 'p', 'linewidth': 2.0, 'zorder': 70,
                               'markersize': 8},  # Green
        "PrivCO": {'color': '#1f77b4', 'linestyle': '-.', 'marker': '^', 'linewidth': 1.5, 'zorder': 60,
                   'markersize': 7},  # Blue
        "OPPS-adapted": {'color': '#9467bd', 'linestyle': '-', 'marker': 'x', 'linewidth': 1.5, 'zorder': 50,
                         'markersize': 7},  # Purple
        "Incentivized FL-SC": {'color': '#8c564b', 'linestyle': '--', 'marker': 'd', 'linewidth': 1.5, 'zorder': 40,
                               'markersize': 7},  # Brown
        "Truthful-DP-MAB": {'color': '#e377c2', 'linestyle': ':', 'marker': '*', 'linewidth': 1.5, 'zorder': 30,
                            'markersize': 9},  # Pink
        "Random": {'color': 'grey', 'linestyle': ':', 'marker': '.', 'linewidth': 1.5, 'zorder': 20, 'markersize': 5},
        # Grey

        # 消融实验变体
        # === Exp 6.9 消融实验 (颜色区分优化) ===
        # 1. Base: 改为蓝色，与 Final 形成对比
        "POETIC-Base": {'color': '#1f77b4', 'linestyle': '--', 'marker': '', 'linewidth': 2.0, 'zorder': 60},
        # 2. Base+Explore: 改为绿色，易于区分
        "POETIC-Base + Explore": {'color': '#2ca02c', 'linestyle': '-.', 'marker': '', 'linewidth': 2.0, 'zorder': 70},
        # 3. Final: 保留红色
        "POETIC-Final": {'color': '#d62728', 'linestyle': '-', 'marker': '', 'linewidth': 2.5, 'zorder': 80},

        # --- ICDE 2026 新增模型 ---
        "POETIC-Full":          {'color': '#d62728', 'linestyle': '-',  'marker': 'o', 'linewidth': 2.0, 'zorder': 90, 'markersize': 6},
        "POETIC-NoDecouple":    {'color': '#1f77b4', 'linestyle': '--', 'marker': 's', 'linewidth': 1.8, 'zorder': 70, 'markersize': 6},
        "POETIC-NoStability":   {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'linewidth': 1.8, 'zorder': 65, 'markersize': 6},
        "POETIC (Full)":        {'color': '#d62728', 'linestyle': '-',  'marker': 'o', 'linewidth': 2.0, 'zorder': 90, 'markersize': 6},
        "w/o Φ":               {'color': '#1f77b4', 'linestyle': '--', 'marker': 's', 'linewidth': 1.8, 'zorder': 70, 'markersize': 6},
        "w/o StabRep":          {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'linewidth': 1.8, 'zorder': 65, 'markersize': 6},
        "w/o Both":             {'color': '#9467bd', 'linestyle': ':',  'marker': 'x', 'linewidth': 1.5, 'zorder': 60, 'markersize': 6},
        "Manipulative (POETIC)":     {'color': '#d62728', 'linestyle': '--', 'marker': '', 'linewidth': 1.8, 'zorder': 75},
        "Stable (POETIC)":           {'color': '#2ca02c', 'linestyle': '-',  'marker': '', 'linewidth': 1.8, 'zorder': 76},
        "Manipulative (w/o StabRep)": {'color': '#ff7f0e', 'linestyle': '--', 'marker': '', 'linewidth': 1.8, 'zorder': 55},
        "Stable (w/o StabRep)":       {'color': '#8c564b', 'linestyle': '-',  'marker': '', 'linewidth': 1.8, 'zorder': 56},
    }

    # --- 2. 强制排序列表 ---
    # 所有图例和绘图循环都必须遵从这个顺序，防止颜色错乱
    ORDERED_MODELS = [
        "Non-Private Optimal",
        "POETIC",
        "POETIC-slow",
        "POETIC-No-Adaptive",
        "POETIC-Final",
        "POETIC-Base + Explore",
        "POETIC-Base",
        "PrivCO",
        "OPPS-adapted",
        "Incentivized FL-SC",
        "Truthful-DP-MAB",
        "Random",
        # ICDE ablation variants
        "POETIC (Full)",
        "POETIC-Full",
        "POETIC-NoDecouple",
        "POETIC-NoStability",
        "w/o Φ",
        "w/o StabRep",
        "w/o Both",
    ]

    def __init__(self, config):
        self.config = config
        self.figures_dir = self.config.FIGURES_DIR
        os.makedirs(self.figures_dir, exist_ok=True)

        # 重置并应用全局样式
        plt.rcdefaults()
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 14,
            'axes.labelsize': 18,
            'axes.titlesize': 20,
            'legend.fontsize': 13,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'figure.dpi': 300,
            'lines.linewidth': 1.5,

            # --- [修复] 黑色实线外框设置 ---
            'axes.edgecolor': 'black',  # 边框颜色：黑
            'axes.linewidth': 1.5,  # 边框粗细：加粗
            'axes.spines.bottom': True,  # 确保下边框显示
            'axes.spines.left': True,  # 确保左边框显示
            'axes.spines.right': True,  # 确保右边框显示
            'axes.spines.top': True,  # 确保上边框显示

            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.6,
            'pdf.fonttype': 42,
            'ps.fonttype': 42
        })
        self.palette = sns.color_palette("colorblind")


    def _get_style(self, model_name, index=0):
        """统一获取样式，优先查内置表，查不到则回退到 Config 或自动颜色"""
        # 1. 查内置表 (最高优先级)
        if model_name in self.MODEL_STYLES:
            return self.MODEL_STYLES[model_name]

        # 2. 查 Config (兼容旧代码)
        config_styles = getattr(self.config, 'MODEL_STYLES', {})
        if model_name in config_styles:
            return config_styles[model_name]

        # 3. 自动分配 (兜底)
        return {
            'color': self.palette[index % len(self.palette)],
            'linestyle': '-',
            'marker': 'o',
            'linewidth': 1.5,
            'zorder': 5
        }

    # 为了兼容可能的旧调用，保留 _get_model_style_safe 但指向 _get_style
    def _get_model_style_safe(self, model_name, index=0):
        return self._get_style(model_name, index)



    def _save_fig(self, fig, filename):
        full_path = os.path.join(self.figures_dir, filename)
        try:
            fig.tight_layout(pad=1.0, rect=[0, 0.03, 1, 0.96])
            fig.savefig(full_path, bbox_inches='tight')
            logging.info(f"Figure saved to {full_path}")
        except Exception as e:
            logging.error(f"Failed to save figure {filename}: {e}")
        finally:
            plt.close(fig)

    def _plot_mean_std_curve(self, ax, data_runs, label, style, markevery=20):
        if not data_runs: return 0
        try:
            np_runs = [np.array(run).flatten() for run in data_runs]
            min_len = min(len(run) for run in np_runs) if np_runs else 0
            if min_len == 0: return 0
        except Exception as e:
            return 0

        truncated_runs = [run[:min_len] for run in np_runs]
        stacked_runs = np.vstack(truncated_runs)
        mean_curve = np.mean(stacked_runs, axis=0)
        std_curve = np.std(stacked_runs, axis=0)
        x_values = range(1, min_len + 1)

        plot_kwargs = {k: v for k, v in style.items() if v is not None}

        # 【修复 TypeError】: 处理 markevery 为 None 的情况
        if markevery is None:
            markevery = 20
        step_size = max(1, min_len // markevery)

        line, = ax.plot(x_values, mean_curve, label=label, **plot_kwargs,
                        markevery=step_size)

        ax.fill_between(x_values, mean_curve - std_curve, mean_curve + std_curve,
                        color=line.get_color(), alpha=0.1)
        return min_len

    def plot_cumulative_regret_vs_rounds(self, results_dict, title, filename):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(title, fontsize=17)
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Cumulative Regret")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        model_names = list(results_dict.keys())
        for i, name in enumerate(model_names):
            runs = results_dict[name]
            style = self._get_model_style_safe(name, i)
            # Regret 曲线通常不需要 marker，因为很密集
            style['marker'] = ''
            self._plot_mean_std_curve(ax, runs, name, style, markevery=None)

        ax.legend(loc='upper left', fontsize=11)
        self._save_fig(fig, filename)

    # --- Refined Function for Exp 6.7 / 6.8 ---
    def plot_incentive_verification_bars(self, mean_results_dict, std_results_dict, title, filename):
        strategies = list(mean_results_dict.keys())
        if not strategies: return

        profits_mean = [mean_results_dict.get(s, {}).get('avg_worker_profit', 0) for s in strategies]
        values_mean = [mean_results_dict.get(s, {}).get('final_cumulative_value', 0) for s in strategies]
        profits_std = [std_results_dict.get(s, {}).get('avg_worker_profit', 0) for s in strategies]
        values_std = [std_results_dict.get(s, {}).get('final_cumulative_value', 0) for s in strategies]

        colors = []
        for i, s in enumerate(strategies):
            style = self._get_model_style_safe(s, i)
            colors.append(style['color'])

        hatches = ['//', '\\\\', 'xx', '..']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x_pos = np.arange(len(strategies))

        ax1.bar(x_pos, profits_mean, yerr=profits_std, color=colors, edgecolor='black',
                capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
        ax1.set_title("(a) Worker Profit vs. Strategy", fontsize=16)
        ax1.set_ylabel("Mean Avg Worker Profit")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(strategies, rotation=30, ha='right')
        for i, bar in enumerate(ax1.patches):
            bar.set_hatch(hatches[i % len(hatches)])
        ax1.margins(y=0.1)

        ax2.bar(x_pos, values_mean, yerr=values_std, color=colors, edgecolor='black',
                capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
        ax2.set_title("(b) Platform Value vs. Strategy", fontsize=16)
        ax2.set_ylabel("Mean Final Cumulative Value")
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(strategies, rotation=30, ha='right')
        for i, bar in enumerate(ax2.patches):
            bar.set_hatch(hatches[i % len(hatches)])
        ax2.margins(y=0.1)

        ax1.yaxis.grid(True, linestyle=':', linewidth=0.6, color='grey')
        ax2.yaxis.grid(True, linestyle=':', linewidth=0.6, color='grey')
        ax1.xaxis.grid(False)
        ax2.xaxis.grid(False)

        self._save_fig(fig, filename)

    # --- Refined Function for Exp 6.9 ---
    def plot_ablation_study(self, agg_results, title, filename):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        model_names = list(agg_results.keys())
        sorted_models = sorted(model_names,
                               key=lambda x: self.ORDERED_MODELS.index(x) if x in self.ORDERED_MODELS else 999)

        # Value Plot
        for i, name in enumerate(sorted_models):
            data = agg_results[name].get('cumulative_value_runs', [])
            style = self._get_style(name, i)
            style['marker'] = ''
            self._plot_mean_std_curve(ax1, data, name, style, markevery=None)

        ax1.set_title("(a) Impact on Platform Value")
        ax1.set_ylabel("Mean Cumulative Value")
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        # Profit Plot
        for i, name in enumerate(sorted_models):
            data = agg_results[name].get('worker_profit_history_agg_runs', [])
            style = self._get_style(name, i)
            style['marker'] = ''

            if data:
                try:
                    np_runs = [np.array(r).flatten() for r in data]
                    min_len = min(len(r) for r in np_runs)
                    x = range(1, min_len + 1)
                    mean_curve = np.mean([r[:min_len] for r in np_runs], axis=0)
                    smooth_window = max(1, int(min_len * 0.05))
                    smoothed = pd.Series(mean_curve).rolling(window=smooth_window, min_periods=1).mean().to_numpy()
                    ax2.plot(x, smoothed, label=name, **{k: v for k, v in style.items() if k != 'marker'})
                except:
                    pass

        ax2.set_title("(b) Impact on Worker Incentives")
        ax2.set_ylabel("Mean Avg Profit (Smoothed)")
        ax2.set_xlabel("Rounds")

        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(handles),
                       frameon=False)

        plt.subplots_adjust(bottom=0.1, hspace=0.2)
        self._save_fig(fig, filename)

    # --- Refined Function for Value vs k / Value vs N ---
    def plot_bar_chart_grouped(self, mean_results_dict, x_categories, metric_key, y_label, title, filename, x_label):
        model_names = list(mean_results_dict.keys())
        valid_x_categories = [
            cat for cat in x_categories
            if any(mean_results_dict[m].get(cat, {}).get(metric_key) is not None for m in model_names)
        ]
        if not valid_x_categories: return

        num_models = len(model_names)
        num_categories = len(valid_x_categories)
        fig_width = max(8, num_categories * num_models * 0.4)
        fig, ax = plt.subplots(figsize=(fig_width, 5))

        bar_width = 0.8 / num_models
        index = np.arange(num_categories)
        hatches = ['/', '\\', 'x', '.', '*', 'o', 'O', '+']

        for i, model_name in enumerate(model_names):
            values_mean = [mean_results_dict[model_name].get(cat, {}).get(metric_key, 0) for cat in valid_x_categories]
            std_metric_key = f"std_{metric_key}"
            values_std = [mean_results_dict[model_name].get(cat, {}).get(std_metric_key, 0) for cat in
                          valid_x_categories]

            positions = index + i * bar_width - (0.8 / 2) + bar_width / 2
            style = self._get_model_style_safe(model_name, i)
            color = style['color']
            hatch = hatches[i % len(hatches)]

            ax.bar(positions, values_mean, yerr=values_std, width=bar_width, label=model_name, color=color, hatch=hatch,
                   edgecolor='black', capsize=3, error_kw={'elinewidth': 0.8, 'capthick': 0.8})

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=17)
        ax.set_xticks(index)
        ax.set_xticklabels([str(xc) for xc in valid_x_categories])

        # 【修复 Value vs k 图例位置】: 从 (1, 1) 改回 'upper left'
        ax.legend(loc='upper left', fontsize=11)

        ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='grey')
        ax.xaxis.grid(False)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.margins(y=0.1)

        self._save_fig(fig, filename)





    def plot_privacy_tradeoff_curves(self, mean_results_by_epsilon, mean_optimal_final_value, title, filename):
        """
        修复说明：
        1. 图例强制一排排列 (ncol=len(handles))。
        2. 放置在底部留白区域的中心。
        """
        # 增加高度以容纳底部的一排图例
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        epsilons = sorted([e for e in mean_results_by_epsilon.keys() if e != np.inf]) + (
            [np.inf] if np.inf in mean_results_by_epsilon else [])
        plot_epsilons_finite = [e for e in epsilons if e != np.inf]
        max_finite_eps = max(plot_epsilons_finite) if plot_epsilons_finite else 1.0
        plot_epsilons = [e if e != np.inf else max_finite_eps * 5 for e in epsilons]

        first_eps = plot_epsilons_finite[0] if plot_epsilons_finite else np.inf
        if first_eps not in mean_results_by_epsilon:
            plt.close(fig);
            return
        model_names = list(mean_results_by_epsilon[first_eps].keys())

        # (a) Utility
        ax1.set_title("(a) Privacy-Utility Trade-off", fontsize=18)
        ax1.set_ylabel("Mean Final Value", fontsize=16)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        for i, model_name in enumerate(model_names):
            # 使用 safe 方法获取颜色
            style = self._get_model_style_safe(model_name, i)
            final_values = [mean_results_by_epsilon.get(e, {}).get(model_name, {}).get('final_cumulative_value', np.nan)
                            for e in epsilons]
            ax1.plot(plot_epsilons, final_values, label=model_name, **style)

        # (b) Regret
        ax2.set_title("(b) Privacy-Regret Trade-off", fontsize=18)
        ax2.set_xlabel(r"Privacy Budget ($\epsilon$)", fontsize=16)
        ax2.set_ylabel("Mean Final Regret", fontsize=16)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        for i, model_name in enumerate(model_names):
            style = self._get_model_style_safe(model_name, i)
            final_regrets = [mean_optimal_final_value - mean_results_by_epsilon.get(e, {}).get(model_name, {}).get(
                'final_cumulative_value', mean_optimal_final_value) for e in epsilons]
            final_regrets_clean = [r if not np.isnan(r) else np.nan for r in final_regrets]
            ax2.plot(plot_epsilons, final_regrets_clean, label=model_name, **style)

        # X轴处理
        ax2.set_xscale('log')
        tick_positions = plot_epsilons
        tick_labels = [str(e) if e != np.inf else r'$\infty$' for e in epsilons]
        if len(tick_positions) > 7:
            indices = np.linspace(0, len(tick_positions) - 1, 7, dtype=int)
            tick_positions = [tick_positions[i] for i in indices]
            tick_labels = [tick_labels[i] for i in indices]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, fontsize=12)
        ax2.minorticks_off()

        # 【核心修复】：图例布局
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            # ncol=len(handles) 强制一排显示
            # bbox_to_anchor=(0.5, 0.04) 调整到底部正中偏上一点的位置
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.03),
                       ncol=len(handles), frameon=False, fontsize=14)

        # 底部预留 10% 的空间给一排图例
        fig.tight_layout(pad=1.5, rect=[0, 0.10, 1, 0.96])

        full_path = os.path.join(self.figures_dir, filename)
        try:
            plt.savefig(full_path)
            logging.info(f"Figure saved to {full_path}")
        except Exception as e:
            logging.error(f"Failed to save figure {filename}: {e}")
        finally:
            plt.close(fig)

    def plot_bar_chart(self, mean_results_dict, metric_key, y_label, title, filename):
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(mean_results_dict.keys())
        values = [mean_results_dict.get(label, {}).get(metric_key, 0) for label in labels]

        std_metric_key = f"std_{metric_key}"
        stds = [mean_results_dict.get(label, {}).get(std_metric_key, 0) for label in labels]
        has_std = any(s > 0 for s in stds)

        colors = []
        for i, label in enumerate(labels):
            style = self._get_model_style_safe(label, i)
            colors.append(style['color'])

        x_pos = np.arange(len(labels))
        if has_std:
            ax.bar(x_pos, values, yerr=stds, color=colors, edgecolor='black', capsize=4,
                   error_kw={'elinewidth': 1, 'capthick': 1})
        else:
            ax.bar(x_pos, values, color=colors, edgecolor='black')

        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=17)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='grey')
        ax.xaxis.grid(False)
        ax.margins(y=0.1)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        self._save_fig(fig, filename)

    def plot_sensitivity_grid(self, sensitivity_data, default_params, metric_key, y_label, title, filename):
        param_names = list(sensitivity_data.keys())
        num_params = len(param_names)
        if num_params == 0: return

        ncols = 3
        nrows = (num_params + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), sharey=True)
        fig.suptitle(title, fontsize=18, y=1.02)
        axes = axes.flat

        for i, param_name in enumerate(param_names):
            ax = axes[i]
            data = sensitivity_data[param_name]
            param_range = data['param_range']
            results = data['results']
            x_label = data['x_label']
            default_value = default_params.get(param_name)

            poetic_results = results.get('POETIC', {})
            y_values = [poetic_results.get(p_val, {}).get(metric_key, np.nan) for p_val in param_range]

            style = self._get_model_style_safe('POETIC', 0)
            plot_kwargs = {k: v for k, v in style.items() if v is not None}

            ax.plot(param_range, y_values, **plot_kwargs)

            if default_value is not None and default_value in param_range:
                default_idx = param_range.index(default_value)
                default_y = y_values[default_idx]
                if not np.isnan(default_y):
                    ax.scatter(default_value, default_y, marker='*', s=200, color='gold', edgecolor='black', zorder=10,
                               label=f'Default ({default_value})')
                    ax.legend(fontsize=10)

            ax.set_xlabel(x_label)
            if i % ncols == 0:
                ax.set_ylabel(y_label)

            ax.set_title(f"vs {x_label}", fontsize=15)
            ax.grid(True, linestyle=':', linewidth=0.6, color='grey')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

            if "Rate" in x_label and all(p > 0 for p in param_range) and max(param_range) / min(param_range) > 10:
                ax.set_xscale('log')

        for i in range(num_params, nrows * ncols):
            axes[i].axis('off')

        self._save_fig(fig, filename)

    # --- [MISSING METHOD 1] Macro Performance & Regret ---
    def plot_macro_performance_curves(self, agg_results, title, filename):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # 关键步骤：对模型名称进行排序
        present_models = list(agg_results.keys())
        sorted_models = sorted(present_models,
                               key=lambda x: self.ORDERED_MODELS.index(x) if x in self.ORDERED_MODELS else 999)

        # --- (a) Overall (包含 Optimal) ---
        max_len_1 = 0
        for i, name in enumerate(sorted_models):
            data = agg_results[name].get('cumulative_value_runs', [])
            style = self._get_style(name, i)
            # 确保 Optimal 不加标记，保持干净
            if "Optimal" in name:
                style = style.copy()
                style['marker'] = ''

            curr = self._plot_mean_std_curve(ax1, data, name, style, markevery=None)
            max_len_1 = max(max_len_1, curr)

        ax1.set_title("(a) Overall Performance Comparison")
        ax1.set_ylabel("Mean Cumulative Value")
        if max_len_1 > 0: ax1.set_xlim(left=0, right=max_len_1)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        # 图例：左上角
        ax1.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='black', fancybox=False)

        # --- (b) Zoomed-in (不含 Optimal) ---
        max_len_2 = 0
        for i, name in enumerate(sorted_models):
            if "Optimal" in name: continue

            data = agg_results[name].get('cumulative_value_runs', [])
            style = self._get_style(name, i)

            curr = self._plot_mean_std_curve(ax2, data, name, style, markevery=None)
            max_len_2 = max(max_len_2, curr)

        ax2.set_title("(b) Zoomed-in Comparison")
        ax2.set_ylabel("Mean Cumulative Value")
        ax2.set_xlabel("Rounds")
        if max_len_2 > 0: ax2.set_xlim(left=0, right=max_len_2)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        # 图例：左上角
        ax2.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='black', fancybox=False)

        plt.subplots_adjust(hspace=0.2)
        self._save_fig(fig, filename)
    # --- [MISSING METHOD 2] Scalability vs N ---
    # src/visualization.py



    def plot_cumulative_regret_vs_rounds(self, agg_results, title, filename):
        # 【修改】调整尺寸为 (10, 7)，更扁平，适合展示随时间增长的趋势
        fig, ax = plt.subplots(figsize=(10, 7))

        model_names = list(agg_results.keys())
        max_len = 0

        for i, name in enumerate(model_names):
            if "Optimal" in name: continue

            runs = agg_results[name].get('cumulative_value_runs', [])  # Regret Data
            base_style = self.config.MODEL_STYLES.get(name, {})
            style = {
                'color': base_style.get('color', self.palette[i % len(self.palette)]),
                'linestyle': base_style.get('linestyle', '-'),
                'marker': 'o',
                'zorder': 5,
                'linewidth': 2.5,
                'markersize': 8
            }
            curr = self._plot_mean_std_curve(ax, runs, name, style, markevery=None)
            max_len = max(max_len, curr)

        ax.set_title(title, fontsize=22, pad=12)
        ax.set_xlabel("Rounds", fontsize=20)
        ax.set_ylabel("Cumulative Regret", fontsize=20)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        if max_len > 0: ax.set_xlim(0, max_len)
        ax.legend(loc='upper left', fontsize=16, frameon=True, edgecolor='black', framealpha=0.9)

        self._save_fig(fig, filename)

        # src/visualization.py

        # --- 修复 1: Exp 6.3 Scalability (图例上移至右侧中间) ---

    def plot_final_value_vs_n(self, mean_results_dict, n_range, title, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        model_names = list(mean_results_dict.keys())
        sorted_models = sorted(model_names,
                               key=lambda x: self.ORDERED_MODELS.index(x) if x in self.ORDERED_MODELS else 999)

        for i, name in enumerate(sorted_models):
            if name not in mean_results_dict: continue
            data_map = mean_results_dict[name]
            y_values = []
            valid_n = []
            for n in n_range:
                if n in data_map:
                    val = data_map[n].get('final_cumulative_value', np.nan)
                    y_values.append(val)
                    valid_n.append(n)

            style = self._get_style(name, i)
            ax.plot(valid_n, y_values, label=name, **style)

        ax.set_xlabel("Number of Workers (N)")
        ax.set_ylabel("Final Cumulative Value")
        ax.set_title(title, fontsize=16)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        # [精准调整]: bbox_to_anchor=(0.98, 0.82)
        # 0.82 的高度意味着比中间(0.5)高出很多，正好卡在最上面的黑线和中间的红线之间
        ax.legend(loc='center right', bbox_to_anchor=(0.98, 0.72), frameon=True, edgecolor='black', fontsize=10)

        self._save_fig(fig, filename)

        # --- 修复 2: Exp 6.6 Efficiency (图例改到左下角) ---

    def plot_efficiency_metric(self, mean_results_dict, x_values, metric_key, y_label, title, filename, x_label):
        fig, ax = plt.subplots(figsize=(8, 6))
        model_names = list(mean_results_dict.keys())
        sorted_models = sorted(model_names,
                               key=lambda x: self.ORDERED_MODELS.index(x) if x in self.ORDERED_MODELS else 999)

        for i, model_name in enumerate(sorted_models):
            if model_name not in mean_results_dict: continue
            results_over_x = mean_results_dict[model_name]
            style = self._get_style(model_name, i)
            avg_metrics = [results_over_x.get(x, {}).get(metric_key, np.nan) for x in x_values]
            ax.plot(x_values, avg_metrics, label=model_name, **style)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # [修复]: Exp 6.6 (k) 改为 'lower left' (左下角)
        # Exp 6.4/6.5 (N) 如果也调用此函数，可能需要根据 title 判断，但通常左下角是安全的
        loc_param = 'lower left' if 'vs k' in title else 'upper left'
        ax.legend(loc=loc_param, frameon=True, edgecolor='black', fontsize=10)

        ax.grid(True, which='both', linestyle=':', linewidth=0.6, color='grey')
        self._save_fig(fig, filename)

        # --- 修复 3: Exp 11 Regret Growth (图例改到左上角) ---

    def plot_line_chart(self, mean_results_dict, x_values, metric_key, y_label, title, filename, x_label):
        fig, ax = plt.subplots(figsize=(8, 6))
        model_names = list(mean_results_dict.keys())
        sorted_models = sorted(model_names,
                               key=lambda x: self.ORDERED_MODELS.index(x) if x in self.ORDERED_MODELS else 999)

        for i, model_name in enumerate(sorted_models):
            if model_name not in mean_results_dict: continue
            results_over_x = mean_results_dict[model_name]
            style = self._get_style(model_name, i)
            values = [results_over_x.get(x, {}).get(metric_key, np.nan) for x in x_values]
            ax.plot(x_values, values, label=model_name, **style)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=16)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        # [修复]: Regret 曲线随 T 减小或平缓，左上角通常空白
        ax.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=10)

        ax.grid(True, linestyle=':', linewidth=0.6, color='grey')
        self._save_fig(fig, filename)
