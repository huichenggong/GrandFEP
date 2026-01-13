from typing import Union
from pathlib import Path
import re

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy import isfinite

class GC_Log:
    def __init__(self, logfile):
        self.logfile = logfile
        self.data = {}
        self.text = []
        if isinstance(logfile, str) or isinstance(logfile, Path):
            with open(logfile, "r") as f:
                self.text = f.read()
        else:
            for log in logfile:
                with open(log, "r") as f:
                    self.text.append(f.read())
            self.text = "\n".join(self.text)

    
    def get_particle_N(self, cutoff=100):
        """
        Get the number of particles from the log file.
        """
        N_list = [int(x) for x in re.findall(r'\bN=(\d+)\b', self.text)]
        N_list = np.array(N_list)
        x = np.arange(len(N_list))
        map = N_list < cutoff
        x_sphere = x[map]
        N_sphere = N_list[map]

        x_box = x[~map]
        N_box = N_list[~map]
        return (x_sphere, N_sphere), (x_box, N_box)
    
    def get_lambda_states_index(self):
        """
        Get the lambda states index from the log file.
        """
        vals = [int(x) for x in re.findall(r"lambda state index:\s*(\d+)", self.text)]
        return np.array(vals)

def read_bar(mbar_log):
    with open(mbar_log) as f:
        lines = f.readlines()
    if len(lines) > 0 and "Total" in lines[-1]:
        words = lines[-1].split()
    else:
        raise ValueError(f"Cannot find Total line in {mbar_log}")
    return float(words[2]), float(words[4])

def read_bar_multi(bar_log_list):
    res_list = [read_bar(f)  for f in bar_log_list]
    dg     = [res[0] for res in res_list]
    dg_err = [res[1] for res in res_list]
    return dg, dg_err

def read_ddG_edges(edges, base: Path, ff_leg_prot_rep_postfix: str|list) -> pd.DataFrame: 
    """
    read ddG for a certain

    Parameters
    ----------
    edges :
        Edges that can is iterable.

    base :
        pathlib.Path object

    ff_leg_prot_rep_postfix :
        If a list is given, sem will be used as the ddG error. If not, the MBAR error will be used.

    Returns
    -------
    ddG
        A DataFrame that has ddG and ddG_err
    """
    ddG_list = []
    ddG_err_list = []
    if isinstance(ff_leg_prot_rep_postfix, str):
        for edge in edges:
            mbar_log = base / edge / ff_leg_prot_rep_postfix
            ddG, ddG_err = read_bar(mbar_log)
            ddG_list.append(ddG)
            ddG_err_list.append(ddG_err)
    elif isinstance(ff_leg_prot_rep_postfix, list):
        for edge in edges:
            mbar_log_list = [base / edge / postfix for postfix in ff_leg_prot_rep_postfix]
            ddG_vals, ddG_errs = read_bar_multi(mbar_log_list)
            ddG_list.append(np.mean(ddG_vals))
            ddG_err_list.append(scipy.stats.sem(ddG_vals))
    else:
        raise ValueError("ff_leg_prot_rep_postfix must be str or list")
    ddG_df = pd.DataFrame({
        "ddG": ddG_list,
        "ddG_err": ddG_err_list
    }, index=edges)
    return ddG_df

def prepare_cinnabar_csv(edge_df: pd.DataFrame, exp_df: pd.DataFrame, protein_df: pd.DataFrame, water_df: pd.DataFrame, output_csv: Union[Path, str] = "tmp.csv"):
    """
    Prepare a CSV file for Cinnabar Cycle Closure.
    
    Parameters
    ----------
    edge_df :
        This DataFrame should contain edge_name as index, and Lig_1, Lig_2

    exp_df :
        This DataFrame should contain Lig_Name as index, and Exp_dG

    protein_df :
        This DataFrame should contain edge_name as index, and ddG, ddG_err

    water_df :
        This DataFrame should contain edge_name as index, and ddG, ddG_err

    output_csv :
        Path to the output CSV file.

    """
    lines = [
        "# Experimental block\n",
        "# Ligand, expt_DDG, expt_dDDG\n"
    ]

    # Experimental data
    for ligand, row in exp_df.iterrows():
        lines.append(f"{ligand},{row['Exp_dG']},0.1\n")
    
    # MD data
    lines.append("# Calculated block\n")
    lines.append("# Ligand1,Ligand2, calc_DDG, calc_dDDG(MBAR), calc_dDDG(additional)\n")
    for edge_name in protein_df.index:
        lig1 = edge_df.loc[edge_name, "Lig_1"]
        lig2 = edge_df.loc[edge_name, "Lig_2"]
        ddG = protein_df.loc[edge_name, "ddG"] - water_df.loc[edge_name, "ddG"]
        ddG_err = np.sqrt(protein_df.loc[edge_name, "ddG_err"]**2 + water_df.loc[edge_name, "ddG_err"]**2)
        lines.append(f"{lig1},{lig2},{ddG},{ddG_err},0.1\n")
    
    with open(output_csv, "w") as f:
        f.writelines(lines)
    return lines


### Plotting functions
def uniform_xylim(axes, ticks=0, aspect_equal=True):
    """
    Set identical x/y limits on all given Matplotlib axes, with padding.

    Parameters
    ----------
    axes : Iterable[matplotlib.axes.Axes] or Axes
        One Axes or any iterable (list/array) of Axes.
    ticks : float, optional
        x/y ticks label interval. Default is 1.

    Notes
    -----
    - Calls relim()/autoscale_view() on each axes so collections (e.g., scatter)
      are included when computing global limits.
    - Preserves axis inversion (if an axes was inverted, it'll stay inverted).
    """
    # normalize to a flat list of Axes
    if isinstance(axes, Axes):
        ax_list = [axes]
    else:
        # flatten e.g. numpy array of axes
        ax_list = axes.reshape(-1).tolist()

    if not ax_list:
        return

    # Collect global data bounds
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for ax in ax_list:
        x1,x2 = ax.get_xlim()
        y1,y2 = ax.get_ylim()
        xmins.append(x1); xmaxs.append(x2)
        ymins.append(y1); ymaxs.append(y2)

    xmin = np.min(xmins); xmax = np.max(xmaxs)
    ymin = np.min(ymins); ymax = np.max(ymaxs)

    if aspect_equal:
        xmin = min(xmin, ymin)
        xmax = max(xmax, ymax)
        ymin = xmin
        ymax = xmax
        for ax in ax_list:
            ax.set_aspect("equal")

    # Set ticks
    
    if not ticks is None:
        if ticks == 0:
            # about 7 ticks, and integer labels
            span = max(xmax - xmin, ymax - ymin)
            ticks = max(1, np.ceil(span / 7))

        x_ticks = np.arange(np.floor(xmin), np.ceil(xmax)+ticks, ticks)
        y_ticks = np.arange(np.floor(ymin), np.ceil(ymax)+ticks, ticks)
        for ax in ax_list:
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

    # Set on all axes, preserving inversion
    for ax in ax_list:
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    
    
    return (xmax, xmin), (ymax, ymin)

def add_gray_1kcal(axes, alpha=0.3, center_line=True):
    """
    Add gray area for y = x ± 1 kcal/mol error.

    Parameters
    ----------
    axes : Iterable[matplotlib.axes.Axes] or Axes
        One Axes or any iterable (list/array) of Axes.
    """
    # normalize to a flat list of Axes
    if isinstance(axes, Axes):
        ax_list = [axes]
    else:
        # flatten e.g. numpy array of axes
        ax_list = axes.reshape(-1).tolist()

    if not ax_list:
        return

    for ax in ax_list:
        x = np.array(ax.get_xlim())
        ax.fill_between(
            x=x,
            y1=x - 1,
            y2=x + 1,
            color="black",
            alpha=alpha,
            zorder=0,
        )
        if center_line:
            ax.plot(x, x, color="black")

# Statistical functions
def rmsd_stat(exp, com, axis=0):
    return np.sqrt(np.mean((com - exp)**2, axis=axis))

def kendall_stat(exp, com):
    tau = scipy.stats.kendalltau(exp, com).correlation
    return tau

def r_squared_stat(exp, com):
    correlation_matrix = np.corrcoef(exp, com)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    return r_squared

def get_stat(com, exp, func_list, n_resamples=5000):
    results = []
    for func in func_list:
        res, res_bs = get_stat_bootstrap(com, exp, func, n_resamples=n_resamples)
        results.append((res, res_bs))
    return results

def get_stat_bootstrap(com_arr, exp_arr, func, n_resamples=5000, confidence_level=0.95, method="BCa", random_state=None):
    """
    Get bootstrap statistics for a given statistic function.

    Parameters
    ----------
    com_arr :
        Computed values array.
    exp_arr :
        Experimental values array.
    func :
        Statistic function that takes (exp, com) as input.
    n_resamples :
        Number of bootstrap resamples. Default is 5000.
    confidence_level :
        Confidence level for the interval. Default is 0.95.
    method :
        Bootstrap method. Default is "BCa".
    random_state :
        Random state for reproducibility. Default is None.
    
    Returns
    -------
    res_point :
        Statistic value for the original data.
    res :
        Bootstrap result object.

    """
    exp = np.asarray(exp_arr, float)
    com = np.asarray(com_arr, float)
    res_point = func(com_arr, exp_arr)
    res = scipy.stats.bootstrap(
        data=(exp, com),
        statistic=func,
        paired=True,
        vectorized=False,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,            # fallback to "percentile" if your SciPy is older
        random_state=random_state,
    )
    return res_point, res

def paired_bootstrap_delta(exp, com1, com2, stat=rmsd_stat,
                           n_resamples=5000, confidence_level=0.95,
                           method="BCa", random_state=None):
    """
    Paired bootstrap for Δ = stat(exp, com1) - stat(exp, com2).

    Returns
    -------
    delta_point : float
        Point estimate on the original data.
    ci : (float, float)
        Lower and upper bounds of the bootstrap CI.
    boot : scipy.stats.BootstrapResult or dict
        Bootstrap result (SciPy object or a dict if manual fallback used).
    """
    exp  = np.asarray(exp,  float)
    com1 = np.asarray(com1, float)
    com2 = np.asarray(com2, float)

    # point estimate
    delta_point = stat(exp, com1) - stat(exp, com2)

    # SciPy paired bootstrap (resamples shared indices across all three arrays)
    def delta_stat(e, c1, c2):
        return stat(e, c1) - stat(e, c2)

    
    boot = scipy.stats.bootstrap(
        data=(exp, com1, com2),
        statistic=delta_stat,
        paired=True,
        vectorized=False,        # statistic is scalar, not vectorized
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,           # 'BCa' if available; else user can pass 'percentile'
        random_state=random_state,
    )
    return delta_point, boot
    
