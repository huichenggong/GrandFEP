#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import sys

import re
import yaml
import numpy as np
import pandas as pd

def lambdaT_2_lambda_chg_vdw(lambda_total):
    """
    Given a lambda_total (0,3)
    
    return
        l_chg_delete
        l_vdw_delete       
        l_vdw_insert       
        l_chg_insert
    """
    l_chg_delete = np.clip(lambda_total * 3,     0, 1)
    l_vdw_delete = np.clip(lambda_total * 3 - 1, 0, 1)
    l_vdw_insert = np.clip(lambda_total * 3 - 1, 0, 1)
    l_chg_insert = np.clip(lambda_total * 3 - 2, 0, 1)
    return l_chg_delete, l_vdw_delete, l_vdw_insert, l_chg_insert

def lambda_chg_vdw_2_lambdaT(l_chg_delete, l_vdw, l_chg_insert):
    """
    """
    lambda_total = (np.array(l_chg_delete) + np.array(l_vdw) + np.array(l_chg_insert)) / 3
    return lambda_total

def k_rest2_schedule(nwin, rest2_scale, method="linear_H"):
    """
    Generate a per-window REST2 scaling schedule of length ``nwin``.

    In REST2, each replica scales a subset of its Hamiltonian by ``k_rest2``.
    A value of 1.0 means no scaling (endpoints, fully coupled/decoupled states);
    values below 1.0 lower the effective barrier for the selected degrees of
    freedom, enhancing sampling in the intermediate windows.

    The schedule is symmetric: ``k_rest2[0] = k_rest2[-1] = 1.0`` and the
    minimum value ``rest2_scale`` is reached near the midpoint.

    Parameters
    ----------
    nwin : int
        Number of replica-exchange windows.
    rest2_scale : float
        Minimum value of ``k_rest2``, reached at the midpoint window.
        Must be in (0, 1]. Use 1.0 to produce a flat schedule (no REST2).
    method : str, optional
        Shape of the schedule. Four options:

        ``"linear_H"`` *(default)*
            Tent shape that is **linear in Hamiltonian**:

        ``"square_H"``
            **Parabolic in Hamiltonian**: ``k_rest2 = 1 - c·x·(1-x)``

        ``"linear_T"``
            Tent shape that is **linear in temperature-space**: the effective
            temperature ``T_eff ∝ 1/k_rest2`` increases linearly from 1 at
            both endpoints to ``1/rest2_scale`` at the midpoint.

        ``"square_T"``
            **Parabolic in temperature-space**: `T_eff = 1 + c·x·(1-x)``
            where ``c`` is chosen so that ``T_eff`` at the midpoint equals
            ``1/rest2_scale``. Equivalent to ``k_rest2 = 1/(1 + c·x·(1-x))``.

    Returns
    -------
    k_rest2 : numpy.ndarray, shape (nwin,)
        Scaling factors for each window. All values are in ``[rest2_scale, 1]``.
    """
    if method=="linear_H":
        x = np.linspace(0, 1, nwin)
        x_mid = x[nwin // 2 - 1]
        rest2_scale_min = 1 - (1 - rest2_scale) / x_mid * 0.5
        k_rest2 = 1 + (1 - rest2_scale_min) * np.maximum(-2 * x, 2 * (x - 1))
    elif method == "square_H":
        x = np.linspace(0, 1, nwin)
        k_rest2 = x * (1 - x)
        k_rest2 *= (1 - rest2_scale) / k_rest2.max()
        k_rest2 = 1 - k_rest2
    elif method == "linear_T":
        x = np.linspace(0, 1, nwin)
        y = np.minimum(2 * x, 2 * (1 - x))
        y *= (1/rest2_scale-1) / y.max()
        y += 1
        k_rest2 = 1/y
    elif method == "square_T":
        x = np.linspace(0, 1, nwin)
        y = (x * (1 - x))
        y *= (1/rest2_scale-1) / y.max()
        k_rest2 = 1/(y+1)
    else:
        raise NotImplemented(f"{method=} has not been implemented.")
    return k_rest2


def write_yml(yml_output, x_new, rest2_scale=1.0, rest_method="linear", comment="#"):
    nwin = len(x_new)
    l_chg_delete, l_vdw, _, l_chg_insert = lambdaT_2_lambda_chg_vdw(x_new)
    k_rest2 = k_rest2_schedule(nwin, rest2_scale, method=rest_method)
    lines = [
        "# window Indices               " + ",".join([f"{i:14d}"   for i in np.arange(nwin)])         + "\n",
        "lambda_angles               : [" + ",".join([f"{i:14.6f}" for i in x_new]) + "]\n",
        "lambda_bonds                : [" + ",".join([f"{i:14.6f}" for i in x_new]) + "]\n",
        "lambda_sterics_core         : [" + ",".join([f"{i:14.6f}" for i in x_new]) + "]\n",
        "lambda_electrostatics_core  : [" + ",".join([f"{i:14.6f}" for i in x_new]) + "]\n",
        "lambda_torsions             : [" + ",".join([f"{i:14.6f}" for i in x_new]) + "]\n",
        "k_rest2                     : [" + ",".join([f"{i:14.6f}" for i in k_rest2]) + "]\n",
        f"{comment}\n",
        "# lambda_total                 " + ",".join([f"{i:14.11f}" for i in x_new]) + "\n",
        "lambda_electrostatics_delete: [" + ",".join([f"{i:14.11f}" for i in l_chg_delete]) + "]\n",
        "lambda_sterics_delete       : [" + ",".join([f"{i:14.11f}" for i in l_vdw])        + "]\n",
        "lambda_sterics_insert       : [" + ",".join([f"{i:14.11f}" for i in l_vdw])        + "]\n",
        "lambda_electrostatics_insert: [" + ",".join([f"{i:14.11f}" for i in l_chg_insert]) + "]\n",
    ]
    with open(yml_output, "w") as f:
        f.writelines(lines)

def read_overlap_log(bar_log):
    """
    Parse the overlap column from mbar/bar.log.
    Lines of interest look like:  ' 0 - 1 :   2.7141 +-  0.1384 0.06'
    Returns an array of overlaps, one per adjacent window pair.
    """
    overlaps = []
    with open(bar_log) as f:
        for line in f:
            if re.match(r'^\s*\d+ - \d+', line):
                overlaps.append(float(line.split()[-1]))
    return np.array(overlaps)

def read_error_csv(mbar_csv):
    df = pd.read_csv(mbar_csv)
    err = []
    for i in range(len(df)-1):
        err.append(df[f"BAR_{i}_err"][i+1])
    return np.array(err)

def read_lambda_yml(yml_inp):
    with open(yml_inp) as f:
        data = yaml.safe_load(f)
    x_axis = lambda_chg_vdw_2_lambdaT(data["lambda_electrostatics_delete"], data["lambda_sterics_delete"], data["lambda_electrostatics_insert"])
    return x_axis, data
    
def update_every_lambda(x, err, alpha=1.0):
    """
    update every lambda points according to the pair-wise error
    """ 
    

    if not len(x) - len(err) == 1:
        raise ValueError(f"The n(x) should be the n(err) + 1")
    S = x[1:] - x[:-1]
    assert np.all(S>0)

    # target normalised lengths  S' ∝ Err / S 
    target =  S / err
    target /= target.sum()            # normalise to sum to 1
    x_new = np.zeros(len(x))
    x_new[1:] = np.cumsum(target)

    # damping
    x_new = (1 - alpha) * x + alpha * x_new
    return x_new

def update_1step(path_in, path_out, learning_rate_max = 0.5, max_step=0.02, max_ratio=0.5, rest_method="linear_H", rest2_scale=1.0):
    """
    Update lambda points by one step according to the error in mbar/bar.csv
    """
    x_old_list = []

    mbar_csv = path_in / "mbar/bar.csv"
    err = read_error_csv(mbar_csv)
    # fill the nan with the max error
    err = np.array(err)
    nan_mask = np.isnan(err)
    if np.any(nan_mask):
        bar_log = path_in / "mbar/bar.log"
        if bar_log.exists():
            try:
                overlap = read_overlap_log(bar_log)
                if len(overlap) == len(err):
                    scale = np.nanmedian(err * overlap)
                    can_estimate = nan_mask & (overlap > 0)
                    if not np.isnan(scale) and np.any(can_estimate):
                        err[can_estimate] = scale / overlap[can_estimate]
                        print(f"Estimated {can_estimate.sum()} nan error(s) from overlap "
                              f"(scale={scale:.4f}, overlap={overlap[can_estimate]}, err={err[can_estimate]})")
            except Exception as e:
                print(f"Could not read overlap from {bar_log}: {e}")
        # fallback for any remaining nan (overlap=0, bar.log missing, or all errors nan)
        still_nan = np.isnan(err)
        if np.any(still_nan):
            fallback = np.nanmax(err)
            fallback = fallback * 1.3 if not np.isnan(fallback) else 1.0
            err[still_nan] = fallback
            print(f"Filled {still_nan.sum()} nan error(s) with fallback {fallback:.4f}")
    print(f"Err Max/Mean : {max(err):.3f} / {np.mean(err):.3f} = {max(err)/np.mean(err):.2f}")
    update_flag = max(err)/np.mean(err) > 1.4

    x_old, data = read_lambda_yml(path_in / f"0/npt.yml")
    x_old = np.array(x_old)
    
    # update opt step
    lrate = learning_rate_max
    x_new = update_every_lambda(x_old, err, alpha=lrate)
    x_delta = x_new - x_old
        
    # no points should move more than half of the segment
    seg_length = x_old[1:] - x_old[:-1]
    # right move or points 0, 1, 2, 
    max_move_ratio_r =  max(x_delta[:-1]/ seg_length)
    max_move_ratio_l =  max(-x_delta[1:]/ seg_length)
    if max_move_ratio_r > max_ratio:
        print(f"Max R move {max_move_ratio_r:.2f}")
    if max_move_ratio_l > max_ratio:
        print(f"Max L move {max_move_ratio_l:.2f}")
    if max_move_ratio_r > max_ratio or max_move_ratio_l > max_ratio:
        max_move_ratio = max(max_move_ratio_l, max_move_ratio_r)
        lrate *= max_ratio / max_move_ratio
        print(f"New learning rate {lrate}")
        x_new = update_every_lambda(x_old, err, alpha=lrate)
        x_delta = x_new - x_old


    max_step_length = max(np.abs(x_new - x_old))
    if max_step_length > max_step:
        print(f"Max Step {max_step_length:.3f}")
        lrate *=  max_step / max_step_length
        print(f"New learning rate {lrate}")
        x_new = update_every_lambda(x_old, err, alpha=lrate)
        x_delta = x_new - x_old
    
    # write next step
    if update_flag:
        os.makedirs(path_out, exist_ok=True)
        print(path_out / "lam.yml")
        write_yml(path_out / "lam.yml", x_new, rest_method=rest_method, rest2_scale=rest2_scale)

    return x_old_list, x_new
    
if __name__ == "__main__":
    # print Command line arguments
    print(f"Command line arguments: {' '.join(sys.argv)}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,  
                        help="Input directory containing 0/npt.yml and mbar/mbar.csv. If not provided, generate a linear lambda schedule from 0 to 1.")
    parser.add_argument("-o", "--output", type=str, default=".",
                        help="Output directory to write lam.yml")
    parser.add_argument("-nwin", type=int, help="Number of windows (only used if input is not provided)")
    parser.add_argument("-lr", "--learning_rate_max", type=float, default=0.5, help="Maximum learning rate")
    parser.add_argument("--max_step", type=float, default=0.02, help="Maximum step size for any lambda point")
    parser.add_argument("--max_ratio", type=float, default=0.75, help="Maximum ratio of segment length for any lambda point")
    parser.add_argument("--rest2_scale", type=float, default=1.0, help="Scaling factor for REST2 k_rest2 (1.0 means no scaling)")
    parser.add_argument("--rest2_method", type=str, default="linear_H", choices=["linear_H", "square_H", "linear_T", "square_T"],
                        help="Scaling factor for REST2 k_rest2")
    args = parser.parse_args()

    if args.input is not None:
        path_in = Path(args.input)
        path_out = Path(args.output)
    
        x_old_list, x_new = update_1step(path_in, path_out, args.learning_rate_max,
                                         max_step=args.max_step,
                                         max_ratio=args.max_ratio,
                                         rest_method=args.rest2_method,
                                         rest2_scale=args.rest2_scale)
    else:
        if args.nwin is None:
            raise ValueError("Number of windows must be provided if input is not provided.")
        nwin = args.nwin
        x_new = np.linspace(0, 1, nwin)
        path_out = Path(args.output)
        os.makedirs(path_out, exist_ok=True)
        write_yml(path_out / "lam.yml", x_new, rest_method=args.rest2_method, rest2_scale=args.rest2_scale)
