import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def load_inputs():
    if not os.path.exists('wind_forces_inputs.pkl'):
        raise FileNotFoundError('wind_forces_inputs.pkl not found; run EQs.py to generate inputs')
    with open('wind_forces_inputs.pkl', 'rb') as f:
        data, testinfo = pickle.load(f)
    return data, testinfo


def compute_dre(data, testinfo=None):
    # Use local DRE implementation if available
    try:
        import DRE_WM
        out = DRE_WM.DREs(data, testinfo if isinstance(testinfo, dict) else None)
    except Exception:
        # fallback to wrapper module DREs if present
        import DREs as drewrap
        out = drewrap.eval(data, testinfo)
    return out


def try_load_runresults():
    if not os.path.exists('output_data.pkl'):
        return None
    try:
        with open('output_data.pkl', 'rb') as f:
            RunData, U_systematic, U_random = pickle.load(f)
        return RunData
    except Exception:
        return None


def make_plots():
    data, testinfo = load_inputs()
    dre_out = compute_dre(data, testinfo)

    # extract nominal values
    thetas = np.array([pt.get('Theta', np.nan) for pt in dre_out], dtype=float)
    CLs = np.array([pt.get('CL', np.nan) for pt in dre_out], dtype=float)
    CDs = np.array([pt.get('CD', np.nan) for pt in dre_out], dtype=float)

    RunData = try_load_runresults()

    # Prepare uncertainty arrays if RunData available
    if RunData is not None:
        CL_u_neg = []
        CL_u_pos = []
        CD_u_neg = []
        CD_u_pos = []
        for i in range(len(dre_out)):
            try:
                voi_CL = RunData[i]['CL']
                voi_CD = RunData[i]['CD']
                # u_low is (r_low - nom) often negative, u_high positive
                cl_neg = abs(voi_CL.u_low) if voi_CL.u_low is not None else 0.0
                cl_pos = abs(voi_CL.u_high) if voi_CL.u_high is not None else 0.0
                cd_neg = abs(voi_CD.u_low) if voi_CD.u_low is not None else 0.0
                cd_pos = abs(voi_CD.u_high) if voi_CD.u_high is not None else 0.0
            except Exception:
                cl_neg = cl_pos = cd_neg = cd_pos = 0.0
            CL_u_neg.append(cl_neg)
            CL_u_pos.append(cl_pos)
            CD_u_neg.append(cd_neg)
            CD_u_pos.append(cd_pos)
        CL_u_neg = np.array(CL_u_neg)
        CL_u_pos = np.array(CL_u_pos)
        CD_u_neg = np.array(CD_u_neg)
        CD_u_pos = np.array(CD_u_pos)
    else:
        CL_u_neg = CL_u_pos = CD_u_neg = CD_u_pos = None

    # Plot 1: Drag polar (CL vs CD)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(CDs, CLs, marker='o', linestyle='-', color='C0', label='Nominal')
    ax1.set_xlabel('C_D')
    ax1.set_ylabel('C_L')
    ax1.set_title('Drag polar: C_L vs C_D')
    if RunData is not None:
        # asymmetric errorbars: shape (2, N)
        xerr = np.vstack([CD_u_neg, CD_u_pos])
        yerr = np.vstack([CL_u_neg, CL_u_pos])
        ax1.errorbar(CDs, CLs, xerr=xerr, yerr=yerr, fmt='o', ecolor='red', color='red', label='Uncertainty (MC)', capsize=3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig('Midterm_CL_vs_CD.png', dpi=200)

    # Plot 2: CL vs Theta
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(thetas, CLs, marker='o', linestyle='-', color='C0', label='Nominal')
    ax2.set_xlabel('Theta (deg)')
    ax2.set_ylabel('C_L')
    ax2.set_title('C_L vs Theta')
    if RunData is not None:
        yerr = np.vstack([CL_u_neg, CL_u_pos])
        ax2.errorbar(thetas, CLs, yerr=yerr, fmt='o', ecolor='red', color='red', label='Uncertainty (MC)', capsize=3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig('Midterm_CL_vs_Theta.png', dpi=200)

    print('Saved Midterm_CL_vs_CD.png and Midterm_CL_vs_Theta.png')
    plt.show()


if __name__ == '__main__':
    make_plots()
