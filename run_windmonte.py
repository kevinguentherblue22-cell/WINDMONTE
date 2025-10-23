"""run_windmonte.py

Monte Carlo runner that perturbs the measured inputs (NF, AF, Theta)
and computes distributions for CL and CD using the existing DREs wrapper.

This file is intentionally created but not executed.

Usage (PowerShell):
& ".\.venv\Scripts\python.exe" .\run_windmonte.py --n-samples 1000

Outputs:
- CL_vs_AOA_with_uncertainty.png
- windmonte_results.npz (contains CL_samples, Theta_vals, mean/CI arrays)
"""

'''import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import DREs


def run_monte_carlo(input_pickle, n_samples=1000, sigma_nf=0.5, sigma_af=0.2, sigma_theta=0.1, out_plot="CL_vs_AOA_with_uncertainty.png", out_npz="windmonte_results.npz", seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Load inputs
    with open(input_pickle, "rb") as f:
        data, G = pickle.load(f)

    num_records = len(data)
    Theta_vals = np.array([rec['Theta'] for rec in data])

    CL_samples = np.zeros((n_samples, num_records))
    CD_samples = np.zeros((n_samples, num_records))

    for s in tqdm(range(n_samples), desc="MC samples"):
        sample = []
        for rec in data:
            NF = rec.get('NF', 0) + np.random.normal(0, sigma_nf)
            AF = rec.get('AF', 0) + np.random.normal(0, sigma_af)
            Theta = rec.get('Theta', 0) + np.random.normal(0, sigma_theta)
            samp_rec = dict(rec)
            samp_rec.update({'NF': NF, 'AF': AF, 'Theta': Theta})
            sample.append(samp_rec)

        D = DREs.eval(sample, G)
        CL_samples[s, :] = [d['CL'] for d in D]
        CD_samples[s, :] = [d['CD'] for d in D]

    mean_CL = CL_samples.mean(axis=0)
    low_CL = np.percentile(CL_samples, 2.5, axis=0)
    high_CL = np.percentile(CL_samples, 97.5, axis=0)

    mean_CD = CD_samples.mean(axis=0)
    low_CD = np.percentile(CD_samples, 2.5, axis=0)
    high_CD = np.percentile(CD_samples, 97.5, axis=0)

    # Save results
    np.savez(out_npz, CL_samples=CL_samples, CD_samples=CD_samples, Theta_vals=Theta_vals,
             mean_CL=mean_CL, low_CL=low_CL, high_CL=high_CL,
             mean_CD=mean_CD, low_CD=low_CD, high_CD=high_CD)

    # Plot CL
    plt.figure(figsize=(8, 5))
    plt.plot(Theta_vals, mean_CL, '-o', label='mean CL')
    plt.fill_between(Theta_vals, low_CL, high_CL, color='C0', alpha=0.2, label='95% CI')
    plt.xlabel('Angle of Attack (deg)')
    plt.ylabel('CL')
    plt.title('CL vs AOA with 95% CI')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_plot, dpi=200)
    plt.close()

    # Plot CD (optional)
    plt.figure(figsize=(8,5))
    plt.plot(Theta_vals, mean_CD, '-o', label='mean CD')
    plt.fill_between(Theta_vals, low_CD, high_CD, color='C1', alpha=0.2, label='95% CI')
    plt.xlabel('Angle of Attack (deg)')
    plt.ylabel('CD')
    plt.title('CD vs AOA with 95% CI')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_plot.replace('CL_','CD_') if 'CL_' in out_plot else 'CD_vs_AOA_with_uncertainty.png', dpi=200)
    plt.close()

    return {
        'CL_samples': CL_samples,
        'CD_samples': CD_samples,
        'Theta_vals': Theta_vals,
        'mean_CL': mean_CL,
        'low_CL': low_CL,
        'high_CL': high_CL,
        'mean_CD': mean_CD,
        'low_CD': low_CD,
        'high_CD': high_CD,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Monte Carlo propagation for WINDMONTE')
    parser.add_argument('--input', default='wind_forces_inputs.pkl', help='Path to inputs pickle')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of Monte Carlo samples')
    parser.add_argument('--sigma-nf', type=float, default=0.5, help='Stddev for NF perturbation')
    parser.add_argument('--sigma-af', type=float, default=0.2, help='Stddev for AF perturbation')
    parser.add_argument('--sigma-theta', type=float, default=0.1, help='Stddev for Theta perturbation (deg)')
    parser.add_argument('--out-plot', default='CL_vs_AOA_with_uncertainty.png')
    parser.add_argument('--out-npz', default='windmonte_results.npz')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    run_monte_carlo(args.input, n_samples=args.n_samples, sigma_nf=args.sigma_nf, sigma_af=args.sigma_af,
                    sigma_theta=args.sigma_theta, out_plot=args.out_plot, out_npz=args.out_npz, seed=args.seed)
'''