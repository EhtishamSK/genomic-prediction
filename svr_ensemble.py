# ==============================================================================
# Script Name: svr_ensemble.py
# SVR – 10-run ensemble (80:20) with full outputs
# ==============================================================================

import os, sys, time, traceback, argparse
from datetime import datetime
import pyreadr, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TRAIT_DESCRIPTIONS = {
    'DSFG':'Destemming force for green fruits (DSFG)', 'DSFR':'Destemming force for red fruits (DSFR)',
    'DSF':'Fruit Destemming Force (DSF)', 'DSRG':'Destemming rating for green fruits (DSRG)',
    'DSRR':'Destemming rating for red fruits (DSRR)', 'GRN':'Green yield (GRN)',
    'RED':'Red yield (RED)', 'TYLD':'Total yield (TYLD)'
}

parser = argparse.ArgumentParser()
parser.add_argument("--trait", required=True)
parser.add_argument("--project_dir", default=None)
parser.add_argument("--n_runs", type=int, default=10)
args = parser.parse_args()

base_dir = os.path.expanduser("~/prediction") if args.project_dir is None else os.path.abspath(args.project_dir)
trait = args.trait
trait_title = TRAIT_DESCRIPTIONS.get(trait, trait)
n_runs = args.n_runs

input_dir   = os.path.join(base_dir, "input", trait)
output_base = os.path.join(base_dir, "output", trait, "svr")
os.makedirs(output_base, exist_ok=True)

start_time = time.time()
start_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_lines, successful_runs, failed_runs, all_obs_pred = [], [], [], []

def log(msg):
    print(msg)
    log_lines.append(msg)

log("="*70)
log("=== [SVR – 10-RUN ENSEMBLE – 80:20] ===")
log(f"Start: {start_str} | Trait: {trait_title}")
log("="*70)

try:
    geno_train_df  = pyreadr.read_r(os.path.join(input_dir, "geno_train_final.rds"))[None]
    pheno_train_df = pyreadr.read_r(os.path.join(input_dir, "pheno_train_final.rds"))[None]
    geno_test_df   = pyreadr.read_r(os.path.join(input_dir, "geno_test_final.rds"))[None]
    pheno_test_df  = pyreadr.read_r(os.path.join(input_dir, "pheno_test_final.rds"))[None]

    y_train = pheno_train_df[trait].values.astype(np.float32)
    y_test  = pheno_test_df[trait].values.astype(np.float32)

    snp_cols = [c for c in geno_train_df.columns if c not in ['geno','Year',trait]]
    geno_train_mat = geno_train_df[snp_cols].values.astype(np.float32)
    geno_test_mat  = geno_test_df.reindex(columns=snp_cols, fill_value=np.nan).values.astype(np.float32)

    if np.isnan(geno_train_mat).any() or np.isnan(geno_test_mat).any():
        col_means = np.nanmean(geno_train_mat, axis=0)
        for mat in (geno_train_mat, geno_test_mat):
            inds = np.where(np.isnan(mat))
            mat[inds] = np.take(col_means, inds[1])

    if 'Year' in pheno_train_df.columns:
        year_train = pd.get_dummies(pheno_train_df['Year'].astype(str), prefix='Year', dtype=float)
        year_test  = pd.get_dummies(pheno_test_df['Year'].astype(str),  prefix='Year', dtype=float)
        year_train, year_test = year_train.align(year_test, join='outer', axis=1, fill_value=0)
        year_train = year_train.values.astype(np.float32)
        year_test  = year_test.values.astype(np.float32)
    else:
        year_train = np.zeros((len(y_train),0), dtype=np.float32)
        year_test  = np.zeros((len(y_test),0), dtype=np.float32)

    X_train = np.hstack([geno_train_mat, year_train]).astype(np.float32)
    X_test  = np.hstack([geno_test_mat, year_test]).astype(np.float32)

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled  = x_scaler.transform(X_test)
    log(f"Data loaded | X_train: {X_train.shape}")

except Exception as e:
    log(f"FATAL: {e}\n{traceback.format_exc()}")
    with open(os.path.join(output_base, f"{trait}_SVR_ensemble_run_log.txt"),"w") as f: f.write("\n".join(log_lines))
    sys.exit(1)

all_metrics, all_gebvs = [], []

for run in range(1, n_runs+1):
    log(f"\n=== RUN {run}/{n_runs} ===")
    try:
        seed = 42 + run
        np.random.seed(seed)
        run_dir = os.path.join(output_base, f"run{run}")
        os.makedirs(run_dir, exist_ok=True)

        model = SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        # GEBVs (Year zeroed)
        if year_test.shape[1] > 0:
            X_gebv = np.hstack([geno_test_mat, np.zeros_like(year_test)])
        else:
            X_gebv = geno_test_mat
        X_gebv_scaled = x_scaler.transform(X_gebv)
        gebv = model.predict(X_gebv_scaled)

        gebv_df = pd.DataFrame({'geno': pheno_test_df['geno'], 'GEBV_SVR': gebv}).groupby('geno', as_index=False).mean()
        gebv_df.to_csv(os.path.join(run_dir, f"{trait}_SVR_GEBV_rankings.csv"), index=False)

        r_val = pearsonr(y_test, y_pred)[0]
        r2_val = r2_score(y_test, y_pred)
        rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics_df = pd.DataFrame([{'Run':run,'Model':'SVR','Trait':trait,
                                    'Predictability (r)':r_val,'R2':r2_val,'RMSE':rmse_val}])
        metrics_df.to_csv(os.path.join(run_dir, f"{trait}_SVR_prediction_metrics.csv"), index=False)
        log(f"r={r_val:.4f} | R²={r2_val:.4f} | RMSE={rmse_val:.4f}")

        all_metrics.append(metrics_df)
        all_gebvs.append(gebv_df.set_index('geno')['GEBV_SVR'])
        successful_runs.append(run)

        obs_pred_df = pd.DataFrame({
            'geno': pheno_test_df['geno'], 'Observed': y_test, 'Predicted': y_pred,
            'Residual': y_test - y_pred,
            'Year': pheno_test_df['Year'] if 'Year' in pheno_test_df.columns else "NA",
            'Run': run, 'Model': 'SVR', 'Trait': trait
        })
        obs_pred_df.to_csv(os.path.join(run_dir, f"{trait}_SVR_observed_vs_predicted.csv"), index=False)
        all_obs_pred.append(obs_pred_df)

        # Plots
        sns.set_theme(style="whitegrid", font_scale=1.1)
        df_plot = obs_pred_df.copy(); df_plot['Year'] = df_plot['Year'].astype(str)
        year_colors = {'2022':'#ff7f7e','2023':'#2ca02c','2025':'#5b9bd5'}

        fig, ax = plt.subplots(figsize=(8,5))
        sns.regplot(data=df_plot, x='Predicted', y='Observed', ax=ax, scatter=False, color='red', ci=95)
        sns.scatterplot(data=df_plot, x='Predicted', y='Observed', hue='Year', palette=year_colors, s=70, ax=ax)
        lims = [min(df_plot[['Predicted','Observed']].min())-3, max(df_plot[['Predicted','Observed']].max())+3]
        ax.plot(lims, lims, 'k--'); ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f"{trait_title}\nSVR – Observed vs Predicted (Run {run})")
        plt.tight_layout(); plt.savefig(os.path.join(run_dir, f"{trait}_SVR_observed_vs_predicted.png"), dpi=300); plt.close()

        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(data=df_plot, x='Predicted', y='Residual', hue='Year', palette=year_colors, s=70, ax=ax)
        ax.axhline(0, color='black')
        ax.set_title(f"{trait_title}\nSVR – Residuals (Run {run})")
        plt.tight_layout(); plt.savefig(os.path.join(run_dir, f"{trait}_SVR_residuals_vs_predicted.png"), dpi=300); plt.close()

    except Exception as e:
        log(f"RUN {run} FAILED: {e}\n{traceback.format_exc()}")
        failed_runs.append((run, str(e)))

if all_metrics:
    metrics_all = pd.concat(all_metrics, ignore_index=True)
    summary = metrics_all[['Predictability (r)','R2','RMSE']].agg(['mean','std']).T
    summary.columns = ['Mean','SD']
    summary.to_csv(os.path.join(output_base, f"{trait}_SVR_ensemble_summary.csv"))
    log(summary.to_string())

    gebv_ens = pd.concat(all_gebvs, axis=1).mean(axis=1).reset_index()
    gebv_ens.columns = ['geno','GEBV_SVR_Ensemble']
    gebv_ens.to_csv(os.path.join(output_base, f"{trait}_SVR_GEBV_ensemble_average.csv"), index=False)

    pd.concat(all_obs_pred, ignore_index=True).to_csv(
        os.path.join(output_base, f"{trait}_SVR_all_runs_observed_vs_predicted.csv"), index=False)

log(f"\nFinished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Successful: {successful_runs} | Failed: {failed_runs}")
with open(os.path.join(output_base, f"{trait}_SVR_ensemble_run_log.txt"), "w") as f:
    f.write("\n".join(log_lines))
print("=== SVR 80:20 ensemble finished ===")
