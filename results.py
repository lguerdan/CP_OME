import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style = "whitegrid", rc = {
   "legend.frameon": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Lucida Grande",
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.labelsize": 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 'large'
})

def get_erm_convergence_result_df(exp_name):
  results = []
  for f in glob.glob(f'results/{exp_name}/*PO.csv'):
    results.append(pd.read_csv(f))
  nsdf = pd.concat(results)
  return nsdf, nsdf.groupby(['baseline', 'NS']).mean()


def get_ate_result_df(exp_name):
    results = []
    for f in glob.glob(f'results/{exp_name}/*TE.csv'):
        results.append(pd.read_csv(f))
    nsdf = pd.concat(results)
    nsdf.drop(columns=['Unnamed: 0'], inplace=True)
    nsdf['param'] = "(" + nsdf['alpha_0'].astype(str)+"," +nsdf['beta_0'].astype(str) + ")"
    nsdf['ate_error'] = nsdf['ate'] - nsdf['ate_hat'] 
    return nsdf

def filter_baselines(df, include_baselines):
    include_baselines = ['Target Oracle', 'RW-SL', 'COM', 'OBS Oracle', 'OBS']
    df = df[df['baseline'].isin(include_baselines)]
    return df.reset_index()


################ Main plot functions ################

def plot_syn_convergence(oracle_run, learned_run, path):

    palette = sns.color_palette(['#0072BB', '#FF6123', '#B37538', '#4E148C', '#847F7F', '#000000','#FF6123',  '#0072BB' ])

    oracle_df, _ = get_erm_convergence_result_df(oracle_run)
    learn_df, _  = get_erm_convergence_result_df(learned_run)

    learn_df.reset_index(inplace=True)
    learned_baselines = ['RW-SL', 'COM-SL']
    learn_df = learn_df[learn_df['baseline'].isin(learned_baselines)]
    learn_df['baseline'].replace('RW-SL', 'RW-SL (learned)', inplace=True)
    learn_df['baseline'].replace('COM-SL', 'COM-SL (learned)', inplace=True)

    oracle_baselines = ['Target Oracle', 'RW-SL', 'COM-SL', 'COM', 'OBS Oracle', 'OBS']
    oracle_df = oracle_df[oracle_df['baseline'].isin(oracle_baselines)]
    baselines_df = pd.concat([learn_df, oracle_df])
    baselines_df.reset_index(inplace=True)

    baselines_df = baselines_df[baselines_df['alpha_0'] != 0.2]
    n_baselines = len(baselines_df['baseline'].unique())
    baselines_df['ACC'] = baselines_df['ACC'] * 100

    fig = sns.lineplot(
        palette=palette[:n_baselines],
        data=baselines_df, x="NS", y="ACC", hue='baseline', ci=68, marker='o')
    fig.set_xlabel('N')
    fig.set_ylabel('Accuracy (%)')

    # Set line styles as dashed for specific lines
    line_styles = ['--', '--', '-', '-', '-', '-', '-', '-']
    for line, style in zip(fig.lines, line_styles):
        line.set_linestyle(style)
        
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc=2, borderaxespad=0., fontsize='large', labelspacing=1.25)

    disp_names = {
        'Target Oracle': 'TPO (Oracle)',
        'RW-SL': r'RW-SL (dashed=\hat{\alpha}, \hat{\beta})',
        'COM': 'CP',
        'COM-SL': r'SL',
        'RW': 'RW',
        'Proxy Oracle': r'$f_{Y_t}$',
        'OBS Oracle': 'UT',
        'OBS': 'UP',
        'RW-SL (learned)' : r'RW-SL',
        'COM-SL (learned)' : r'SL'
    }

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [5,1,0,4,3,2]
    plt.legend([handles[idx] for idx in order],[disp_names[labels[idx]] for idx in order],
            bbox_to_anchor=(1.0, 1.0), loc=2, borderaxespad=0., fontsize='large', labelspacing=1.45)


    fig.set_xticks([5000, 10000, 20000, 30000, 40000, 50000, 60000], labels=['5k', '10k', '20k', '30k', '40k', '50k', '60k'])
    plt.savefig(path, dpi=500, bbox_inches = "tight")

    return plt

def get_tabular_syn_results(oracle_run, learn_run, ns):

    oracle_df, _ = get_erm_convergence_result_df(oracle_run)
    learn_df, _  = get_erm_convergence_result_df(learn_run)

    learn_df.reset_index(inplace=True)
    learned_baselines = ['RW-SL', 'COM-SL']
    learn_df = learn_df[learn_df['baseline'].isin(learned_baselines)]
    learn_df['baseline'].unique()
    learn_df['baseline'].replace('RW-SL', 'RW-SL (learned)', inplace=True)
    learn_df['baseline'].replace('COM-SL', 'COM-SL (learned)', inplace=True)

    oracle_baselines = ['Target Oracle', 'RW-SL', 'COM-SL', 'COM', 'OBS Oracle', 'OBS']
    oracle_df = oracle_df[oracle_df['baseline'].isin(oracle_baselines)]
    baselines_df = pd.concat([learn_df, oracle_df])
    baselines_df.reset_index(inplace=True)

    baselines_df['param'] = "(" + baselines_df['alpha_0'].astype(str)+"," +baselines_df['beta_0'].astype(str) + ")"

    # Mean dataframe
    table_df = baselines_df[baselines_df['NS'] == ns][['baseline', 'param', 'ACC']].groupby(['baseline', 'param']).mean().reset_index()
    mean_df = pd.pivot_table(table_df, values='ACC', index=['baseline'], columns=['param'])
    lst = ["OBS", "OBS Oracle", "COM", "COM-SL (learned)", "COM-SL",  "RW-SL (learned)","RW-SL", "Target Oracle"]
    mean_df = mean_df.loc[lst]
    mean_df = 100*mean_df

    # # Se dataframe
    table_df =  baselines_df[baselines_df['NS'] == ns][['baseline', 'param', 'ACC']].groupby(['baseline', 'param']).sem().reset_index()
    sem_df = pd.pivot_table(table_df, values='ACC', index=['baseline'], columns=['param'])
    lst = ["OBS", "OBS Oracle", "COM", "COM-SL (learned)", "COM-SL",  "RW-SL (learned)","RW-SL", "Target Oracle"]
    sem_df = sem_df.loc[lst]
    sem_df = 100*sem_df

    pd.options.display.float_format = "{:,.2f}".format

    for col in mean_df:
        mean_df[col] = mean_df[col].apply(lambda x: '{0:.2f}'.format(x))
        sem_df[col] = sem_df[col].apply(lambda x: '{0:.2f}'.format(x))
        mean_df[col] = mean_df[col].astype(str).map(str) + " (" + sem_df[col].astype(str) + ")"

    print(mean_df.to_latex())

def plot_tao_bias(ohie_oracle_run, jobs_oracle_run, ohie_learned_run, jobs_learned_run, path):

  odf_oracle = get_ate_result_df(ohie_oracle_run).reset_index()
  jdf_oracle = get_ate_result_df(jobs_oracle_run).reset_index()
  odf_learned = get_ate_result_df(ohie_learned_run).reset_index()
  jdf_learned = get_ate_result_df(jobs_learned_run).reset_index()

  # Learned baselines
  plot_assumptions = ['weak_seperability', 'baserate_min_anchor', 'baserate_max_anchor']
  odf_learned = odf_learned[odf_learned['assumption'].isin(plot_assumptions) & (odf_learned['baseline'] == 'RW-SL')]
  odf_learned = odf_learned.sort_values(by=['assumption', 'alpha_0'], ascending=True)
  jdf_learned = jdf_learned[jdf_learned['assumption'].isin(plot_assumptions) & (jdf_learned['baseline'] == 'RW-SL')]
  jdf_learned = jdf_learned.sort_values(by=['assumption', 'alpha_0'], ascending=True)


  # Oracle baselines
  oracle_baselines = [ 'COM', 'OBS Oracle', 'OBS', 'COM-SL', 'RW-SL', 'Target Oracle']
  odf_oracle = odf_oracle[odf_oracle['baseline'].isin(oracle_baselines)]
  jdf_oracle = jdf_oracle[jdf_oracle['baseline'].isin(oracle_baselines)]
  odf_oracle = odf_oracle.sort_values(by=['baseline', 'alpha_0'], ascending=True)
  jdf_oracle = jdf_oracle.sort_values(by=['baseline', 'alpha_0'], ascending=True)

  baseline_names = {
      'Target Oracle': 'CT',
      'RW-SL': r'RW-SL',
      'COM': 'CP',
      'COM-SL': r'SL',
      'OBS Oracle': 'UT',
      'OBS': 'UP',
      'RW-SL (learned)' : r'RW-SL ($\hat{\alpha}$, $\hat{\beta}$)',
      'COM-SL (learned)' : r'SL ($\hat{\alpha}$, $\hat{\beta}$)',
  }

  assumption_names = {
      'weak_seperability': r'Min/Max',
      'baserate_max_anchor': 'Br/Max',
      'baserate_min_anchor': r'Br/Min',
  }

  fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex='col', gridspec_kw={'width_ratios':[2,1]})
  palette = sns.color_palette(['#9467bd', '#17becf',  '#556b2f', '#ff7f0e','#f03b20'])

  # OHIE benchmark with oracle parameters
  g = sns.barplot(data=odf_oracle, x='baseline', y='ate_error', hue='param', palette=palette[:len(oracle_baselines)],
                  order=oracle_baselines, ci=68, errwidth=2, ax=axs[0,0])
  axs[0,0].set_ylabel(r'ATE Error ($\tau - \hat{\tau}$)', fontsize='large')
  axs[0,0].set_title('OHIE', fontsize='large')
  g.legend().set_visible(False)
  axs[0,0].set_xlabel("")


  # JOBS benchmark with oracle parameters
  g = sns.barplot(data=jdf_oracle, x='baseline',y='ate_error',hue='param', palette=palette[:len(oracle_baselines)],
                  ci=68, errwidth=2, ax=axs[1,0], order=oracle_baselines)
  g.legend().set_visible(False)
  axs[1,0].set_ylabel(r'ATE Error ($\tau - \hat{\tau}$)', fontsize='large')
  axs[1,0].set_title('JOBS', fontsize='large')
  axs[1,0].set_xlabel("")

  labels = [baseline_names[item.get_text()] for item in axs[1,0].get_xticklabels()]
  axs[1,0].set_xticklabels(labels, fontsize='large')


  # OHIE benchmark with learned parameters
  g = sns.barplot(data=odf_learned, x='assumption',y='ate_error',hue='param', palette=palette[:len(oracle_baselines)],
                  ci=68, errwidth=2, ax=axs[0,1], order=plot_assumptions)

  axs[0,1].set_ylabel(r'ATE Error ($\tau - \hat{\tau}$)')
  axs[0,1].set_title(r'OHIE: RW-SL ($\hat{\alpha}_t, \hat{\beta}_t$)', fontsize='large')
  axs[0,1].legend( title=r"         ($\alpha_0, \beta_0$)", bbox_to_anchor=(1, 0.3), fontsize='large')
  axs[0,1].set_xlabel("")
  axs[0,1].set_ylabel("")


  # JOBS benchmark with learned parameters
  g = sns.barplot(data=jdf_learned, x='assumption',y='ate_error',hue='param', palette=palette[:len(oracle_baselines)],
                  ci=68, errwidth=2, ax=axs[1,1], order=plot_assumptions)
  g.legend().set_visible(False)
  axs[1,1].set_title(r'JOBS: RW-SL ($\hat{\alpha}_t, \hat{\beta}_t$)', fontsize='large')
  axs[1,1].set_xlabel("")
  axs[1,1].set_ylabel("")

  labels = [assumption_names[item.get_text()] for item in axs[1,1].get_xticklabels()]
  axs[1,1].set_xticklabels(labels, fontsize='large')

  plt.savefig(path, dpi=500, bbox_inches = "tight")

  return plt
#####################################################
