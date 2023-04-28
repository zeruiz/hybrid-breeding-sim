from RecurrentSelectionTidy import rrs
from RecurrentSelectionTidy import rrs_adaptive
import RecurrentSelection
import numpy as np
import pandas as pd
import scipy

N = 100
B = 40
T = 7

mat = scipy.io.loadmat('/Users/zeeray/Documents/Hybrid Breeding Simulation Platform/GD369_transparent5.mat')
ginfo = {
    'GA': np.swapaxes(mat['GFs'], 0, 2),
    'GB': np.swapaxes(mat['GMs'], 0, 2),
    'RF': mat['RFLs'].squeeze(),
    'a': mat['bs'].squeeze(),
    'dp': mat['dps'].squeeze(),
    'pidx': mat['Pidxs'].squeeze()-1,
    'dn': mat['dns'].squeeze(),
    'nidx': mat['Nidxs'].squeeze()-1
}
params = {
    'T': T,
    'H2': 0.5,
    'M': 40,
    'K': 20,
    'S': 20,
}

mat2 = scipy.io.loadmat('/Users/zeeray/Documents/Hybrid Breeding Simulation Platform/GD369_opaque5.mat')
ginfo2 = {
    'GA': np.swapaxes(mat2['GFl'], 0, 2),
    'GB': np.swapaxes(mat2['GMl'], 0, 2),
    'RF': mat2['RFLl'].squeeze(),
    'a': mat2['bl'].squeeze(),
    'dp': mat2['dpl'].squeeze(),
    'pidx': mat2['Pidxl'].squeeze()-1,
    'dn': mat2['dnl'].squeeze(),
    'nidx': mat2['Nidxl'].squeeze()-1,
    'marker': mat2['marker'].squeeze()-1
}
params2 = {
    'T': T,
    'H2': 0.5,
    'M': 40,
    'K': 20,
    'S': 20,
}

# df_Bay_trans_comp, _, genoA121, genoB121, genoC121, phenoA121, phenoB121, phenoC121, H2_trans_comp, rmse1_trans_comp, rmse2_trans_comp = rrs(ginfo, params, "transparent", "Bayesian", True, B)
df_perf_trans_adj, _, genoA112, genoB112, genoC112, phenoA112, phenoB112, phenoC112, _, _, _ = rrs(ginfo, params, "transparent", "perfect", False, B)
df_perf_trans_comp, _, genoA111, genoB111, genoC111, phenoA111, phenoB111, phenoC111, _, _, _ = rrs(ginfo, params, "transparent", "perfect", True, B)
# df_Bay_trans_adj, _, genoA122, genoB122, genoC122, phenoA122, phenoB122, phenoC122, H2_trans_adj, rmse1_trans_adj, rmse2_trans_adj = rrs(ginfo, params, "transparent", "Bayesian", False, B)
df_perf_opa_comp, _, genoA211, genoB211, genoC211, phenoA211, phenoB211, phenoC211, _, _, _ = rrs(ginfo2, params2, "opaque", "perfect", True, B)
df_perf_opa_adj, _, genoA212, genoB212, genoC212, phenoA212, phenoB212, phenoC212, _, _, _ = rrs(ginfo2, params2, "opaque", "perfect", False, B)
# df_Bay_opa_comp, _, genoA221, genoB221, genoC221, phenoA221, phenoB221, phenoC221, H2_opa_comp, rmse1_opa_comp, rmse2_opa_comp = rrs(ginfo2, params2, "opaque", "Bayesian", True, B)
# df_Bay_opa_adj, _, genoA222, genoB222, genoC222, phenoA222, phenoB222, phenoC222, H2_opa_adj, rmse1_opa_adj, rmse2_opa_adj = rrs(ginfo2, params2, "opaque", "Bayesian", False, B)

# params['H2'] = 0.1
# params2['H2'] = 0.1
# df_pheno_opa_comp, _, genoA231, genoB231, genoC231, phenoA231, phenoB231, phenoC231, _, _, _ = rrs(ginfo2, params2, "opaque", "phenotypic", True, B)
# df_pheno_trans_adj, _, genoA132, genoB132, genoC132, phenoA132, phenoB132, phenoC132, _, _, _ = rrs(ginfo, params, "transparent", "phenotypic", False, B)
# df_pheno_trans_comp, _, genoA131, genoB131, genoC131, phenoA131, phenoB131, phenoC131, _, _, _ = rrs(ginfo, params, "transparent", "phenotypic", True, B)
# df_pheno_opa_adj, _, genoA232, genoB232, genoC232, phenoA232, phenoB232, phenoC232, _, _, _ = rrs(ginfo2, params2, "opaque", "phenotypic", False, B)

# df_BayAda_trans_comp, _, genoA121, genoB121, genoC121, phenoA121, phenoB121, phenoC121 = rrs_adaptive(ginfo, params, "transparent", "Bayesian", True, B)
# df_BayAda_trans_adj, _, genoA122, genoB122, genoC122, phenoA122, phenoB122, phenoC122 = rrs_adaptive(ginfo, params, "transparent", "Bayesian", False, B)
# df_BayAda_opa_comp, _, genoA221, genoB221, genoC221, phenoA221, phenoB221, phenoC221 = rrs_adaptive(ginfo2, params2, "opaque", "Bayesian", True, B)
# df_BayAda_opa_adj, _, genoA222, genoB222, genoC222, phenoA222, phenoB222, phenoC222 = rrs_adaptive(ginfo2, params2, "opaque", "Bayesian", False, B)

# df_Bay_trans_comp.to_csv('SCA_test2copy/bayesian_transparent_comp.csv')
# df_pheno_trans_comp.to_csv('SCA_test2copy/phenotypic_transparent_comp.csv')
df_perf_trans_adj.to_csv('GCA_test2copy/perfect_transparent_adj.csv')
df_perf_trans_comp.to_csv('GCA_test2copy/perfect_transparent_comp.csv')
# df_Bay_trans_adj.to_csv('SCA_test2copy/bayesian_transparent_adj.csv')
# df_pheno_trans_adj.to_csv('SCA_test2copy/phenotypic_transparent_adj.csv')
df_perf_opa_comp.to_csv('GCA_test2copy/perfect_opaque_comp.csv')
df_perf_opa_adj.to_csv('GCA_test2copy/perfect_opaque_adj.csv')

# df_Bay_opa_comp.to_csv('SCA_test2copy/bayesian_opaque_comp.csv')
# df_pheno_opa_comp.to_csv('SCA_test2copy/phenotypic_opaque_comp.csv')
# df_Bay_opa_adj.to_csv('SCA_test2copy/bayesian_opaque_adj.csv')
# df_pheno_opa_adj.to_csv('SCA_test2copy/phenotypic_opaque_adj.csv')

# pd.DataFrame(H2_trans_comp).to_csv('H2HatAB_trans_comp1.csv')
# pd.DataFrame(rmse1_trans_comp).to_csv('rmse1_transparent_comp2.csv')
# pd.DataFrame(rmse2_trans_comp).to_csv('rmse2_transparent_comp2.csv')
# pd.DataFrame(H2_trans_adj).to_csv('H2HatAB_trans_trunc1.csv')
# pd.DataFrame(rmse1_trans_adj).to_csv('rmse1_transparent_adj2.csv')
# pd.DataFrame(rmse2_trans_adj).to_csv('rmse2_transparent_adj2.csv')
# pd.DataFrame(H2_opa_comp).to_csv('H2HatAB_opaq_comp1.csv')
# pd.DataFrame(rmse1_opa_comp).to_csv('rmse1_opaque_comp2.csv')
# pd.DataFrame(rmse2_opa_comp).to_csv('rmse2_opaque_comp2.csv')
# pd.DataFrame(H2_opa_adj).to_csv('H2HatAB_opaq_trunc1.csv')
# pd.DataFrame(rmse1_opa_adj).to_csv('rmse1_opaque_adj2.csv')
# pd.DataFrame(rmse2_opa_adj).to_csv('rmse2_opaque_adj2.csv')