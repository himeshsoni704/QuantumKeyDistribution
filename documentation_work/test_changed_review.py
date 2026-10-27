"""Focused regression checks; extract definitions so the many-hour experiments do not run."""
import sys, os, ast, types, pathlib, json, hashlib, copy, zlib
from math import factorial
root=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'documentation_work/review_test_dependencies'))
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import brentq
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, average_precision_score, accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.base import clone, BaseEstimator, ClassifierMixin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from joblib import Parallel, delayed
torch.set_num_threads(2)
source=(root/'changed.py').read_text(encoding='utf-8'); tree=ast.parse(source)
compile(source,str(root/'changed.py'),'exec')
testdir=root/'documentation_work/review_test_output';testdir.mkdir(exist_ok=True)
os.chdir(testdir)
pathlib.Path('data').mkdir(exist_ok=True);pathlib.Path('plots').mkdir(exist_ok=True)
env=dict(globals(),__name__='review_test_module',_pathlib=pathlib,_json=json,_hashlib=hashlib,_copy=copy,
         CODE_VERSION=hashlib.sha256(source.encode()).hexdigest(),MASTER_SEED=20260913,RUN_PROFILE='quick',
         _FALLBACK_COUNTER=[0],HAS_XGB=False,K_MAIN_DATASET=2000,N_REPEATS=2,
         DL_N_CLEAN=12,DL_N_ATT=12,DL_EPOCHS=1,DL_DEVICE=torch.device('cpu'),
         SWEEP_K=(2000,),SWEEP_STRENGTHS=(0,.1),N_HONEST=10,N_ATT=10,MODELS={},
         FINAL_REGENERATE=True,GEN_JOBS=1,N_JOBS=1,_GD_FP='test_classical',_E91_FP='test_e91',_DL_FP='test_dl')
sys.modules['review_test_module']=types.ModuleType('review_test_module')
const=set('SC _P K_MARGIN COMMISSION_K BKM07_CTRL_FORWARD_NOISE BB84_LOSS_CAP_ETA_BOB PNS_MATCH_SIGNAL_GAIN_ONLY HONEST_DRIFT_FRAC HONEST_DRIFT_MAX GYS MU_SIGNAL MU_DECOY MU_VACUUM DECOY_PROBS CHANNEL_DISTANCE_RANGE_KM STATE_0 STATE_1 STATE_PLUS STATE_MINUS ALICE_ANGLES BOB_ANGLES KEY_PAIRS CHSH_PAIRS CHSH_SIGNS TSIRELSON_BOUND I2 SX SZ PSI_MINUS RHO_SINGLET EVE_BASES BB84_BASE_FEATURES BB84_FEATURE_NAMES BKM_FEATURE_NAMES E91_FEATURE_NAMES E91_CHSH_ONLY_NAMES DISTANCE_RANGE_BKM DL_PROTOCOLS DL_ATTACKS DL_ATTACK_COUNTS N_FEATURES N_CLASSICAL_FEATURES _PROTO_TO_IDX D_CLS H_CLS ARCH SPLIT_AUDIT THRESHOLD_AUDIT RESULT_ARTIFACTS P_LEDGER FEATURE_GROUPS_BB84'.split())
pending=[]
for node in tree.body:
    if isinstance(node,ast.Assign):
        targets={n.id for t in node.targets for n in ast.walk(t) if isinstance(n,ast.Name)}
        if targets and targets<=const:
            try:exec(compile(ast.Module(body=[node],type_ignores=[]),'<definitions>','exec'),env)
            except NameError: pending.append(node)
    elif isinstance(node,(ast.FunctionDef,ast.ClassDef)):
        try:exec(compile(ast.Module(body=[node],type_ignores=[]),'<definitions>','exec'),env)
        except NameError:pending.append(node)
for node in pending:
    try:exec(compile(ast.Module(body=[node],type_ignores=[]),'<definitions>','exec'),env)
    except NameError as e: print('Unused definition not loaded:',getattr(node,'name','assignment'),e)
env['SEEDS']=env['SeedBook']()
env['_DL_SESSION_BUILDERS']={p:env['_'+p+'_dl_session'] for p in ('bb84','bkm07','e91')}
env['_CUSUM_STREAM_BUILDERS']={p:env['_'+p+'_cusum_stream'] for p in ('bb84','bkm07','e91')}
env['MODELS']={a+( '+feat' if use else ''):(a,use) for a in env['ARCH'] for use in (False,True)}
passed=[]
def check(name,condition):
    assert condition,name
    passed.append(name);print('PASS:',name,flush=True)

for mode in ('none','intercept_resend','pns','mixed_pns_ir'):
    f=env['collect_bb84_features'](distance_km=0,eve_mode=mode,eve_intensity=.2,target_k_signal_bits=80,max_pulses=500,
        rng=np.random.default_rng(11),n_windows=8,return_record=True)
    r=f['_record']; check('BB84 exact K continuation '+mode,f['_k_achieved']==80 and f['_N_generated']>=f['_N_used'] and np.isfinite([f[k] for k in env['BB84_FEATURE_NAMES']]).all())
    check('BB84 stopped at K-th signal key '+mode,bool((r['sift'] & (r['k']==0))[-1]) and len(r['click'])==f['_N_used'])
f=env['collect_bkm07_features'](distance_km=0,eve_mode='symmetric',eve_fwd=.1,eve_ret=.1,target_k_key_rounds=50,
    max_pulses=5000,n_windows=8,rng=np.random.default_rng(12),return_record=True)
check('BKM07 exact K continuation',f['_k_achieved']==50 and f['_N_generated']>5000 and f['_N_used']<=f['_N_generated'])
f=env['extract_e91_features'](V=.95,target_k_key_pairs=200,k_margin=.01,n_windows=8,rng=np.random.default_rng(13),return_record=True)
check('E91 exact K continuation',f['_k_achieved']==200 and f['_N_generated']>500 and len(f['_record'][0])==f['_N_used'])
mask=env['_classical_vector']({'a':0,'b':np.nan},['a','b'])
check('Padding distinguished from genuine zero',len(mask)==32 and mask[16]==1 and mask[17]==0 and np.all(mask[18:]==0))
th=env['fixed_fpr_threshold'](np.ones(100),.01)
check('Tied-score FPR target',np.mean(np.ones(100)>th)<=.01)
try:env['fixed_fpr_threshold']([.1,.2],source='test')
except ValueError:passed.append('Test-set thresholds rejected')
else:raise AssertionError('Accepted test-set threshold')
rng=np.random.default_rng(22); groups=np.repeat(np.arange(60),3); y=np.tile([0,1,1],60)
X=rng.normal(size=(len(y),4));X[:,0]+=y*.5
tr,va,te=env['grouped_three_way'](X,y,groups,name='regression')
check('Channel groups disjoint',not(set(groups[tr])&set(groups[te])) and not(set(groups[va])&set(groups[te])))
try:env['audit_split'](groups,[0],[1],[2],'intentional overlap')
except AssertionError:passed.append('Leakage audit rejects twin overlap')
else:raise AssertionError('Leakage audit accepted overlap')
# Use the real sklearn fallback, smaller iteration budget for regression testing only.
env['make_boosted']=lambda seed=0,**kw:HistGradientBoostingClassifier(max_iter=8,max_depth=2,random_state=seed)
cal=env['calibration_analysis'](X,y,'test',groups=groups)
check('Validation calibration metrics',set(('FPR_lo','FPR_hi','pr_auc','brier','ece')).issubset(cal['fpr_table']) and cal['fpr_table'].threshold_source.eq('validation').all())
check('Prevalence precision grid',set(cal['precision_table'].pi_deploy)=={.001,.01,.05,.1})
ci=env['binomial_ci'](0,100);check('Zero false alarms retain uncertainty',ci[0]==0 and ci[1]>0)
paired=env['paired_auc_comparison'](y,X[:,0],X[:,0],groups,B=20)
check('Identical scores paired comparison',paired['delta_auc']==0 and paired['paired_bootstrap_p']==1)
check('Holm retains undefined tests',np.isnan(env['holm']([.01,np.nan,.03])[1]))
svc=env['GroupedSigmoidSVC']().fit(X,y,groups=groups)
check('Grouped SVM probability calibration',np.isfinite(svc.predict_proba(X[te])).all())
check('Grouped nested SVM OOF',np.isfinite(env['oof_scores'](svc,X,y,groups)).all())

# Physics-derived DL sessions at a small K test encoding and same-record fusion.
env['CHANNEL_DISTANCE_RANGE_KM']=(0,0);env['DISTANCE_RANGE_BKM']=(0,0)
for proto in ('bb84','bkm07','e91'):
    sess=env['simulate_session'](proto,'clean',0,np.random.default_rng(3),target_events=128)
    check('DL key budget '+proto,sess.metadata['_k_achieved']==128 and int(sess.X[:,0].sum())==128 and len(sess.x_classical)==32)
    check('DL feature/record consistency '+proto,np.isfinite(sess.X).all() and len(sess.X)>=128)

# Artificial independent sessions exercise every network variant and actual backpropagation.
g=np.repeat(np.arange(36),2); proto=np.repeat(np.repeat(['bb84','bkm07','e91'],12),2)
y=np.repeat(np.tile([0,1],18),2).astype(np.float32)
X=rng.normal(size=(72,16,8)).astype(np.float32); C=np.ones((72,32),np.float32);C[:,:16]=rng.normal(size=(72,16))
data=dict(X=X,protocol=proto,is_attacked=y,group=g,X_classical=C,attack_fine=y.astype(int))
tr,va,te=env['session_split'](g,labels=y,seed=1,test_size=.25,val_size=.25)
loader=lambda ix,shuffle=False:env['make_dl_loader'](X,proto,y,ix,X_classical=C,groups=g,shuffle=shuffle,batch_size=12)
for opts in ({},{'use_classical':False},{'shared_adapter':True},{'use_attention':False}):
    model=env['CrossProtocolDetector'](hidden=8,latent=8,**opts)
    env['train_binary'](model,loader(tr,True),loader(va),'cpu',epochs=1,verbose=False,use_adversarial=True,use_contrastive=True)
    result=env['evaluate_binary'](model,loader(te),'cpu',loader(va),with_ci=True)
    check('DL forward/train/evaluate '+str(opts),result['unit']=='session' and len(result['y'])==len(np.unique(g[te])) and np.isfinite(result['auc']))
probe=env['protocol_probe'](model,data,'cpu',indices=te)
check('Session probe outputs',set(('protocol_probe_auc','protocol_probe_accuracy','chance_auc')).issubset(probe))
# Exercise all architecture constructors; Full passes explicit masks unchanged.
for name in env['ARCH']:
    model=env['ARCH'][name](d_cls=32)
    inp=torch.from_numpy(env['win_stats'](X[:4])) if name=='MLP' else torch.from_numpy(X[:4])
    out=model(inp,torch.from_numpy(C[:4]));check('Architecture '+name,out.shape==(4,) and torch.isfinite(out).all().item())
env['_save_csv'](pd.DataFrame([{'auc':.7,'protocol':'BB84'}]),'data/provenance_check.csv',index=False)
saved=pd.read_csv('data/provenance_check.csv')
check('CSV provenance',saved.code_version.iloc[0]==env['CODE_VERSION'] and pathlib.Path('data/provenance_check.csv.provenance.json').exists())
suite=env['run_review_dl_experiments'](data,'cpu',seeds=(0,))
check('Matched DL experiment suite',set(('scratch','transfer_full','source_only_shared','deep_SVDD','one_class_SVM','isolation_forest')).issubset(suite.model))
check('Matched DL reports all protocols',set(suite.protocol)=={'bb84','bkm07','e91'})
pairs=pd.read_csv('data/review_dl_paired_tests.csv')
for row in pairs.itertuples():
    means=suite[suite.protocol==row.protocol].set_index('model').auc
    assert np.isclose(row.delta_auc,means[row.model_a]-means[row.model_b])
check('Paired comparisons match reported calibrated AUCs',True)
result=dict(passed=passed,count=len(passed),code_sha256=env['CODE_VERSION'],numpy=np.__version__,sklearn=sklearn.__version__,torch=torch.__version__)
(root/'documentation_work/review_test_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
