"""Focused DL tests and a small, reproducible physics-derived comparison.

Run with --benchmark for the paired baseline/updated experiment. Definitions
are extracted from the study so its unrelated long-running experiments do not run.
"""
from pathlib import Path
import os, json, sys, ast, argparse, time, hashlib

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'documentation_work/dl_regularization'
os.environ['QKD_DL_CONFIG'] = json.dumps(dict(log_history=False))

def load_env(path):
    setup = (ROOT/'documentation_work/test_changed_review.py').read_text(encoding='utf-8').split('passed=[]')[0]
    setup = setup.replace("source=(root/'changed.py').read_text(encoding='utf-8')", f"source=pathlib.Path({str(path)!r}).read_text(encoding='utf-8')")
    setup = setup.replace("testdir=root/'documentation_work/review_test_output'", "testdir=root/'documentation_work/dl_regularization/test_output'")
    ns = {'__file__':str(ROOT/'documentation_work/test_changed_review.py')}
    exec(compile(setup,str(path),'exec'),ns)
    return ns['env']

def tests(e):
    np, torch = e['np'], e['torch']
    from sklearn.metrics import confusion_matrix
    e['confusion_matrix'] = confusion_matrix
    passed = []
    def check(name, condition):
        assert condition, name
        passed.append(name); print('PASS:',name,flush=True)
    cfg = e['dl_training_config']()
    g = np.repeat(np.arange(40),3)
    y = np.repeat(np.tile([0,1],20),3).astype(np.float32)
    rng = np.random.default_rng(123)
    X = rng.normal(0,.15,(120,16,8)).astype(np.float32)
    X[:,:,1] += y[:,None]
    C = np.ones((120,32),np.float32); C[:,:16]=rng.normal(0,.1,(120,16)); C[:,0]=y
    p = np.array(['bb84']*len(y))
    tr, va, te = e['session_split'](g,labels=y,test_size=.25,val_size=.25,seed=4)
    loader = lambda ix,sh=False:e['make_dl_loader'](X,p,y,ix,X_classical=C,groups=g,shuffle=sh,batch_size=16)
    w = e['_dl_session_weights'](np.array([0,1,1,1]),4)
    check('Each session has equal total sampling weight',np.isclose(w[0],w[1:].sum()))
    model = e['CrossProtocolDetector'](hidden=8,latent=8)
    optimizer = e['_dl_optimizer'](model,.001)
    decay_ids = {id(v) for group in optimizer.param_groups if group['weight_decay'] for v in group['params']}
    check('AdamW excludes biases and normalization vectors',all(id(v) not in decay_ids for name,v in model.named_parameters() if v.ndim==1))
    penalty=e['_dl_l1'](model);penalty.backward()
    check('L1 reaches linear weights but excludes LSTM and norm parameters',model.adapters['bb84'].weight.grad is not None and model.lstm.weight_ih_l0.grad is None and model.embedding_norm.weight.grad is None)
    # Exercise both definitions used by the exported script, including its later compatibility patch.
    model.freeze_trunk = e['_freeze_trunk'].__get__(model)
    model.train = e['_train'].__get__(model)
    model.freeze_trunk(); model.train()
    frozen={n:v.detach().clone() for n,v in model.state_dict().items() if n.startswith(('conv.','lstm.','pool.','embedding_norm.'))}
    e['train_binary'](model,loader(tr,True),loader(va),'cpu',epochs=2,verbose=False,select_best=False)
    check('Frozen trunk weights and BatchNorm buffers stay unchanged',all(torch.equal(v,model.state_dict()[n]) for n,v in frozen.items()))
    check('Fixed-budget transfer does not record validation metrics',len(model.training_history_)==2 and all('val_score' not in r for r in model.training_history_))
    check('Frozen embedding dropout is disabled',not model.embedding_dropout.training)
    os.environ['QKD_DL_CONFIG']=json.dumps(dict(log_history=False,patience=2,min_epochs=3))
    mon=e['DLTrainingMonitor'](); mini=torch.nn.Linear(1,1)
    original=mini.weight.detach().clone()
    for epoch in range(1,10):
        if epoch>1:
            with torch.no_grad():mini.weight.add_(1)
        if mon.update(mini,epoch,1.,.6,1.):break
    mon.finish(mini,'test')
    check('Early stopping stops a plateau and restores the best checkpoint',epoch==3 and torch.equal(mini.weight,original))
    os.environ['QKD_DL_CONFIG']=json.dumps(dict(log_history=False))
    # Every architecture trains, predicts finite values, and supports singleton inference.
    for name in e['MODELS']:
        xx=e['win_stats'](X) if name.startswith('MLP') else X
        pred,best=e['fit_dl'](name,xx[tr],C[tr],y[tr],xx[va],C[va],y[va],epochs=2,bs=13,protocol='bb84',train_groups=g[tr],val_groups=g[va],device='cpu')
        check('Train/predict '+name,np.isfinite(pred(xx[te],C[te])).all() and pred(xx[te[:1]],C[te[:1]]).shape==(1,))
    pred,best=e['fit_dl']('MLP+feat',e['win_stats'](X[tr]),C[tr],y[tr],e['win_stats'](X[va]),C[va],y[va],epochs=60,bs=16,seed=7,protocol='bb84',train_groups=g[tr],val_groups=g[va])
    auc=e['roc_auc_score'](y[te],pred(e['win_stats'](X[te]),C[te]))
    print('Learnability AUC:',auc,'epochs:',pred.training_summary_['epochs_run'],flush=True)
    synthetic_auc=auc
    check('Learnability control exceeds 0.95 AUC on independent synthetic sessions',auc>.95)
    sy=np.array([0,0,1,1,1]); sp=np.array([-2.,2.,1.,1.,1.]); sg=np.array([1,1,2,2,2])
    auc,_=e['_dl_binary_summary'](sy,sp,sg,logits=True)
    check('Checkpoint metric aggregates by session',auc==1.)
    try:e['fit_dl']('MLP',e['win_stats'](X[tr]),C[tr],y[tr],e['win_stats'](X[va]),C[va],y[va],train_groups=g[tr],val_groups=g[tr][:len(va)])
    except ValueError:check('Overlapping train/validation sessions rejected',True)
    else:raise AssertionError('Session overlap accepted')
    model=e['CrossProtocolDetector'](hidden=8,latent=8)
    e['train_fine'](model,'bb84',loader(tr,True),loader(va),'cpu',epochs=2,verbose=False)
    macro,cm,vl=e['evaluate_fine'](model,'bb84',loader(te),'cpu',return_loss=True)
    check('Multiclass training and validation aggregate independent sessions',cm.sum()==len(np.unique(g[te])) and np.isfinite([macro,vl]).all())
    pd=e['pd']
    meta=pd.DataFrame(dict(y=[0,0,1,1,1],attack=['clean','clean','IR','IR','IR'],strength=[0,0,.03,.3,.6]))
    rows=e['dl_attack_diagnostics'](meta,np.arange(5),np.array([.1,.2,.15,.7,.9]),.5,'test','bb84',96,0)
    check('Strength diagnostics use the supplied frozen threshold',len(rows)==4 and next(r for r in rows if r['strength_band']=='weak')['recall']==0 and next(r for r in rows if r['strength_band']=='strong')['recall']==1 and all(r['threshold']==.5 for r in rows))
    # Exercise the production benchmark/report wrapper as well as its individual trainer.
    meta=pd.DataFrame(dict(sid=np.arange(40),y=y[::3].astype(int),attack=np.where(y[::3]>0,'intercept_resend','clean'),strength=y[::3]*.5))
    e['SESS']={'bb84':(meta,list(X[::3]),C[::3])}
    e['make_boosted']=lambda seed=0:e['HistGradientBoostingClassifier'](max_iter=2,random_state=seed)
    e['time']=time
    result,score=e['run_dl']('bb84',16,epochs=1,seed=4,models=('MLP+feat',))
    report=Path('data/dl_attack_diagnostics_bb84_L16_seed4.csv')
    check('End-to-end benchmark exports epoch and attack diagnostics',report.exists() and result[0]['epochs_run']==1 and 'accuracy' in result[0])
    os.environ['QKD_DL_CONFIG']=json.dumps(dict(log_history=True))
    monitor=e['DLTrainingMonitor']();monitor.update(model,1,.5,.6,.7);monitor.finish(model,'history_check')
    logged=json.loads(Path(model.training_summary_['history_file']).read_text())
    check('Training history records configuration and selected checkpoint',logged['selected_epoch']==1 and logged['config']['l1']==1e-4 and len(logged['history'])==1)
    os.environ['QKD_DL_CONFIG']=json.dumps(dict(log_history=False))
    result=dict(checks=passed,count=len(passed),synthetic_auc=synthetic_auc,code_sha256=hashlib.sha256((ROOT/'changed.py').read_bytes()).hexdigest(),torch=torch.__version__,cuda_available=torch.cuda.is_available())
    (WORK/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

def benchmark(updated):
    np=updated['np']; baseline=load_env(WORK/'baseline_changed.py')
    rows=[]
    for protocol in ('bb84','bkm07','e91'):
        # All versions see exactly the same sessions and splits. Local distances
        # keep this diagnostic bounded; this is not the full deployment study.
        updated['CHANNEL_DISTANCE_RANGE_KM']=(0.,10.)
        updated['DISTANCE_RANGE_BKM']=(0.,3.)
        cache=WORK/('diagnostic_'+protocol+'.npz')
        if cache.exists():
            z=np.load(cache);X,C,y,g,regime=[z[k] for k in ('X','C','y','g','regime')]
        else:
            xx=[];cc=[];yy=[];gg=[];rr=[]
            for sid in range(48):
                attack='clean' if sid<24 else updated['DL_ATTACKS'][protocol][1]
                strength=0 if sid<24 else (.6 if sid<36 else .03)
                record=updated['_DL_SESSION_BUILDERS'][protocol](attack,strength,np.random.default_rng(2000+sid),256)
                kept,error,ba,bb,tf,gap,aux,chsh,nb,features,meta=record
                assert meta['_k_achieved']==256
                encoded=updated['encode_session'](kept,error,ba,bb,tf,gap,aux,chsh_running=chsh,n_bases=nb)
                windows=updated['make_windows'](encoded,length=96,stride=48)
                xx.extend(windows);cc.extend([features]*len(windows));yy.extend([int(sid>=24)]*len(windows));gg.extend([sid]*len(windows));rr.extend([0 if sid<24 else (1 if sid<36 else 2)]*len(windows))
            X=np.asarray(xx,np.float32);C=np.asarray(cc,np.float32);y=np.asarray(yy,np.float32);g=np.asarray(gg);regime=np.asarray(rr)
            np.savez_compressed(cache,X=X,C=C,y=y,g=g,regime=regime)
        for seed in (0,1):
            tr,va,te=updated['session_split'](g,labels=regime,test_size=.25,val_size=.25,seed=seed)
            for label,e,epochs in [('baseline',baseline,12),('updated',updated,30)]:
                t=time.monotonic()
                args=dict(epochs=epochs,seed=seed,protocol=protocol)
                if label=='updated':args.update(train_groups=g[tr],val_groups=g[va],device='cpu')
                pred,best=e['fit_dl']('Full+feat',X[tr],C[tr],y[tr],X[va],C[va],y[va],**args)
                logits=pred(X[te],C[te]);ids,scores=e['session_scores'](logits,g[te])
                _,first=np.unique(g[te],return_index=True);truth=y[te][first];reg=regime[te][first]
                row=dict(protocol=protocol,seed=seed,version=label,model='Full+feat',epochs_budget=epochs,
                    epochs_run=pred.training_summary_['epochs_run'] if label=='updated' else epochs,
                    session_auc=float(e['roc_auc_score'](truth,scores)),test_sessions=len(ids),seconds=round(time.monotonic()-t,2))
                for code,key in [(1,'strong_auc'),(2,'weak_auc')]:
                    mask=(reg==0)|(reg==code);row[key]=float(e['roc_auc_score'](truth[mask],scores[mask]))
                rows.append(row);print(json.dumps(row),flush=True)
                (WORK/'benchmark_results.json').write_text(json.dumps(dict(scope='Diagnostic only: intercept attacks, K=256, 48 sessions/protocol, two seeds, short distances; baseline12 vs updated30 epochs',results=rows),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true');args=parser.parse_args()
    env=load_env(ROOT/'changed.py')
    benchmark(env) if args.benchmark else tests(env)
