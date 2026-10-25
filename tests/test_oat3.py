
import ast
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pipeline as p
import oat3_completa as o

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def full(tmp_path_factory):
    folder = tmp_path_factory.mktemp('oat3')
    original = o.salvar_resultados
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(o, 'PASTA_RESULTADOS', folder)
        patch.setattr(o, 'salvar_resultados', lambda resultado, figura: original(resultado, figura, folder))
        yield o.main()

@pytest.fixture(scope='module')
def grid_only():
    return p.main()

def test_raw_dataset():
    raw=pd.read_excel(p.ARQUIVO_DADOS)
    assert len(raw)==731
    assert raw.Data.min()==pd.Timestamp('2024-01-01')
    assert raw.Data.max()==pd.Timestamp('2025-12-31')
    assert raw[['Trocas_Oleo','Manutencao_Motor']].isna().sum().tolist()==[6,3]
    assert (raw.Manutencao_Motor<0).sum()==5

def test_target_and_baselines_alignment():
    raw=pd.read_excel(p.ARQUIVO_DADOS).set_index('Data')
    raw[['Trocas_Oleo','Manutencao_Motor']]=raw[['Trocas_Oleo','Manutencao_Motor']].mask(raw[['Trocas_Oleo','Manutencao_Motor']]<0)
    data,cols=p.carregar_dados()
    assert len(data)==724
    assert cols==['Trocas_Oleo','Manutencao_Motor']
    np.testing.assert_allclose(data.Demanda_Amanha,raw.Trocas_Oleo.shift(-1).loc[data.index])
    np.testing.assert_allclose(data.Naive,raw.Trocas_Oleo.ffill().loc[data.index])
    np.testing.assert_allclose(data.Media_Movel_7,raw.Trocas_Oleo.ffill().rolling(7,min_periods=1).mean().loc[data.index])
    assert not data[cols].lt(0).any().any()

@pytest.mark.parametrize('bad',['duplicate','gap','invalid'])
def test_reject_invalid_dates(monkeypatch,bad):
    raw=pd.read_excel(p.ARQUIVO_DADOS)
    if bad=='duplicate': raw.loc[1,'Data']=raw.loc[0,'Data']
    elif bad=='gap': raw=raw.drop(index=20)
    else: raw.loc[1,'Data']=pd.NaT
    monkeypatch.setattr(p.pd,'read_excel',lambda *a,**k: raw.copy())
    with pytest.raises(ValueError): p.carregar_dados()

def test_future_changes_do_not_change_past_features(monkeypatch):
    raw=pd.read_excel(p.ARQUIVO_DADOS)
    before,_=p.carregar_dados()
    cutoff=pd.Timestamp('2025-06-01')
    raw.loc[raw.Data>cutoff,['Trocas_Oleo','Manutencao_Motor']]=9999.0
    monkeypatch.setattr(p.pd,'read_excel',lambda *a,**k:raw.copy())
    after,_=p.carregar_dados()
    cols=['Trocas_Oleo','Manutencao_Motor','Naive','Media_Movel_7']
    idx=before.index[before.index<=cutoff]
    pd.testing.assert_frame_equal(before.loc[idx,cols],after.loc[idx,cols])

def test_splits_respect_time():
    e=o.preparar_experimento()
    assert (len(e['X_train']),len(e['X_test']))==(579,145)
    assert e['X_train'].index.max()<e['X_test'].index.min()
    for train,val in e['cv'].split(e['X_train']):
        assert e['X_train'].index[train[-1]]+pd.Timedelta(days=1)<e['X_train'].index[val[0]]

def test_imputer_and_scaler_keep_training_statistics():
    model=p.criar_pipeline(RandomForestRegressor(n_estimators=2,random_state=42))
    x=pd.DataFrame({'a':[1.,3.,np.nan,5.],'b':[2.,4.,6.,8.]})
    model.fit(x,[1,2,3,4])
    imputer=model.named_steps['imputacao']; scaler=model.named_steps['padronizacao']
    np.testing.assert_allclose(imputer.statistics_,[3,5])
    before=scaler.mean_.copy()
    model.predict(pd.DataFrame({'a':[1e8,np.nan],'b':[1e8,np.nan]}))
    np.testing.assert_array_equal(before,scaler.mean_)
    np.testing.assert_allclose(before,[3,5])

@pytest.mark.parametrize('real,pred,mae,rmse,mape',[
    ([0,10,20],[5,8,22],3,np.sqrt(11),15),
    ([0,0],[1,3],2,np.sqrt(5),np.nan),
    ([0,4],[0,4],0,0,0),
])
def test_metrics_including_zeros(real,pred,mae,rmse,mape):
    r=p.calcular_metricas(real,pred)
    assert r['MAE']==pytest.approx(mae)
    assert r['RMSE']==pytest.approx(rmse)
    if np.isnan(mape): assert np.isnan(r['MAPE (demanda > 0) %'])
    else: assert r['MAPE (demanda > 0) %']==pytest.approx(mape)

@pytest.mark.parametrize('n',[0,-1,10,1.5,None,True])
def test_invalid_random_iterations(n):
    with pytest.raises(ValueError): o.executar_random({},None,o.GRIDS['Random Forest'],n_iter=n)

def test_percentage_reduction():
    assert o.reducao_percentual(10,8)==20
    assert o.reducao_percentual(10,12)==-20
    assert np.isnan(o.reducao_percentual(0,0))

def test_model_and_search_selection(full):
    assert full['modelo_escolhido']=='Random Forest'
    assert full['melhor_metodo']=='Grid Search'
    expected={'modelo__max_depth':3,'modelo__n_estimators':50}
    for method,count in [('grid',9),('random',6)]:
        s=full[method]['busca']
        assert s.best_params_==expected
        assert len(s.cv_results_['params'])==count
        assert s.cv.n_splits==5 and s.cv.gap==1
        assert s.scoring=='neg_root_mean_squared_error'
        assert s.n_jobs==1
        assert s.n_features_in_==2
    assert full['grid']['busca'].best_score_==full['random']['busca'].best_score_

def test_predictions_match_original_saved_delivery(full):
    expected=pd.read_csv(ROOT/'output/oat3/previsoes.csv',index_col=0,parse_dates=True)
    pd.testing.assert_frame_equal(full['previsoes'],expected,check_freq=False,rtol=1e-10,atol=1e-10)
    assert full['previsoes'].shape==(145,8)
    assert np.isfinite(full['previsoes'].to_numpy()).all()
    assert full['previsoes'].index.min()==pd.Timestamp('2025-08-09')
    assert full['previsoes'].index.max()==pd.Timestamp('2025-12-31')

def test_metrics_match_original_saved_delivery(full):
    expected=pd.read_csv(ROOT/'output/oat3/metricas.csv',index_col=0)
    pd.testing.assert_frame_equal(full['metricas'],expected,rtol=1e-10,atol=1e-10)
    real=full['previsoes']['Real'].to_numpy()
    for col in full['previsoes'].columns[1:]:
        pred=full['previsoes'][col].to_numpy()
        row=full['metricas'].loc[col]
        assert row.MAE==pytest.approx(np.abs(real-pred).mean())
        assert row.RMSE==pytest.approx(np.sqrt(((real-pred)**2).mean()))
        assert row['MAPE (demanda > 0) %']==pytest.approx((np.abs(real-pred)/real).mean()*100)

def test_grid_script_matches_complete_experiment(full,grid_only):
    np.testing.assert_allclose(grid_only['previsoes']['Random Forest ajustado'],full['previsoes']['Random Forest - Grid Search'])
    np.testing.assert_allclose(grid_only['previsoes']['Real'],full['previsoes']['Real'])
    assert grid_only['busca'].best_params_==full['grid']['busca'].best_params_
    assert grid_only['reducao_rmse_percentual']==pytest.approx(full['reducao_referencia_percentual'])

def test_artifacts_and_plot(full):
    folder=o.PASTA_RESULTADOS
    assert {p.name for p in folder.iterdir()}=={'metricas.csv','cenarios.csv','comparacao.csv','previsoes.csv','conclusao.md','comparacao_final.png'}
    assert (folder/'comparacao_final.png').stat().st_size>10000
    assert len(full['cenarios'])==3
    assert full['comparacao']['Tempo (s)'].gt(0).all()
    conclusion=(folder/'conclusao.md').read_text(encoding='utf-8')
    assert '[INSERIR' not in conclusion
    assert '145 dias' in conclusion and '10.70%' in conclusion
    fig=o.criar_grafico(full)
    assert len(fig.axes[0].lines)==4
    assert all(len(line.get_ydata())==60 for line in fig.axes[0].lines)
    plt.close(fig)

@pytest.mark.parametrize('filename,module',[('oat3_completa.ipynb',o),('oat3_gridsearch.ipynb',p)])
def test_notebook_functions_equal_scripts(filename,module):
    nb=json.loads((ROOT/filename).read_text(encoding='utf-8'))
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    nbf={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
    sf={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(module.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
    for name in sf.keys()&nbf.keys(): assert sf[name]==nbf[name],name
