from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icmat_foundry.propnet.contracts import SPLIT_TO_CODE
from icmat_foundry.propnet.data import load_source_archive
from icmat_foundry.finals_post50_passive_rnd_20260803.training.common import (
    CONTRACT_ROOT,
    FitResult,
    SEED_BASE,
    ThermalGuard,
    export_static_onnx,
    mae,
    parameter_count,
    r2,
    rmse,
    save_fit,
    set_seed,
    state_to_cpu,
    utc_now,
    write_family_receipt,
    write_json,
)


FAMILY_ID = "RND-MAT-03"
SOURCE = ROOT / "research/data_assets/icmat_foundry/nist_jarvis_dft/raw/jdft_3d-9-24-2025.json.zip"
CACHE = ROOT / "icmat_foundry/finals_50model/data/jarvis_feature_bank_v1.npz"


class IRNet(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(149, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, max(24, hidden // 2)), nn.ReLU(), nn.Linear(max(24, hidden // 2), 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value.reshape(-1, 149))


class IRExport(nn.Module):
    def __init__(self, model: IRNet, target_mean: float, target_std: float) -> None:
        super().__init__(); self.model=model
        self.register_buffer("target_mean",torch.tensor([target_mean],dtype=torch.float32));self.register_buffer("target_std",torch.tensor([target_std],dtype=torch.float32))

    def forward(self,value:torch.Tensor)->torch.Tensor:
        transformed=self.model(value)*self.target_std+self.target_mean
        return torch.exp(transformed)-1.0


CONFIGS=(
    {"id":"v0_compact","hidden":64,"dropout":0.0,"loss":"huber"},
    {"id":"v1_wide","hidden":128,"dropout":0.0,"loss":"huber"},
    {"id":"v2_dropout","hidden":96,"dropout":0.10,"loss":"huber"},
    {"id":"v3_tail_mse","hidden":96,"dropout":0.05,"loss":"mse"},
)


def finite(value:Any)->float|None:
    try:output=float(value)
    except(TypeError,ValueError):return None
    return output if math.isfinite(output) and 0<=output<=10000 else None


def predict(model:IRNet,x:np.ndarray,mean:float,std:float,device:torch.device)->np.ndarray:
    model.eval();outputs=[]
    with torch.inference_mode():
        for start in range(0,len(x),4096):outputs.append(model(torch.from_numpy(x[start:start+4096]).to(device)).cpu().numpy().reshape(-1))
    return np.expm1(np.concatenate(outputs)*std+mean)


def metrics(y:np.ndarray,p:np.ndarray)->dict[str,float]:
    return {"mae":mae(y,p),"rmse":rmse(y,p),"r2":r2(y,p),"log1p_mae":mae(np.log1p(y),np.log1p(np.clip(p,0,None)))}


def train_fit(fit_id:str,config:dict[str,Any],seed:int,x_train:np.ndarray,y_train:np.ndarray,x_val:np.ndarray,y_val:np.ndarray,target_mean:float,target_std:float,device:torch.device,thermal:ThermalGuard)->tuple[IRNet,dict[str,Any]]:
    thermal.check();set_seed(seed);model=IRNet(int(config["hidden"]),float(config["dropout"])).to(device)
    transformed=((np.log1p(y_train)-target_mean)/target_std).astype(np.float32)
    loader=DataLoader(TensorDataset(torch.from_numpy(x_train),torch.from_numpy(transformed[:,None])),batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4);best_state=None;best=float("inf");history=[];stale=0;started=time.perf_counter()
    for epoch in range(60):
        model.train();losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device);optimizer.zero_grad(set_to_none=True);out=model(xb);loss=F.mse_loss(out,yb) if config["loss"]=="mse" else F.smooth_l1_loss(out,yb);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        prediction=predict(model,x_val,target_mean,target_std,device);score=mae(np.log1p(y_val),np.log1p(np.clip(prediction,0,None)));history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"val_log1p_mae":score})
        if score<best-1e-5:best=score;best_state=state_to_cpu(model);stale=0
        else:stale+=1
        if stale>=6 and epoch>=10:break
    if best_state is None:raise RuntimeError(f"no IR checkpoint {fit_id}")
    model.load_state_dict(best_state);model.to(device);selection=metrics(y_val,predict(model,x_val,target_mean,target_std,device))
    save_fit(FitResult(FAMILY_ID,fit_id,config,seed,selection,best_state,time.perf_counter()-started,parameter_count(model),{"history":history,"epochs":len(history)}));return model,selection


def main()->int:
    rows,integrity=load_source_archive(SOURCE)
    with np.load(CACHE,allow_pickle=False) as loaded:bank={name:loaded[name] for name in loaded.files}
    if len(rows)!=len(bank["jids"]) or any(str(row.get("jid",""))!=str(jid) for row,jid in zip(rows,bank["jids"])):
        raise RuntimeError("IR source/cache JID ordering mismatch")
    targets=np.full(len(rows),np.nan,dtype=np.float32)
    for index,row in enumerate(rows):
        value=finite(row.get("max_ir_mode"));targets[index]=np.nan if value is None else value
    valid=np.isfinite(targets);codes=bank["split_codes"]
    masks={name:valid&(codes==SPLIT_TO_CODE[name]) for name in ("train","tune","test")}
    if min(int(mask.sum()) for mask in masks.values())<100:raise RuntimeError("IR split coverage insufficient")
    feature_mean=bank["features"][masks["train"]].mean(0).astype(np.float32);feature_std=bank["features"][masks["train"]].std(0).astype(np.float32);feature_std[feature_std<1e-6]=1
    x=np.clip((bank["features"]-feature_mean)/feature_std,-8,8).astype(np.float32)
    target_mean=float(np.log1p(targets[masks["train"]]).mean());target_std=float(np.log1p(targets[masks["train"]]).std());target_std=max(target_std,1e-6)
    task_contract={"schema":"x5_icmat_foundry.post50_task_contract.v1","family_id":FAMILY_ID,"task":"JARVIS computed maximum IR mode intensity regression","target":"max_ir_mode","transform":"log1p","split":"existing formula/approximate-structure-family disjoint split","source_archive_sha256":integrity["archive_sha256"],"claim_boundary":"version-fixed public JARVIS DFT proxy, not measured IR intensity"}
    write_json(CONTRACT_ROOT/"tasks/RND-MAT-03.json",task_contract,seal=True)
    baselines={}
    for name,model in (("ridge",Ridge(alpha=100.0)),("extra_trees",ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,max_features=0.7,n_jobs=4,random_state=SEED_BASE))):
        started=time.perf_counter();model.fit(x[masks["train"]],np.log1p(targets[masks["train"]]));prediction=np.expm1(model.predict(x[masks["tune"]]));selection=metrics(targets[masks["tune"]],prediction);save_fit(FitResult(FAMILY_ID,f"baseline_{name}",{"kind":name},SEED_BASE,selection,None,time.perf_counter()-started,None,{"baseline":True}),keep_checkpoint=False);baselines[name]=model
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");thermal=ThermalGuard()
    scores={};models={}
    for ci,config in enumerate(CONFIGS):
        model,selection=train_fit(f"grid_{config['id']}",config,SEED_BASE+11000+ci,x[masks["train"]],targets[masks["train"]],x[masks["tune"]],targets[masks["tune"]],target_mean,target_std,device,thermal);scores[config["id"]]=selection["log1p_mae"];models[config["id"]]=model.cpu()
    ranked=sorted(CONFIGS,key=lambda config:scores[config["id"]]);variants=[("quality",copy.deepcopy(ranked[0]),SEED_BASE+11900),("lite",{"id":"v4_lite","hidden":32,"dropout":0.0,"loss":"huber"},SEED_BASE+11901)]
    final_models={}
    for name,config,seed in variants:
        model,_=train_fit(f"final_{name}",config,seed,x[masks["train"]],targets[masks["train"]],x[masks["tune"]],targets[masks["tune"]],target_mean,target_std,device,thermal);final_models[name]=model.cpu().eval()
    lock={"schema":"x5_icmat_foundry.post50_selection_lock.v1","created_at":utc_now(),"family_id":FAMILY_ID,"fit_count_before_test":8,"ranked_configuration_ids":[item["id"] for item in ranked],"test_opened":False,"post_test_retuning_allowed":False};write_json(CONTRACT_ROOT/"selection_locks/RND-MAT-03.json",lock,seal=True)
    median=float(np.median(targets[masks["train"]]));locked={"median":{"mae":mae(targets[masks["test"]],np.full(np.sum(masks["test"]),median))}}
    for name,model in baselines.items():locked[name]=metrics(targets[masks["test"]],np.expm1(model.predict(x[masks["test"]])))
    exports={}
    for name,model in final_models.items():
        prediction=predict(model.to(device),x[masks["test"]],target_mean,target_std,device);locked[name]=metrics(targets[masks["test"]],prediction);wrapper=IRExport(model.cpu(),target_mean,target_std)
        exports[name]=export_static_onnx(FAMILY_ID,name,wrapper,[torch.from_numpy(x[masks["test"]][:1].reshape(1,1,1,149))],["jarvis_features_fp32"],["max_ir_mode"],{"feature_mean":feature_mean,"feature_std":feature_std},{"configuration":next(item[1] for item in variants if item[0]==name),"seed":next(item[2] for item in variants if item[0]==name)})
    quality=locked["quality"];gate={"raw_mae_beats_median":quality["mae"]<locked["median"]["mae"],"log_mae_beats_ridge":quality["log1p_mae"]<locked["ridge"]["log1p_mae"]};gate["pass"]=all(gate.values())
    receipt=write_family_receipt(FAMILY_ID,{"model_name":"IRIntensity-v2","fit_count":8,"fit_count_contract_pass":True,"task_contract":task_contract,"selection_lock":lock,"locked_test":locked,"innovation_gate":gate,"candidate_class":"RND_VALIDATED" if gate["pass"] else "RND_USABLE_EXPERIMENTAL","exports":exports,"thermal_samples":thermal.samples,"claim_boundary":["public JARVIS DFT proxy only","not a graph neural network claim","no audit, routing, or execution authority","PC metrics are not X5 evidence"]})
    print(json.dumps({"family":FAMILY_ID,"fit_count":8,"gate":gate,"receipt":receipt["path"]},ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
