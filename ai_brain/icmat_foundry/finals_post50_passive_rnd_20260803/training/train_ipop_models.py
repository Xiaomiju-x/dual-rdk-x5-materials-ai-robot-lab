from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pymatgen.core import Composition, Element
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


SOURCE = ROOT / "research/data_assets/ipop_v3/ipop_normalized_v1.csv"
FEATURE_WIDTH = 119 * 3 + 12
TASKS = (
    {"family_id": "RND-MAT-01", "name": "FluoHost-IQE", "target": "internal_qe_pct", "unit": "%", "claim": "public literature internal quantum-efficiency proxy"},
    {"family_id": "RND-MAT-02", "name": "FluoHost-Lifetime", "target": "log10_decay_ns", "unit": "log10(ns)", "claim": "public literature decay-lifetime proxy"},
)


class RoleAwareNet(nn.Module):
    def __init__(self, y_mean: float, y_std: float, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden"])
        self.role_aware = bool(config["role_aware"])
        self.dropout = nn.Dropout(float(config["dropout"]))
        if self.role_aware:
            self.host = nn.Sequential(nn.Linear(119, hidden), nn.ReLU())
            self.dopant = nn.Sequential(nn.Linear(119, hidden), nn.ReLU())
            self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Sigmoid())
            self.head = nn.Sequential(nn.Linear(hidden * 2 + 12, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.head = nn.Sequential(nn.Linear(FEATURE_WIDTH, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.register_buffer("y_mean", torch.tensor([y_mean], dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor([y_std], dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.reshape(-1, FEATURE_WIDTH)
        if self.role_aware:
            host = self.host(value[:, :119])
            dopant_1 = self.dopant(value[:, 119:238])
            dopant_2 = self.dopant(value[:, 238:357])
            dopant = dopant_1 + dopant_2
            gate = self.gate(torch.cat((host, dopant_1, dopant_2), dim=1))
            interaction = host * dopant * gate
            normalized = self.head(torch.cat((self.dropout(host), self.dropout(interaction), value[:, 357:]), dim=1))
        else:
            normalized = self.head(self.dropout(value))
        return self.y_mean + normalized * self.y_std


CONFIGS = (
    {"id": "v0_plain", "role_aware": False, "hidden": 48, "dropout": 0.0},
    {"id": "v1_role32", "role_aware": True, "hidden": 32, "dropout": 0.0},
    {"id": "v2_role48", "role_aware": True, "hidden": 48, "dropout": 0.0},
    {"id": "v3_role64", "role_aware": True, "hidden": 64, "dropout": 0.0},
    {"id": "v4_role_dropout", "role_aware": True, "hidden": 48, "dropout": 0.10},
    {"id": "v5_lite", "role_aware": True, "hidden": 20, "dropout": 0.0},
)


def component_split(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=np.float64)
    original_indices = np.flatnonzero(np.isfinite(values))
    parent = np.arange(len(original_indices), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for column in ("reference_doi", "host_formula_reduced"):
        groups: dict[str, list[int]] = defaultdict(list)
        for local, original in enumerate(original_indices):
            value = str(frame.iloc[original][column]).strip().lower()
            if value and value != "nan":
                groups[value].append(local)
        for group in groups.values():
            for item in group[1:]:
                union(group[0], item)
    components: dict[int, list[int]] = defaultdict(list)
    for local in range(len(original_indices)):
        components[find(local)].append(local)
    splits = np.empty(len(original_indices), dtype="U5")
    component_ids = np.empty(len(original_indices), dtype="U16")
    for members in components.values():
        identifiers = []
        for local in members:
            row = frame.iloc[original_indices[local]]
            identifiers.extend((str(row["reference_doi"]).lower(), str(row["host_formula_reduced"]).lower()))
        digest = hashlib.sha256("|".join(sorted(set(identifiers))).encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        split = "train" if bucket < 70 else "val" if bucket < 85 else "test"
        for local in members:
            splits[local] = split
            component_ids[local] = digest[:16]
    counts = {name: int(np.sum(splits == name)) for name in ("train", "val", "test")}
    if min(counts.values()) < 50:
        raise RuntimeError(f"IPOP connected-component split too small for {target}: {counts}")
    metadata = {
        "target_rows": len(original_indices),
        "component_count": len(components),
        "max_component_rows": max(len(value) for value in components.values()),
        "split_counts": counts,
        "group_definition": "connected components over reference DOI and reduced host formula",
    }
    return original_indices, splits, {"component_ids": component_ids.tolist(), **metadata}


def atomic_number(value: Any) -> int:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 0
    try:
        return int(Element(text).Z)
    except Exception:
        return 0


def raw_features(frame: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(indices), FEATURE_WIDTH), dtype=np.float32)
    numeric_fields = (
        "dopant_1_valency", "dopant_1_concentration", "dopant_2_valency",
        "dopant_2_concentration", "temperature_k", "excitation_source_nm",
    )
    for output_index, original in enumerate(indices):
        row = frame.iloc[original]
        composition = Composition(str(row["host_formula_raw"])).fractional_composition
        for element, amount in composition.items():
            if 1 <= element.Z <= 118:
                matrix[output_index, element.Z] = float(amount)
        for offset, column in ((119, "dopant_1"), (238, "dopant_2")):
            z = atomic_number(row[column])
            if z:
                matrix[output_index, offset + z] = 1.0
        numeric_offset = 357
        for field_index, field in enumerate(numeric_fields):
            value = pd.to_numeric(pd.Series([row[field]]), errors="coerce").iloc[0]
            matrix[output_index, numeric_offset + field_index * 2] = 0.0 if not np.isfinite(value) else float(value)
            matrix[output_index, numeric_offset + field_index * 2 + 1] = 0.0 if np.isfinite(value) else 1.0
    return matrix


def fit_feature_preprocess(raw: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = raw[train_mask, 357::2].mean(axis=0).astype(np.float32)
    std = raw[train_mask, 357::2].std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    output = raw.copy()
    output[:, 357::2] = np.clip((output[:, 357::2] - mean) / std, -8.0, 8.0)
    return output, mean, std


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval(); output=[]
    for start in range(0,len(x),2048):
        output.append(model(torch.from_numpy(x[start:start+2048]).to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(output)


def regression_metrics(target: np.ndarray, prediction: np.ndarray, task: dict[str, Any]) -> dict[str, float]:
    result={"mae":mae(target,prediction),"rmse":rmse(target,prediction),"r2":r2(target,prediction)}
    if task["target"]=="log10_decay_ns":
        factor=np.power(10.0,np.abs(target-prediction));result["factor2_accuracy"]=float(np.mean(factor<=2));result["factor10_accuracy"]=float(np.mean(factor<=10))
    return result


def train_network(
    family_id:str,fit_id:str,config:dict[str,Any],seed:int,x_train:np.ndarray,y_train:np.ndarray,
    x_val:np.ndarray,y_val:np.ndarray,y_mean:float,y_std:float,device:torch.device,thermal:ThermalGuard,
) -> tuple[RoleAwareNet,dict[str,Any],dict[str,Any]]:
    thermal.check();set_seed(seed);model=RoleAwareNet(y_mean,y_std,config).to(device)
    loader=DataLoader(TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train.astype(np.float32)[:,None])),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    best_state=None;best_mae=float("inf");history=[];stale=0;started=time.perf_counter()
    for epoch in range(60):
        model.train();losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device);optimizer.zero_grad(set_to_none=True);output=model(xb)
            loss=F.smooth_l1_loss((output-y_mean)/y_std,(yb-y_mean)/y_std);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        val_prediction=predict(model,x_val,device);score=mae(y_val,val_prediction);history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"val_mae":score})
        if score<best_mae-1e-5:best_mae=score;best_state=state_to_cpu(model);stale=0
        else:stale+=1
        if stale>=6 and epoch>=10:break
    if best_state is None:raise RuntimeError(f"no IPOP checkpoint {fit_id}")
    model.load_state_dict(best_state);model.to(device);selection={"mae":mae(y_val,predict(model,x_val,device)),"rmse":rmse(y_val,predict(model,x_val,device))}
    receipt=save_fit(FitResult(family_id,fit_id,config,seed,selection,best_state,time.perf_counter()-started,parameter_count(model),{"history":history,"epochs":len(history)}))
    return model,selection,receipt


def run_task(frame:pd.DataFrame,task:dict[str,Any],device:torch.device,thermal:ThermalGuard)->dict[str,Any]:
    family_id=task["family_id"];indices,splits,split_meta=component_split(frame,task["target"])
    raw=raw_features(frame,indices);train_mask=splits=="train";val_mask=splits=="val";test_mask=splits=="test"
    x,feature_mean,feature_std=fit_feature_preprocess(raw,train_mask)
    target=pd.to_numeric(frame.iloc[indices][task["target"]],errors="coerce").to_numpy(np.float32)
    y_mean=float(target[train_mask].mean());y_std=float(target[train_mask].std());y_std=max(y_std,1e-6)
    task_contract={
        "schema":"x5_icmat_foundry.post50_task_contract.v1","family_id":family_id,"task":task["name"],"target":task["target"],
        "allowed_features":["host composition","dopant identities/valencies/concentrations","temperature","declared excitation source"],
        "forbidden_post_measurement_features":["emission_max_nm","cie_x","cie_y","internal/external QE when not target","decay when not target","excitation maxima","monitoring energy"],
        "split":split_meta,"source_license":"CC BY 4.0","claim_boundary":task["claim"],"model_output":"one scientific regression value only",
    }
    write_json(CONTRACT_ROOT/f"tasks/{family_id}.json",task_contract,seal=True)
    baseline_models={};baseline_scores={}
    for name,model in (("ridge",Ridge(alpha=10.0)),("extra_trees",ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,max_features=0.7,n_jobs=4,random_state=SEED_BASE))):
        started=time.perf_counter();model.fit(x[train_mask],target[train_mask]);prediction=model.predict(x[val_mask]);selection=regression_metrics(target[val_mask],prediction,task)
        save_fit(FitResult(family_id,f"baseline_{name}",{"kind":name},SEED_BASE,selection,None,time.perf_counter()-started,None,{"baseline":True}),keep_checkpoint=False)
        baseline_models[name]=model;baseline_scores[name]=selection
    config_scores={config["id"]:[] for config in CONFIGS}
    for ci,config in enumerate(CONFIGS):
        for repeat in range(2):
            _,selection,_=train_network(family_id,f"grid_{config['id']}_r{repeat}",config,SEED_BASE+9000+ci*10+repeat,x[train_mask],target[train_mask],x[val_mask],target[val_mask],y_mean,y_std,device,thermal)
            config_scores[config["id"]].append(float(selection["mae"]))
    ranked=sorted(CONFIGS,key=lambda config:float(np.mean(config_scores[config["id"]])))
    variants=[("quality",copy.deepcopy(ranked[0]),SEED_BASE+9900),("lite",next(config for config in CONFIGS if config["id"]=="v5_lite"),SEED_BASE+9901)]
    models={}
    for name,config,seed in variants:
        model,_,_=train_network(family_id,f"final_{name}_{config['id']}",config,seed,x[train_mask],target[train_mask],x[val_mask],target[val_mask],y_mean,y_std,device,thermal);models[name]=model.cpu().eval()
    lock={"schema":"x5_icmat_foundry.post50_selection_lock.v1","created_at":utc_now(),"family_id":family_id,"fit_count_before_test":16,"ranked_configuration_ids":[config["id"] for config in ranked],"test_opened":False,"post_test_retuning_allowed":False}
    write_json(CONTRACT_ROOT/f"selection_locks/{family_id}.json",lock,seal=True)
    median=float(np.median(target[train_mask]));locked={"median":{"mae":mae(target[test_mask],np.full(np.sum(test_mask),median))}}
    for name,model in baseline_models.items():locked[name]=regression_metrics(target[test_mask],model.predict(x[test_mask]),task)
    exports={}
    for name,model in models.items():
        prediction=predict(model.to(device),x[test_mask],device);locked[name]=regression_metrics(target[test_mask],prediction,task)
        exports[name]=export_static_onnx(family_id,name,model.cpu(),[torch.from_numpy(x[test_mask][:1].reshape(1,1,1,FEATURE_WIDTH))],["presynthesis_features_fp32"],["property_prediction"],{"numeric_mean":feature_mean,"numeric_std":feature_std},{"configuration":next(item[1] for item in variants if item[0]==name),"seed":next(item[2] for item in variants if item[0]==name)})
    quality=locked["quality"];best_simple=min(locked["median"]["mae"],locked["ridge"]["mae"],locked["extra_trees"]["mae"])
    gate={"beats_median":quality["mae"]<locked["median"]["mae"],"beats_best_trained_simple":quality["mae"]<best_simple};gate["pass"]=all(gate.values())
    return write_family_receipt(family_id,{"model_name":task["name"],"fit_count":16,"fit_count_contract_pass":True,"task_contract":task_contract,"selection_lock":lock,"locked_test":locked,"validation_baselines":baseline_scores,"innovation_gate":gate,"candidate_class":"RND_VALIDATED" if gate["pass"] else "RND_USABLE_EXPERIMENTAL","exports":exports,"claim_boundary":[task["claim"],"strict pre-measurement feature contract","no audit, routing, or execution authority","PC metrics are not local experimental or X5 evidence"]})


def main()->int:
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");thermal=ThermalGuard();frame=pd.read_csv(SOURCE)
    receipts=[]
    for task in TASKS:receipts.append(run_task(frame,task,device,thermal))
    print(json.dumps({"families":[item["family_id"] for item in receipts],"fit_count":32,"receipts":[item["path"] for item in receipts]},ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
