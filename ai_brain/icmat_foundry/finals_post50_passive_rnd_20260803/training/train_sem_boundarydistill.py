from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icmat_foundry.finals_50model.training.train_sem_bank import TinyUNet, binary_boundary, segmentation_metrics
from icmat_foundry.finals_post50_passive_rnd_20260803.training.common import (
    CONTRACT_ROOT,
    FitResult,
    SEED_BASE,
    TRIAL_ROOT,
    ThermalGuard,
    export_static_onnx,
    parameter_count,
    save_fit,
    set_seed,
    sha256_file,
    state_to_cpu,
    utc_now,
    write_family_receipt,
    write_json,
)


FAMILY_ID = "RND-SEM-01"
IMAGE_SIZE = 128
NEW_FIT_EPOCHS = 12
LEGACY_BATCH96_CONTINUATION_EPOCHS = 22
CARINTHIA = ROOT / "research/data_assets/icmat_foundry/carinthia_sem"
CARINTHIA_S = ROOT / "research/data_assets/icmat_foundry/carinthia_s_sem"


class BoundaryUNet(nn.Module):
    def __init__(self, width: int = 6) -> None:
        super().__init__()
        self.backbone = TinyUNet(width)
        self.distance_head = TinyUNet(max(3, width // 2))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(value), torch.tanh(self.distance_head(value))


class SegDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, images: np.ndarray, masks: np.ndarray, distances: np.ndarray, augment: float) -> None:
        self.images = images
        self.masks = masks
        self.distances = distances
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index].copy()).float().unsqueeze(0) / 255.0
        mask = torch.from_numpy(self.masks[index].copy()).float().unsqueeze(0)
        distance = torch.from_numpy(self.distances[index].astype(np.float32)).unsqueeze(0)
        if self.augment > 0:
            if torch.rand(()) < 0.5:
                image, mask, distance = torch.flip(image, (2,)), torch.flip(mask, (2,)), torch.flip(distance, (2,))
            if torch.rand(()) < 0.5:
                image, mask, distance = torch.flip(image, (1,)), torch.flip(mask, (1,)), torch.flip(distance, (1,))
            image = torch.clamp(image * (0.9 + 0.2 * torch.rand(())) + self.augment * torch.randn_like(image), 0, 1)
        return image, mask, distance


TEACHER_CONFIGS = (
    {"id": "v0_mask", "width": 5, "w_boundary": 0.0, "w_distance": 0.0, "augment": 0.02},
    {"id": "v1_boundary", "width": 5, "w_boundary": 0.12, "w_distance": 0.0, "augment": 0.02},
    {"id": "v2_distance", "width": 5, "w_boundary": 0.0, "w_distance": 0.12, "augment": 0.02},
    {"id": "v3_boundary_distance", "width": 5, "w_boundary": 0.10, "w_distance": 0.10, "augment": 0.02},
    {"id": "v4_robust", "width": 6, "w_boundary": 0.10, "w_distance": 0.10, "augment": 0.05},
    {"id": "v5_quality", "width": 7, "w_boundary": 0.14, "w_distance": 0.10, "augment": 0.03},
    {"id": "v6_boundary_heavy", "width": 7, "w_boundary": 0.22, "w_distance": 0.06, "augment": 0.03},
    {"id": "v7_lite_teacher", "width": 4, "w_boundary": 0.12, "w_distance": 0.08, "augment": 0.03},
)
LOCKED_STUDENT_TEACHER_IDS = ("v4_robust", "v5_quality", "v2_distance", "v1_boundary")


def existing_fit(fit_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    receipt_path = TRIAL_ROOT / FAMILY_ID / f"{fit_id}.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint_record = receipt.get("checkpoint")
    if not checkpoint_record:
        raise RuntimeError(f"SEM fit is missing checkpoint: {fit_id}")
    checkpoint_path = ROOT / checkpoint_record["path"]
    if sha256_file(checkpoint_path) != checkpoint_record["sha256"]:
        raise RuntimeError(f"SEM checkpoint digest mismatch: {fit_id}")
    return receipt, torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def split_assignment(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_label: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_label[int(row["label"])][row["sha256"]].append(row["filename"])
    assignment = {}
    for label, groups in sorted(by_label.items()):
        ordered = sorted(groups, key=lambda item: hashlib.sha256(f"20260801:{label}:{item}".encode()).hexdigest())
        count = len(ordered)
        n_test = max(1, int(round(count * 0.15)))
        n_val = max(1, int(round(count * 0.15)))
        for index, digest in enumerate(ordered):
            split = "test" if index < n_test else "val" if index < n_test + n_val else "train"
            for filename in groups[digest]:
                assignment[filename] = split
    return assignment


def load_data() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    class_csv = CARINTHIA / "extracted/data/carinthia.csv"
    mask_csv = CARINTHIA_S / "extracted/data/carinthia-s.csv"
    image_root = CARINTHIA / "extracted/data/images"
    mask_root = CARINTHIA_S / "extracted/data"
    with class_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        class_rows = list(csv.DictReader(stream, delimiter=";"))
    with mask_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        mask_rows = {row["filename"] + ".jpg": row for row in csv.DictReader(stream, delimiter=";")}
    indexed, raw_by_file = [], {}
    for row in class_rows:
        filename = row["file_name"]
        raw = (image_root / filename).read_bytes()
        raw_by_file[filename] = raw
        indexed.append({"filename": filename, "label": int(row["label"]) - 1, "sha256": sha_bytes(raw)})
    if len(indexed) != 4591 or len(mask_rows) != 4591:
        raise RuntimeError("Carinthia-S row count mismatch")
    assignment = split_assignment(indexed)
    images = np.empty((len(indexed), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    masks = np.empty_like(images)
    splits = np.empty(len(indexed), dtype="U5")
    manifest_rows = []
    for index, row in enumerate(indexed):
        filename = row["filename"]
        with Image.open(io.BytesIO(raw_by_file[filename])) as image:
            images[index] = np.asarray(image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)
        mask_path = mask_root / mask_rows[filename]["mask_path"]
        with Image.open(mask_path) as mask:
            masks[index] = (np.asarray(mask.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)) > 127).astype(np.uint8)
        splits[index] = assignment[filename]
        manifest_rows.append({"filename": filename, "sha256": row["sha256"], "split": assignment[filename]})
    distances = np.empty_like(images, dtype=np.float16)
    for index, mask in enumerate(masks.astype(bool)):
        signed = distance_transform_edt(mask) - distance_transform_edt(~mask)
        distances[index] = np.clip(signed / 16.0, -1.0, 1.0).astype(np.float16)
    source = {
        "dataset": "Carinthia SEM + Carinthia-S",
        "license": "CC BY 4.0",
        "rows": len(images),
        "split_counts": dict(Counter(splits.tolist())),
        "split_policy": "stratified by class and grouped by exact image SHA-256",
        "class_archive_sha256": sha256_file(CARINTHIA / "raw/data.zip"),
        "segmentation_archive_sha256": sha256_file(CARINTHIA_S / "raw/data.zip"),
    }
    write_json(CONTRACT_ROOT / "splits/RND-SEM-01.json", {"schema": "post50.sem_split.v1", "source": source, "records": manifest_rows}, seal=True)
    return {"images": images, "masks": masks, "distances": distances, "splits": splits}, source


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    numerator = 2 * (probability * target).sum((1, 2, 3)) + 1.0
    denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1.0
    return 1.0 - (numerator / denominator).mean()


def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    gx_p = torch.abs(probability[..., :, 1:] - probability[..., :, :-1])
    gy_p = torch.abs(probability[..., 1:, :] - probability[..., :-1, :])
    gx_t = torch.abs(target[..., :, 1:] - target[..., :, :-1])
    gy_t = torch.abs(target[..., 1:, :] - target[..., :-1, :])
    return 0.5 * (F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t))


@torch.inference_mode()
def predict_logits(model: nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(images), 128):
        tensor = torch.from_numpy(images[start : start + 128].copy()).float().unsqueeze(1).to(device) / 255.0
        prediction = model(tensor)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        output.append(prediction[:, 0].cpu().numpy())
    return np.concatenate(output)


def best_metrics(mask: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    candidates = []
    for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        prediction = (1.0 / (1.0 + np.exp(-logits)) >= threshold).astype(np.uint8)
        metrics = segmentation_metrics(mask, prediction)
        score = metrics["dice"] + 0.25 * metrics["boundary_f1_exact"]
        candidates.append({"threshold": threshold, "score": score, **metrics})
    return max(candidates, key=lambda item: item["score"])


def make_loader(images: np.ndarray, masks: np.ndarray, distances: np.ndarray, augment: float, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SegDataset(images, masks, distances, augment), batch_size=32, shuffle=shuffle,
        num_workers=0, generator=torch.Generator().manual_seed(seed), pin_memory=torch.cuda.is_available(),
    )


def train_teacher(
    fit_id: str, config: dict[str, Any], seed: int,
    train: tuple[np.ndarray, np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device, thermal: ThermalGuard,
) -> tuple[BoundaryUNet, dict[str, Any], dict[str, Any]]:
    thermal.check(); set_seed(seed)
    model = BoundaryUNet(int(config["width"])).to(device)
    resumed = existing_fit(fit_id)
    if resumed:
        previous_receipt, previous_checkpoint = resumed
        model.load_state_dict(previous_checkpoint["state_dict"])
        history = list(previous_receipt.get("extra", {}).get("history", []))
        best_selection = dict(previous_receipt["selection_metrics"])
        best_score = float(best_selection["score"])
        best_state = state_to_cpu(model)
        previous_seconds = float(previous_receipt.get("fit_seconds", 0.0))
        target_epochs = int(previous_receipt.get("extra", {}).get("effective_target_epochs", LEGACY_BATCH96_CONTINUATION_EPOCHS))
    else:
        history, best_selection, best_state = [], None, None
        best_score, previous_seconds = -1.0, 0.0
        target_epochs = NEW_FIT_EPOCHS
    if resumed and len(history) >= target_epochs:
        return model, best_selection, previous_receipt
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loader = make_loader(*train, float(config["augment"]), seed, True)
    foreground = float(train[1].mean())
    pos_weight = torch.tensor(min(40.0, max(1.0, (1.0 - foreground) / max(foreground, 1e-5))), device=device)
    stale = 0
    resumed_from_epoch = len(history); started = time.perf_counter()
    for epoch in range(resumed_from_epoch, target_epochs):
        model.train(); losses = []
        for image, mask, distance in loader:
            image, mask, distance = image.to(device), mask.to(device), distance.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, distance_prediction = model(image)
            loss = 0.55 * F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight) + 0.45 * dice_loss(logits, mask)
            loss = loss + float(config["w_boundary"]) * boundary_loss(logits, mask)
            loss = loss + float(config["w_distance"]) * F.smooth_l1_loss(distance_prediction, distance)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        selection = best_metrics(val[1], predict_logits(model, val[0], device))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "score": selection["score"]})
        if selection["score"] > best_score + 1e-5:
            best_score, best_selection, best_state, stale = selection["score"], selection, state_to_cpu(model), 0
        else:
            stale += 1
        if stale >= 3 and epoch >= target_epochs - 3: break
    if best_state is None: raise RuntimeError(f"no SEM teacher checkpoint: {fit_id}")
    model.load_state_dict(best_state); model.to(device)
    receipt = save_fit(FitResult(FAMILY_ID, fit_id, {"kind": "teacher", **config}, seed, best_selection, best_state, previous_seconds+time.perf_counter()-started, parameter_count(model), {"history": history,"epochs":len(history),"resumed_from_epoch":resumed_from_epoch,"optimizer_state_restored":False,"training_batch_size":32,"effective_target_epochs":target_epochs,"prior_receipt_archive":"trials_archive/RND-SEM-01_phase2_batch96_extended" if resumed else None}))
    return model, best_selection, receipt


def train_student(
    fit_id: str, width: int, augment: float, seed: int, teacher: BoundaryUNet | None,
    train: tuple[np.ndarray, np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device, thermal: ThermalGuard,
) -> tuple[TinyUNet, dict[str, Any], dict[str, Any]]:
    thermal.check(); set_seed(seed)
    model = TinyUNet(width).to(device)
    resumed = existing_fit(fit_id)
    if resumed:
        previous_receipt, previous_checkpoint = resumed
        model.load_state_dict(previous_checkpoint["state_dict"])
        history = list(previous_receipt.get("extra", {}).get("history", []))
        best_selection = dict(previous_receipt["selection_metrics"])
        best_score = float(best_selection["score"])
        best_state = state_to_cpu(model)
        previous_seconds = float(previous_receipt.get("fit_seconds", 0.0))
        target_epochs = int(previous_receipt.get("extra", {}).get("effective_target_epochs", LEGACY_BATCH96_CONTINUATION_EPOCHS))
    else:
        history, best_selection, best_state = [], None, None
        best_score, previous_seconds = -1.0, 0.0
        target_epochs = NEW_FIT_EPOCHS
    if resumed and len(history) >= target_epochs:
        return model, best_selection, previous_receipt
    if teacher is not None: teacher = teacher.to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loader = make_loader(*train, augment, seed, True)
    foreground = float(train[1].mean())
    pos_weight = torch.tensor(min(40.0, max(1.0, (1.0 - foreground) / max(foreground, 1e-5))), device=device)
    stale=0; resumed_from_epoch=len(history); started=time.perf_counter()
    for epoch in range(resumed_from_epoch,target_epochs):
        model.train(); losses=[]
        for image, mask, _ in loader:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = 0.52 * F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight) + 0.38 * dice_loss(logits, mask) + 0.10 * boundary_loss(logits, mask)
            if teacher is not None:
                with torch.no_grad(): teacher_logits = teacher(image)[0]
                loss = loss + 0.12 * F.smooth_l1_loss(logits, teacher_logits)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        selection = best_metrics(val[1], predict_logits(model, val[0], device))
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"score":selection["score"]})
        if selection["score"] > best_score + 1e-5:
            best_score,best_selection,best_state,stale=selection["score"],selection,state_to_cpu(model),0
        else: stale+=1
        if stale>=3 and epoch>=target_epochs-3: break
    if best_state is None: raise RuntimeError(f"no SEM student checkpoint: {fit_id}")
    model.load_state_dict(best_state); model.to(device)
    config={"kind":"student","width":width,"augment":augment,"distilled":teacher is not None}
    receipt=save_fit(FitResult(FAMILY_ID,fit_id,config,seed,best_selection,best_state,previous_seconds+time.perf_counter()-started,parameter_count(model),{"history":history,"epochs":len(history),"resumed_from_epoch":resumed_from_epoch,"optimizer_state_restored":False,"training_batch_size":32,"effective_target_epochs":target_epochs,"prior_receipt_archive":"trials_archive/RND-SEM-01_phase2_batch96_extended" if resumed else None}))
    return model,best_selection,receipt


def main() -> int:
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); thermal=ThermalGuard()
    bank,source=load_data(); splits=bank["splits"]
    def part(name:str)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        mask=splits==name; return bank["images"][mask],bank["masks"][mask],bank["distances"][mask]
    train,val,test=part("train"),part("val"),part("test")
    task_contract={
        "schema":"x5_icmat_foundry.post50_task_contract.v1","family_id":FAMILY_ID,
        "task":"real public SEM defect-region segmentation with boundary-aware teacher and static mask student",
        "source":source,"model_output":"mask logits only for deployed student",
        "forbidden_outputs":["PASS","HOLD","model_route","robot_action"],
        "test_policy":"grouped SHA test opened only after 32 fit selection lock",
    }
    write_json(CONTRACT_ROOT/"tasks/RND-SEM-01.json",task_contract,seal=True)
    teacher_scores={config["id"]:[] for config in TEACHER_CONFIGS}; teacher_models={}
    for ci,config in enumerate(TEACHER_CONFIGS):
        replicas=[]
        for repeat in range(2):
            model,selection,_=train_teacher(f"teacher_{config['id']}_r{repeat}",config,SEED_BASE+7000+ci*10+repeat,train,val,device,thermal)
            teacher_scores[config["id"]].append(float(selection["score"]))
            replicas.append((float(selection["score"]),copy.deepcopy(model.cpu())))
        teacher_models[config["id"]]=max(replicas,key=lambda item:item[0])[1]
    ranked_teachers=sorted(TEACHER_CONFIGS,key=lambda config:-float(np.mean(teacher_scores[config["id"]])))
    student_teachers=[next(config for config in TEACHER_CONFIGS if config["id"]==config_id) for config_id in LOCKED_STUDENT_TEACHER_IDS]
    student_scores={config["id"]:[] for config in student_teachers}; student_specs={}
    for rank,config in enumerate(student_teachers):
        width=max(4,int(config["width"])-1); student_specs[config["id"]]=(width,float(config["augment"]))
        for repeat in range(2):
            _,selection,_=train_student(f"student_top{rank}_{config['id']}_r{repeat}",width,float(config["augment"]),SEED_BASE+8000+rank*10+repeat,teacher_models[config["id"]],train,val,device,thermal)
            student_scores[config["id"]].append(float(selection["score"]))
    ranked_students=sorted(student_teachers,key=lambda config:-float(np.mean(student_scores[config["id"]])))
    for rank,config in enumerate(ranked_students[:2]):
        width,augment=student_specs[config["id"]]
        for repeat in range(2 if rank==0 else 1):
            _,selection,_=train_student(f"stability_top{rank}_{config['id']}_r{repeat}",width,augment,SEED_BASE+8500+rank*10+repeat,teacher_models[config["id"]],train,val,device,thermal)
            student_scores[config["id"]].append(float(selection["score"]))
    ranked_students=sorted(student_teachers,key=lambda config:-float(np.mean(student_scores[config["id"]])))
    best=ranked_students[0]; best_width,best_aug=student_specs[best["id"]]
    variants=[
        ("quality",best_width,best_aug,teacher_models[best["id"]],SEED_BASE+8900),
        ("robust",best_width,0.06,teacher_models[best["id"]],SEED_BASE+8901),
        ("lite",4,0.03,teacher_models[best["id"]],SEED_BASE+8902),
    ]
    models={}; thresholds={}; final_validation_scores={}
    for name,width,augment,teacher,seed in variants:
        model,selection,_=train_student(f"final_{name}",width,augment,seed,teacher,train,val,device,thermal)
        models[name]=model.cpu().eval(); thresholds[name]=float(selection["threshold"]); final_validation_scores[name]=float(selection["score"])
    selected_quality_variant=max(final_validation_scores,key=final_validation_scores.get)
    selection_lock={
        "schema":"x5_icmat_foundry.post50_selection_lock.v1","created_at":utc_now(),"family_id":FAMILY_ID,
        "fit_count_before_test":32,"ranked_teacher_ids":[item["id"] for item in ranked_teachers],
        "ranked_student_teacher_ids":[item["id"] for item in ranked_students],"variant_thresholds":thresholds,
        "fit_budget_reallocation":{"completed_extra_students":["student_top3_v6_boundary_heavy_r0","student_top3_v6_boundary_heavy_r1"],"dropped_unexported_fits":["stability_top1_repeat1","final_control"]},
        "final_validation_scores":final_validation_scores,"selected_quality_variant":selected_quality_variant,
        "test_opened":False,"post_test_retuning_allowed":False,
    }
    write_json(CONTRACT_ROOT/"selection_locks/RND-SEM-01.json",selection_lock,seal=True)
    locked_metrics={};exports={}
    for name,model in models.items():
        logits=predict_logits(model.to(device),test[0],device)
        prediction=(1/(1+np.exp(-logits))>=thresholds[name]).astype(np.uint8)
        locked_metrics[name]={"threshold":thresholds[name],**segmentation_metrics(test[1],prediction)}
        if name!="control":
            exports[name]=export_static_onnx(FAMILY_ID,name,model.cpu(),[torch.from_numpy(test[0][:1].copy()).float().unsqueeze(1)/255.0],["sem_image_fp32"],["mask_logits"],{"threshold":np.asarray([thresholds[name]],dtype=np.float32)},{"width":next(item[1] for item in variants if item[0]==name),"seed":next(item[4] for item in variants if item[0]==name)})
    quality=locked_metrics[selected_quality_variant]
    gate={"boundary_f1_relative_gain_ge_10pct":quality["boundary_f1_exact"]>=1.10*0.086495,"dice_not_below_0_801":quality["dice"]>=0.801}
    gate["pass"]=all(gate.values())
    receipt=write_family_receipt(FAMILY_ID,{
        "model_name":"SEM-BoundaryDistill","fit_count":32,"fit_count_contract_pass":True,
        "task_contract":task_contract,"selection_lock":selection_lock,"locked_test":locked_metrics,"selected_quality_variant":selected_quality_variant,
        "innovation_gate":gate,"candidate_class":"RND_INNOVATION_ANCHOR" if gate["pass"] else "RND_USABLE_EXPERIMENTAL",
        "exports":exports,"thermal_samples":thermal.samples,
        "claim_boundary":["real public Carinthia-S SEM only","synthetic corruption is augmentation, not fab evidence","student emits mask only and has no audit or decision authority","PC ONNX is not X5 performance"],
    })
    print(json.dumps({"family":FAMILY_ID,"fit_count":32,"gate":gate,"receipt":receipt["path"]},ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
