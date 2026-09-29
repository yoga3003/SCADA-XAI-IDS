
# ============================================================
# TRUST-AWARE XAI SCADA IDS — COMPLETE GOOGLE COLAB PIPELINE
# Dataset: WUSTL-IIOT-2018 processed CSV
# ============================================================
# Outputs:
#   results/*.csv
#   results/*.json
#   results/figures/*.png
#   results/models/*
#
# IMPORTANT:
# 1. Upload wustl-scada-2018.csv when prompted, OR mount Drive.
# 2. All reported paper numbers must come from the generated files.
# 3. The GAN result is not assumed to improve performance.
# 4. The perturbation model is a controlled stress test, not a named
#    real-world attack.
# ============================================================

# ---------------- CELL 1: Install ----------------
!pip -q install xgboost shap scipy scikit-learn pandas numpy matplotlib seaborn tensorflow

# ---------------- CELL 2: Imports/config ----------------
import os, json, time, random, warnings, hashlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from xgboost import XGBClassifier
from scipy.stats import spearmanr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEEDS = [42, 123, 2026, 7, 99]
PRIMARY_SEED = 42

FEATURES = ["Sport","TotPkts","TotBytes","SrcPkts","DstPkts","SrcBytes"]
TARGET = "Target"

RESULT_DIR = "/content/results"
FIG_DIR = os.path.join(RESULT_DIR, "figures")
MODEL_DIR = os.path.join(RESULT_DIR, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(PRIMARY_SEED)

# ---------------- CELL 3: Upload dataset ----------------
from google.colab import files

uploaded = files.upload()
CSV_PATH = next((k for k in uploaded if k.lower().endswith(".csv")), None)
if CSV_PATH is None:
    raise FileNotFoundError("Upload the WUSTL-IIOT-2018 CSV file.")

print("Using:", CSV_PATH)

# ---------------- CELL 4: Load + verify ----------------
df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

required = FEATURES + [TARGET]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

df = df[required].copy()

# Numeric conversion
for c in required:
    df[c] = pd.to_numeric(df[c], errors="coerce")

before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
after_clean = len(df)

df[TARGET] = df[TARGET].astype(int)
if not set(df[TARGET].unique()).issubset({0,1}):
    raise ValueError("Target must be binary 0/1.")

# Exact duplicate leakage audit
duplicate_count = df.duplicated(keep=False).sum()
unique_rows = len(df.drop_duplicates())

audit = {
    "raw_rows": int(before),
    "rows_after_cleaning": int(after_clean),
    "columns": required,
    "exact_duplicate_rows": int(duplicate_count),
    "unique_complete_rows": int(unique_rows),
    "normal_rows": int((df[TARGET]==0).sum()),
    "attack_rows": int((df[TARGET]==1).sum()),
    "missing_values_after_cleaning": int(df.isna().sum().sum()),
}
print(json.dumps(audit, indent=2))

with open(f"{RESULT_DIR}/dataset_audit.json","w") as f:
    json.dump(audit,f,indent=2)

# Principal analysis set: exact duplicate removal
df_u = df.drop_duplicates().reset_index(drop=True)

X = df_u[FEATURES].astype(np.float32)
y = df_u[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=PRIMARY_SEED
)

print("Deduplicated:", len(df_u))
print("Train:", len(X_train), "Test:", len(X_test))

# ---------------- CELL 5: Metrics helper ----------------
def metrics_row(name, y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true,y_pred),
        "precision": precision_score(y_true,y_pred,zero_division=0),
        "recall": recall_score(y_true,y_pred,zero_division=0),
        "f1": f1_score(y_true,y_pred,zero_division=0),
        "roc_auc": roc_auc_score(y_true,y_prob),
        "pr_auc": average_precision_score(y_true,y_prob),
        "tn": int(confusion_matrix(y_true,y_pred).ravel()[0]),
        "fp": int(confusion_matrix(y_true,y_pred).ravel()[1]),
        "fn": int(confusion_matrix(y_true,y_pred).ravel()[2]),
        "tp": int(confusion_matrix(y_true,y_pred).ravel()[3]),
    }

# ---------------- CELL 6: E1 classical baselines ----------------
models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=PRIMARY_SEED))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=1,
        class_weight="balanced_subsample", n_jobs=-1, random_state=PRIMARY_SEED
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
        random_state=PRIMARY_SEED
    ),
    "XGBoost": XGBClassifier(
        n_estimators=350, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", tree_method="hist",
        random_state=PRIMARY_SEED, n_jobs=-1
    )
}

baseline_rows=[]
fitted={}
for name,m in models.items():
    t=time.time()
    m.fit(X_train,y_train)
    prob=m.predict_proba(X_test)[:,1]
    row=metrics_row(name,y_test,prob)
    row["train_seconds"]=time.time()-t
    baseline_rows.append(row)
    fitted[name]=m
    print(name,row)

baseline_df=pd.DataFrame(baseline_rows)
baseline_df.to_csv(f"{RESULT_DIR}/E1_baselines.csv",index=False)

# Confusion matrix for RF
rf=fitted["Random Forest"]
rf_prob=rf.predict_proba(X_test)[:,1]
cm=confusion_matrix(y_test,(rf_prob>=.5).astype(int))
plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt="d",cbar=False)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig1_rf_confusion_matrix.png",dpi=300)
plt.show()

# ---------------- CELL 7: DNN baseline ----------------
scaler=StandardScaler()
Xtr_s=scaler.fit_transform(X_train)
Xte_s=scaler.transform(X_test)

dnn=Pipeline([
    ("scale", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(128,64,32),
                          activation="relu", solver="adam",
                          alpha=1e-4, batch_size=256,
                          learning_rate_init=1e-3,
                          max_iter=80, early_stopping=True,
                          validation_fraction=.15,
                          random_state=PRIMARY_SEED))
])
t=time.time()
dnn.fit(X_train,y_train)
dnn_prob=dnn.predict_proba(X_test)[:,1]
dnn_row=metrics_row("DNN",y_test,dnn_prob)
dnn_row["train_seconds"]=time.time()-t
pd.DataFrame([dnn_row]).to_csv(f"{RESULT_DIR}/E1_DNN.csv",index=False)
print(dnn_row)

# ---------------- CELL 8: Stacking-style RF + DNN hybrid ----------------
# OOF RF probabilities prevent RF target leakage into DNN training.
skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=PRIMARY_SEED)
oof_rf=np.zeros(len(X_train))

for tr_idx,va_idx in skf.split(X_train,y_train):
    rf_cv=RandomForestClassifier(
        n_estimators=250,class_weight="balanced_subsample",
        n_jobs=-1,random_state=PRIMARY_SEED
    )
    rf_cv.fit(X_train.iloc[tr_idx],y_train.iloc[tr_idx])
    oof_rf[va_idx]=rf_cv.predict_proba(X_train.iloc[va_idx])[:,1]

rf_full=RandomForestClassifier(
    n_estimators=400,class_weight="balanced_subsample",
    n_jobs=-1,random_state=PRIMARY_SEED
)
rf_full.fit(X_train,y_train)
test_rf_prob=rf_full.predict_proba(X_test)[:,1]

hyb_train=np.column_stack([StandardScaler().fit_transform(X_train),oof_rf])
# Use one shared scaler for the six features
hyb_scaler=StandardScaler()
Xtr_h=np.column_stack([hyb_scaler.fit_transform(X_train),oof_rf])
Xte_h=np.column_stack([hyb_scaler.transform(X_test),test_rf_prob])

hyb=MLPClassifier(hidden_layer_sizes=(128,64,32),max_iter=100,
                  early_stopping=True,validation_fraction=.15,
                  random_state=PRIMARY_SEED,batch_size=256)
hyb.fit(Xtr_h,y_train)
hyb_prob=hyb.predict_proba(Xte_h)[:,1]
hyb_row=metrics_row("RF+DNN Stacking Hybrid",y_test,hyb_prob)
pd.DataFrame([hyb_row]).to_csv(f"{RESULT_DIR}/E1_hybrid.csv",index=False)
print(hyb_row)

# ---------------- CELL 9: ROC/PR curves ----------------
plt.figure(figsize=(6,5))
for name,m in fitted.items():
    p=m.predict_proba(X_test)[:,1]
    fpr,tpr,_=roc_curve(y_test,p)
    plt.plot(fpr,tpr,label=name)
fpr,tpr,_=roc_curve(y_test,dnn_prob); plt.plot(fpr,tpr,label="DNN")
plt.plot([0,1],[0,1],"--",label="Chance")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_roc_curves.png",dpi=300); plt.show()

plt.figure(figsize=(6,5))
for name,m in fitted.items():
    p=m.predict_proba(X_test)[:,1]
    pr,re,_=precision_recall_curve(y_test,p)
    plt.plot(re,pr,label=name)
pr,re,_=precision_recall_curve(y_test,dnn_prob)
plt.plot(re,pr,label="DNN")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curves"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig3_pr_curves.png",dpi=300); plt.show()

# ---------------- CELL 10: E2 limited-data GAN ----------------
# Uses ONLY the training partition.
# We sample 10,000 training rows (or all if smaller), preserving class ratio.
rng=np.random.default_rng(PRIMARY_SEED)
n_limited=min(10000,len(X_train))
idx=rng.choice(len(X_train),size=n_limited,replace=False)
X_lim=X_train.iloc[idx].copy()
y_lim=y_train.iloc[idx].copy()

# Train GAN only on attack samples.
attack_lim=X_lim[y_lim.values==1].copy()
if len(attack_lim)<50:
    raise ValueError("Too few attack samples in 10k subset for GAN experiment.")

# log1p all nonnegative traffic features; Sport is also nonnegative.
attack_log=np.log1p(attack_lim.values.astype(np.float32))
gan_scaler=MinMaxScaler(feature_range=(-1,1))
attack_scaled=gan_scaler.fit_transform(attack_log)

LATENT=32
DIM=len(FEATURES)

def build_generator():
    return keras.Sequential([
        layers.Input(shape=(LATENT,)),
        layers.Dense(64,activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(128,activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(256,activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(DIM,activation="tanh")
    ])

def build_discriminator():
    return keras.Sequential([
        layers.Input(shape=(DIM,)),
        layers.Dense(256,activation="relu"),
        layers.Dropout(.2),
        layers.Dense(128,activation="relu"),
        layers.Dropout(.2),
        layers.Dense(64,activation="relu"),
        layers.Dense(1,activation="sigmoid")
    ])

generator=build_generator()
discriminator=build_discriminator()
bce=keras.losses.BinaryCrossentropy()
g_opt=keras.optimizers.Adam(2e-4,beta_1=.5)
d_opt=keras.optimizers.Adam(2e-4,beta_1=.5)

BATCH=128
EPOCHS=300
gan_ds=tf.data.Dataset.from_tensor_slices(attack_scaled.astype(np.float32)).shuffle(
    len(attack_scaled),seed=PRIMARY_SEED).batch(BATCH,drop_remainder=False)

g_losses=[]; d_losses=[]
for epoch in range(EPOCHS):
    gl=[]; dl=[]
    for real in gan_ds:
        bs=tf.shape(real)[0]
        z=tf.random.normal((bs,LATENT))
        with tf.GradientTape() as dt:
            fake=generator(z,training=True)
            real_out=discriminator(real,training=True)
            fake_out=discriminator(fake,training=True)
            d_loss=bce(tf.ones_like(real_out)*.9,real_out)+bce(tf.zeros_like(fake_out),fake_out)
        d_grads=dt.gradient(d_loss,discriminator.trainable_variables)
        d_opt.apply_gradients(zip(d_grads,discriminator.trainable_variables))

        z=tf.random.normal((bs,LATENT))
        with tf.GradientTape() as gt:
            fake=generator(z,training=True)
            fake_out=discriminator(fake,training=True)
            g_loss=bce(tf.ones_like(fake_out),fake_out)
        g_grads=gt.gradient(g_loss,generator.trainable_variables)
        g_opt.apply_gradients(zip(g_grads,generator.trainable_variables))
        gl.append(float(g_loss)); dl.append(float(d_loss))
    g_losses.append(np.mean(gl)); d_losses.append(np.mean(dl))

pd.DataFrame({"epoch":np.arange(1,EPOCHS+1),"g_loss":g_losses,"d_loss":d_losses}).to_csv(
    f"{RESULT_DIR}/E2_GAN_losses.csv",index=False)

plt.figure(figsize=(7,4))
plt.plot(g_losses,label="Generator")
plt.plot(d_losses,label="Discriminator")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("GAN Training Loss")
plt.legend(); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_gan_losses.png",dpi=300); plt.show()

# Generate synthetic attacks equal to attack count in limited set.
n_syn=len(attack_lim)
z=tf.random.normal((n_syn,LATENT))
syn_scaled=generator(z,training=False).numpy()
syn_log=gan_scaler.inverse_transform(syn_scaled)
syn=np.expm1(syn_log)
syn=np.maximum(syn,0)

syn_df=pd.DataFrame(syn,columns=FEATURES)
syn_df[TARGET]=1

# Synthetic fidelity audit
real_stats=attack_lim[FEATURES].describe().T[["min","max","mean","std"]]
syn_stats=syn_df[FEATURES].describe().T[["min","max","mean","std"]]
quality=real_stats.join(syn_stats,lsuffix="_real",rsuffix="_synthetic")
quality["mean_ratio"]=quality["mean_synthetic"]/(quality["mean_real"].replace(0,np.nan))
quality.to_csv(f"{RESULT_DIR}/E2_GAN_quality.csv")

# Train limited-data RF on real-only and GAN augmented.
rf_10k=RandomForestClassifier(n_estimators=400,class_weight="balanced_subsample",
                              n_jobs=-1,random_state=PRIMARY_SEED)
rf_10k.fit(X_lim,y_lim)
p_real=rf_10k.predict_proba(X_test)[:,1]
r_real=metrics_row("RF_10k_real_only",y_test,p_real)

aug_X=pd.concat([X_lim,pd.DataFrame(syn,columns=FEATURES)],ignore_index=True)
aug_y=pd.concat([y_lim.reset_index(drop=True),pd.Series(np.ones(n_syn,dtype=int))],
                ignore_index=True)
rf_gan=RandomForestClassifier(n_estimators=400,class_weight="balanced_subsample",
                              n_jobs=-1,random_state=PRIMARY_SEED)
rf_gan.fit(aug_X,aug_y)
p_gan=rf_gan.predict_proba(X_test)[:,1]
r_gan=metrics_row("RF_10k_GAN_augmented",y_test,p_gan)

pd.DataFrame([r_real,r_gan]).to_csv(f"{RESULT_DIR}/E2_GAN_results.csv",index=False)
print(r_real); print(r_gan)

# ---------------- CELL 11: E3 controlled robustness ----------------
# Refit primary RF on full deduplicated training data.
rf_primary=RandomForestClassifier(
    n_estimators=500,class_weight="balanced_subsample",
    n_jobs=-1,random_state=PRIMARY_SEED
)
rf_primary.fit(X_train,y_train)

train_min=X_train.min().values
train_max=X_train.max().values

def perturb(Xdf,level,seed=2026):
    rng=np.random.default_rng(seed)
    arr=Xdf.values.astype(np.float64)
    noise=rng.normal(0,level,size=arr.shape)
    out=arr*(1+noise)
    out=np.clip(out,train_min,train_max)
    return pd.DataFrame(out,columns=FEATURES,index=Xdf.index)

rob_rows=[]
for level in [0,.01,.05,.10,.20]:
    Xt=X_test if level==0 else perturb(X_test,level,2026)
    p=rf_primary.predict_proba(Xt)[:,1]
    row=metrics_row(f"RF_perturb_{level:.2f}",y_test,p)
    row["perturbation"]=level
    rob_rows.append(row)

rob_df=pd.DataFrame(rob_rows)
rob_df.to_csv(f"{RESULT_DIR}/E3_robustness.csv",index=False)

plt.figure(figsize=(7,4))
plt.plot(rob_df["perturbation"]*100,rob_df["f1"]*100,marker="o")
plt.xlabel("Perturbation level (%)"); plt.ylabel("F1 (%)")
plt.title("Baseline Robustness Under Controlled Perturbation")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig5_robustness.png",dpi=300); plt.show()

# ---------------- CELL 12: E4 perturbation-aware training ----------------
# Augment training data with 5% perturbed copies.
X_train_p=perturb(X_train,.05,2026)
X_rob_train=pd.concat([X_train,X_train_p],ignore_index=True)
y_rob_train=pd.concat([y_train.reset_index(drop=True),y_train.reset_index(drop=True)],ignore_index=True)

rf_rob=RandomForestClassifier(
    n_estimators=500,class_weight="balanced_subsample",
    n_jobs=-1,random_state=PRIMARY_SEED
)
rf_rob.fit(X_rob_train,y_rob_train)

robtrain_rows=[]
for level in [0,.01,.05,.10,.20]:
    Xt=X_test if level==0 else perturb(X_test,level,2027)
    p=rf_rob.predict_proba(Xt)[:,1]
    row=metrics_row(f"RobustRF_perturb_{level:.2f}",y_test,p)
    row["perturbation"]=level
    robtrain_rows.append(row)

robtrain_df=pd.DataFrame(robtrain_rows)
robtrain_df.to_csv(f"{RESULT_DIR}/E4_robust_training.csv",index=False)

plt.figure(figsize=(7,4))
plt.plot(rob_df["perturbation"]*100,rob_df["f1"]*100,marker="o",label="Baseline RF")
plt.plot(robtrain_df["perturbation"]*100,robtrain_df["f1"]*100,marker="s",label="Perturbation-aware RF")
plt.xlabel("Perturbation level (%)"); plt.ylabel("F1 (%)")
plt.title("Robustness Before and After Perturbation-Aware Training")
plt.legend(); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_robust_training.png",dpi=300); plt.show()

# ---------------- CELL 13: E5 SHAP global + stability ----------------
import shap

# Explain a fixed 200-sample subset for reproducibility.
N_EXPLAIN=min(200,len(X_test))
ex_idx=np.random.default_rng(PRIMARY_SEED).choice(len(X_test),N_EXPLAIN,replace=False)
X_explain=X_test.iloc[ex_idx].copy()

explainer=shap.TreeExplainer(rf_primary)
sv=explainer.shap_values(X_explain)

# SHAP version compatibility: binary output may be list or array.
if isinstance(sv,list):
    sv_abs=np.abs(sv[1])
else:
    arr=np.asarray(sv)
    if arr.ndim==3:
        sv_abs=np.abs(arr[:,:,1])
    else:
        sv_abs=np.abs(arr)

mean_abs=sv_abs.mean(axis=0)
shap_global=pd.DataFrame({"feature":FEATURES,"mean_abs_shap":mean_abs})
shap_global=shap_global.sort_values("mean_abs_shap",ascending=False)
shap_global.to_csv(f"{RESULT_DIR}/E5_SHAP_global.csv",index=False)

plt.figure(figsize=(7,4))
plt.barh(shap_global["feature"],shap_global["mean_abs_shap"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP value|"); plt.title("Global SHAP Feature Importance")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig7_shap_global.png",dpi=300); plt.show()

# Explanation stability under 1% perturbation.
X_pert=perturb(X_explain,.01,2028)
sv2=explainer.shap_values(X_pert)
if isinstance(sv2,list):
    sv2_abs=np.abs(sv2[1])
else:
    arr=np.asarray(sv2)
    sv2_abs=np.abs(arr[:,:,1]) if arr.ndim==3 else np.abs(arr)

spears=[]; top1=[]; top3=[]
for i in range(N_EXPLAIN):
    a=sv_abs[i]; b=sv2_abs[i]
    rho=spearmanr(a,b).statistic
    if np.isnan(rho): rho=1.0
    spears.append(rho)
    order_a=np.argsort(-a)
    order_b=np.argsort(-b)
    top1.append(int(order_a[0]==order_b[0]))
    top3.append(len(set(order_a[:3]).intersection(set(order_b[:3])))/3)

shap_stability={
    "samples":N_EXPLAIN,
    "mean_spearman":float(np.mean(spears)),
    "median_spearman":float(np.median(spears)),
    "top1_agreement":float(np.mean(top1)),
    "top3_overlap":float(np.mean(top3)),
    "perturbation":0.01
}
with open(f"{RESULT_DIR}/E5_SHAP_stability.json","w") as f:
    json.dump(shap_stability,f,indent=2)
print(shap_stability)

# ---------------- CELL 14: E6-E9 multi-seed validation ----------------
# Primary model repeated over 5 seeds.
seed_rows=[]
for seed in SEEDS:
    set_seed(seed)
    Xtr,Xte,ytr,yte=train_test_split(
        X,y,test_size=.20,stratify=y,random_state=seed
    )
    rf_seed=RandomForestClassifier(
        n_estimators=400,class_weight="balanced_subsample",
        n_jobs=-1,random_state=seed
    )
    t=time.time()
    rf_seed.fit(Xtr,ytr)
    p=rf_seed.predict_proba(Xte)[:,1]
    row=metrics_row("Random Forest",yte,p)
    row["seed"]=seed
    row["train_seconds"]=time.time()-t
    seed_rows.append(row)

seed_df=pd.DataFrame(seed_rows)
seed_df.to_csv(f"{RESULT_DIR}/E6_multiseed_RF.csv",index=False)

summary={}
for metric in ["accuracy","precision","recall","f1","roc_auc","pr_auc"]:
    summary[metric+"_mean"]=float(seed_df[metric].mean())
    summary[metric+"_std"]=float(seed_df[metric].std(ddof=1))
with open(f"{RESULT_DIR}/E6_multiseed_summary.json","w") as f:
    json.dump(summary,f,indent=2)

# ---------------- CELL 15: E7 ablation summary ----------------
# Compact ablation table using already executed experiments.
ablation=[
    ["RF baseline","Clean detection",float(metrics_row("RF",y_test,rf_prob)["f1"])],
    ["RF + DNN stacking","Hybridization",float(hyb_row["f1"])],
    ["RF 10k real-only","Limited-data baseline",float(r_real["f1"])],
    ["RF 10k + GAN","Generative augmentation",float(r_gan["f1"])],
    ["RF + perturbation-aware training","Robustness adaptation",
     float(robtrain_df.loc[robtrain_df.perturbation==.05,"f1"].iloc[0])]
]
abdf=pd.DataFrame(ablation,columns=["configuration","purpose","f1"])
abdf.to_csv(f"{RESULT_DIR}/E7_ablation.csv",index=False)

# ---------------- CELL 16: E8 statistical/robustness deltas ----------------
clean_f1=float(rob_df.loc[rob_df.perturbation==0,"f1"].iloc[0])
delta_rows=[]
for _,r in rob_df.iterrows():
    if r["perturbation"]==0: continue
    rr=float(robtrain_df.loc[robtrain_df.perturbation==r["perturbation"],"f1"].iloc[0])
    delta_rows.append({
        "perturbation":float(r["perturbation"]),
        "baseline_f1":float(r["f1"]),
        "robust_f1":rr,
        "absolute_recovery":rr-float(r["f1"]),
        "baseline_degradation_from_clean":clean_f1-float(r["f1"]),
        "robust_degradation_from_clean":float(robtrain_df.loc[
            robtrain_df.perturbation==r["perturbation"],"f1"].iloc[0])-clean_f1
    })
delta_df=pd.DataFrame(delta_rows)
delta_df.to_csv(f"{RESULT_DIR}/E8_robustness_deltas.csv",index=False)

# ---------------- CELL 17: E9 final machine-readable results ----------------
final_results={
    "dataset_audit":audit,
    "primary_split":{"test_size":.20,"random_state":PRIMARY_SEED,"duplicate_removal":True},
    "E1_baselines":baseline_df.to_dict(orient="records"),
    "E1_DNN":dnn_row,
    "E1_hybrid":hyb_row,
    "E2_GAN":pd.DataFrame([r_real,r_gan]).to_dict(orient="records"),
    "E3_robustness":rob_df.to_dict(orient="records"),
    "E4_robust_training":robtrain_df.to_dict(orient="records"),
    "E5_SHAP_global":shap_global.to_dict(orient="records"),
    "E5_SHAP_stability":shap_stability,
    "E6_multiseed_summary":summary,
    "E7_ablation":ablation,
    "E8_robustness_deltas":delta_rows
}
with open(f"{RESULT_DIR}/final_results.json","w") as f:
    json.dump(final_results,f,indent=2)

# ---------------- CELL 18: Package outputs ----------------
import shutil
zip_path="/content/SCADA_XAI_IDS_results.zip"
shutil.make_archive("/content/SCADA_XAI_IDS_results","zip",RESULT_DIR)
print("DONE.")
print("Download:",zip_path)

# Optional:
# from google.colab import files
# files.download(zip_path)
