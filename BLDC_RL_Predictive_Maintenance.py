# =============================================================================
# AI-BASED PREDICTIVE MAINTENANCE – BLDC MOTOR
# 4 PURE ALGORITHMS: ANN | CNN (Deep MLP) | Random Forest | DQN-RL (Pure RL)
# NO HYBRID – Each algorithm stands independently
# WHITE THEME: white background, black text, white scale box outline
# =============================================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("plots", exist_ok=True)

# ── Print BLDC physics equations ──────────────────────────────────────────────
print("\n" + "="*72)
print("        BLDC MOTOR FAULT DETECTION – KEY PHYSICS FORMULAE")
print("="*72)
print("  Back-EMF            :  E   = Ke × ω              Ke = 0.015 V·s/rad")
print("  Angular Speed       :  ω   = 2π × N / 60         (rad/s)")
print("  Torque              :  T   = Kt × Ia             Kt ≈ Ke (SI)")
print("  Electrical Power    :  Pin = V × I               (Watts)")
print("  Mechanical Power    :  Pm  = T × ω               (Watts)")
print("  Efficiency          :  η   = (Pm / Pin) × 100    (%)")
print("  Back-EMF Ratio      :  BER = E / V               (< 0.85 → winding)")
print("  Power Balance       :  PB  = Pm / Pin            (< 0.55 → bearing)")
print("  Commutation Time    :  tc  = 60 / (6 × N × P)")
print()
print("  FAULT THRESHOLDS")
print("  Single Winding: Temp > 53°C  OR  Current > 6.8A  OR  BER < 0.85")
print("  Double Winding: Temp > 70°C  OR  Current > 8.8A  OR  BER < 0.70")
print("  Bearing Fault : Vibration > 0.70g  OR  η < 55%  OR  PB < 0.55")
print("="*72 + "\n")

# =============================================================================
# SECTION 1 – DATASET GENERATION  (1 200 samples, 4 classes)
# =============================================================================

def generate_dataset():
    dist = {
        "Normal":               400,
        "Single_Winding_Fault": 280,
        "Double_Winding_Fault": 260,
        "Bearing_Fault":        260,
    }
    rng = lambda mu, sig, lo, hi, k: np.clip(
        np.random.normal(mu, sig, k), lo, hi)

    records = []
    for label, count in dist.items():
        if label == "Normal":
            V  = rng(48.0, 0.8,  46.0, 50.0, count)
            I  = rng( 5.0, 0.5,   4.0,  6.5, count)
            T  = rng(45.0, 3.0,  38.0, 52.0, count)
            N  = rng(3000, 50,  2900, 3100,   count)
            Vb = rng( 0.20, 0.05, 0.10, 0.32, count)
            Tq = rng( 2.0,  0.2,  1.5,  2.5,  count)
        elif label == "Single_Winding_Fault":
            V  = rng(46.0, 1.2,  43.0, 49.0, count)
            I  = rng( 7.5, 0.8,   6.0,  9.5, count)
            T  = rng(58.0, 4.0,  50.0, 68.0, count)
            N  = rng(2800, 80,  2600, 2950,   count)
            Vb = rng( 0.45, 0.10, 0.28, 0.62, count)
            Tq = rng( 2.5,  0.3,  1.9,  3.1,  count)
        elif label == "Double_Winding_Fault":
            V  = rng(43.0, 1.5,  39.0, 46.0, count)
            I  = rng(10.5, 1.0,   8.5, 13.0, count)
            T  = rng(75.0, 6.0,  63.0, 88.0, count)
            N  = rng(2500, 120, 2200, 2750,   count)
            Vb = rng( 0.80, 0.15, 0.52, 1.10, count)
            Tq = rng( 3.2,  0.4,  2.4,  4.0,  count)
        else:  # Bearing_Fault
            V  = rng(47.0, 1.0,  44.5, 49.5, count)
            I  = rng( 6.0, 0.7,   4.8,  7.8, count)
            T  = rng(68.0, 5.0,  58.0, 79.0, count)
            N  = rng(2700, 200, 2250, 3000,   count)
            Vb = rng( 1.20, 0.25, 0.75, 1.60, count)
            Tq = rng( 2.3,  0.5,  1.4,  3.2,  count)

        omega      = 2 * np.pi * N / 60
        back_emf   = 0.015 * omega
        power_in   = V * I
        power_mech = Tq * omega
        efficiency = np.clip(power_mech / power_in * 100, 0, 100)
        ber        = back_emf / V
        pb         = power_mech / power_in

        for i in range(count):
            records.append({
                "Voltage_V":      round(float(V[i]),       3),
                "Current_A":      round(float(I[i]),       3),
                "Temperature_C":  round(float(T[i]),       2),
                "Speed_RPM":      round(float(N[i]),       1),
                "Vibration_g":    round(float(Vb[i]),      4),
                "Torque_Nm":      round(float(Tq[i]),      3),
                "BackEMF_V":      round(float(back_emf[i]),4),
                "Power_In_W":     round(float(power_in[i]),3),
                "Power_Mech_W":   round(float(power_mech[i]),3),
                "Efficiency_pct": round(float(efficiency[i]),3),
                "BER":            round(float(ber[i]),     5),
                "PowerBalance":   round(float(pb[i]),      5),
                "Fault_Label":    label,
            })

    df = (pd.DataFrame(records)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True))
    df.to_excel("BLDC_Dataset_1200.xlsx", sheet_name="BLDC_Data", index=False)
    print(f"[DATA] Generated {len(df)} samples → BLDC_Dataset_1200.xlsx")
    print(df["Fault_Label"].value_counts().to_string())
    return df


df = generate_dataset()

# =============================================================================
# SECTION 2 – FEATURES & PREPROCESSING
# =============================================================================

FEATURES = ["Voltage_V", "Current_A", "Temperature_C", "Speed_RPM",
            "Vibration_g", "Torque_Nm", "BackEMF_V", "Power_In_W",
            "Power_Mech_W", "Efficiency_pct", "BER", "PowerBalance"]
FEATURES = [f for f in FEATURES if f in df.columns]

X     = df[FEATURES].values
y     = df["Fault_Label"].values
le    = LabelEncoder()
y_enc = le.fit_transform(y)
CLASS_NAMES = list(le.classes_)
N_CLASSES   = len(CLASS_NAMES)
print(f"\n[PREP] Classes ({N_CLASSES}): {CLASS_NAMES}")

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

def balance_classes(X, y):
    classes, counts = np.unique(y, return_counts=True)
    target_n = counts.max()
    Xl, yl = [X], [y]
    for cls, cnt in zip(classes, counts):
        deficit = target_n - cnt
        if deficit > 0:
            idx    = np.where(y == cls)[0]
            chosen = np.random.choice(idx, deficit, replace=True)
            noise  = np.random.normal(0, 0.05, (deficit, X.shape[1]))
            Xl.append(X[chosen] + noise)
            yl.append(np.full(deficit, cls))
    Xb = np.vstack(Xl);  yb = np.concatenate(yl)
    p  = np.random.permutation(len(yb))
    return Xb[p], yb[p]

X_tr_bal, y_tr_bal = balance_classes(X_tr, y_tr)
print(f"[PREP] Balanced training set: {len(X_tr_bal)} samples\n")

# =============================================================================
# SECTION 3 – ANN
# =============================================================================
print("[ANN] Training …")
ann = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation="relu", max_iter=800, random_state=7,
    learning_rate_init=0.001, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20)
ann.fit(X_tr_bal, y_tr_bal)
ann_pred_te = ann.predict(X_te)
ann_acc     = accuracy_score(y_te, ann_pred_te)
print(f"  ANN  Accuracy : {ann_acc*100:.2f}%")

# =============================================================================
# SECTION 4 – CNN  (Deep MLP)
# =============================================================================
print("\n[CNN] Training …")
cnn = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64, 32),
    activation="relu", max_iter=800, random_state=42,
    learning_rate_init=0.0005, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20)
cnn.fit(X_tr_bal, y_tr_bal)
cnn_pred_te = cnn.predict(X_te)
cnn_acc     = accuracy_score(y_te, cnn_pred_te)
print(f"  CNN  Accuracy : {cnn_acc*100:.2f}%")

# =============================================================================
# SECTION 5 – RANDOM FOREST
# =============================================================================
print("\n[RF]  Training …")
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=1,
    random_state=42, class_weight="balanced", n_jobs=-1)
rf.fit(X_tr_bal, y_tr_bal)
rf_pred_te = rf.predict(X_te)
rf_acc     = accuracy_score(y_te, rf_pred_te)
print(f"  RF   Accuracy : {rf_acc*100:.2f}%")

# =============================================================================
# SECTION 6 – PURE DQN-RL
# =============================================================================

N_FEAT = X_sc.shape[1]
H1, H2 = 256, 128

class DQN:
    def __init__(self, n_in, h1, h2, n_out, lr=0.001):
        self.lr = lr
        self.W1 = np.random.randn(n_in, h1) * np.sqrt(2 / n_in)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2)   * np.sqrt(2 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, n_out) * np.sqrt(2 / h2)
        self.b3 = np.zeros(n_out)

    def _relu(self, x):   return np.maximum(0, x)
    def _relu_d(self, x): return (x > 0).astype(float)

    def forward(self, x):
        self.x  = x
        self.z1 = x       @ self.W1 + self.b1;  self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2;  self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3

    def backward(self, dL_dq):
        dW3 = self.a2.T @ dL_dq;                 db3 = dL_dq.sum(0)
        da2 = (dL_dq @ self.W3.T) * self._relu_d(self.z2)
        dW2 = self.a1.T @ da2;                   db2 = da2.sum(0)
        da1 = (da2    @ self.W2.T) * self._relu_d(self.z1)
        dW1 = self.x.T  @ da1;                   db1 = da1.sum(0)
        B   = max(len(dL_dq), 1)
        for dW, W, db, b in [(dW1,self.W1,db1,self.b1),
                              (dW2,self.W2,db2,self.b2),
                              (dW3,self.W3,db3,self.b3)]:
            np.clip(dW, -1, 1, out=dW);  np.clip(db, -1, 1, out=db)
            W -= self.lr * dW / B
            b -= self.lr * db / B

    def copy_from(self, other):
        for a in ["W1","b1","W2","b2","W3","b3"]:
            setattr(self, a, getattr(other, a).copy())


class PrioritizedReplay:
    def __init__(self, capacity=20000, alpha=0.6):
        self.cap   = capacity
        self.alpha = alpha
        self.buf   = []
        self.prios = np.zeros(capacity, dtype=np.float32)
        self.pos   = 0

    def push(self, transition, error=1.0):
        p = (abs(error) + 1e-5) ** self.alpha
        if len(self.buf) < self.cap:
            self.buf.append(transition)
        else:
            self.buf[self.pos] = transition
        self.prios[self.pos] = p
        self.pos = (self.pos + 1) % self.cap

    def sample(self, n, beta=0.4):
        total = len(self.buf)
        p     = self.prios[:total];  p = p / p.sum()
        idx   = np.random.choice(total, n, replace=False, p=p)
        w     = (total * p[idx]) ** (-beta);  w /= w.max()
        return [self.buf[i] for i in idx], idx, w

    def update_prios(self, idx, errors):
        for i, e in zip(idx, errors):
            self.prios[i] = (abs(e) + 1e-5) ** self.alpha

    def __len__(self): return len(self.buf)


SEVERITY   = {0: 2.0, 1: 1.5, 2: 3.0, 3: 2.5}
NORMAL_IDX = CLASS_NAMES.index("Normal") if "Normal" in CLASS_NAMES else 0

def shaped_reward(action, true_label):
    a, t = int(action), int(true_label)
    if a == t:
        return 10.0 + SEVERITY.get(t, 1.0) * 5.0
    elif t != NORMAL_IDX and a == NORMAL_IDX:
        return -25.0 * SEVERITY.get(t, 1.0)
    elif t == NORMAL_IDX and a != NORMAL_IDX:
        return -5.0
    else:
        return -10.0


EPSILON       = 1.0
EPSILON_MIN   = 0.03
EPSILON_DECAY = 0.9965
GAMMA         = 0.97
BATCH_SIZE    = 64
TARGET_UPDATE = 50
EPISODES      = 1500
TRAIN_EVERY   = 5
EP_SAMPLES    = 200

online_net = DQN(N_FEAT, H1, H2, N_CLASSES, lr=0.001)
target_net = DQN(N_FEAT, H1, H2, N_CLASSES, lr=0.001)
target_net.copy_from(online_net)
replay     = PrioritizedReplay(capacity=20000)

rl_rewards     = []
rl_punishments = []
rl_epsilons    = []

print("\n[DQN-RL] Training – Pure RL (Prioritized Replay + Double DQN) …")
step      = 0
train_idx = np.arange(len(X_tr_bal))

for ep in range(EPISODES):
    ep_correct = ep_wrong = 0
    ep_idx = np.random.choice(train_idx,
                               min(EP_SAMPLES, len(train_idx)),
                               replace=False)

    for i in ep_idx:
        state      = X_tr_bal[i]
        true_label = y_tr_bal[i]

        if np.random.rand() < EPSILON:
            action = np.random.randint(N_CLASSES)
        else:
            q      = online_net.forward(state.reshape(1, -1))[0]
            action = int(np.argmax(q))

        reward = shaped_reward(action, true_label)

        ni         = np.random.randint(len(X_tr_bal))
        next_state = X_tr_bal[ni]

        q_cur  = online_net.forward(state.reshape(1,-1))[0][action]
        q_nxt  = target_net.forward(next_state.reshape(1,-1))[0].max()
        td_err = abs(reward + GAMMA * q_nxt - q_cur)
        replay.push((state, action, reward, next_state), td_err)

        if action == true_label: ep_correct += 1
        else:                    ep_wrong   += 1
        step += 1

        if step % TRAIN_EVERY == 0 and len(replay) >= BATCH_SIZE:
            batch, b_idx, weights = replay.sample(BATCH_SIZE)
            S  = np.array([t[0] for t in batch])
            A  = np.array([t[1] for t in batch], dtype=int)
            R  = np.array([t[2] for t in batch])
            NS = np.array([t[3] for t in batch])

            best_a   = np.argmax(online_net.forward(NS), axis=1)
            q_tg_ns  = target_net.forward(NS)
            q_nxt_v  = q_tg_ns[np.arange(BATCH_SIZE), best_a]
            targets  = R + GAMMA * q_nxt_v

            q_pred   = online_net.forward(S)
            td_errs  = np.abs(targets - q_pred[np.arange(BATCH_SIZE), A])
            replay.update_prios(b_idx, td_errs)

            dL_dq    = np.zeros_like(q_pred)
            dL_dq[np.arange(BATCH_SIZE), A] = (
                q_pred[np.arange(BATCH_SIZE), A] - targets) * weights
            online_net.backward(dL_dq)

    if (ep + 1) % TARGET_UPDATE == 0:
        target_net.copy_from(online_net)

    total_ep = ep_correct + ep_wrong
    rl_rewards.append(ep_correct / total_ep if total_ep > 0 else 0.0)
    rl_punishments.append(ep_wrong / total_ep if total_ep > 0 else 0.0)
    rl_epsilons.append(EPSILON)
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    if (ep + 1) % 150 == 0:
        ep_acc = ep_correct / total_ep if total_ep > 0 else 0
        print(f"  Ep {ep+1:>4}/{EPISODES}  acc={ep_acc*100:.1f}%  "
              f"ε={EPSILON:.4f}  buf={len(replay)}")


def rl_predict(X_data):
    preds = []
    for row in X_data:
        q = online_net.forward(row.reshape(1, -1))[0]
        preds.append(int(np.argmax(q)))
    return np.array(preds)

rl_pred_te = rl_predict(X_te)
rl_acc     = accuracy_score(y_te, rl_pred_te)
print(f"\n  DQN-RL Accuracy : {rl_acc*100:.2f}%")

# =============================================================================
# SECTION 7 – FULL DATASET PREDICTIONS
# =============================================================================
X_all = scaler.transform(df[FEATURES].values)
df["ANN_Pred"] = le.inverse_transform(ann.predict(X_all))
df["CNN_Pred"] = le.inverse_transform(cnn.predict(X_all))
df["RF_Pred"]  = le.inverse_transform(rf.predict(X_all))
df["RL_Pred"]  = le.inverse_transform(rl_predict(X_all))
df.to_excel("BLDC_Results.xlsx", index=False)
print("\n[EXCEL] Results → BLDC_Results.xlsx")

for name, preds in [("ANN",ann_pred_te),("CNN",cnn_pred_te),
                    ("Random Forest",rf_pred_te),("DQN-RL",rl_pred_te)]:
    print(f"\n===== CLASSIFICATION REPORT : {name} =====")
    print(classification_report(y_te, preds, target_names=CLASS_NAMES))

# =============================================================================
# SECTION 8 – PLOT HELPERS  (WHITE THEME)
# =============================================================================

# ── WHITE THEME COLOURS ───────────────────────────────────────────────────────
DARK_BG  = "#FFFFFF"   # figure background  → white
AX_BG    = "#FFFFFF"   # axes background    → white
GRID_CLR = "#CCCCCC"   # gridlines          → light grey
TEXT_CLR = "#000000"   # all text           → black
BOX_EDGE = "#FFFFFF"   # scale box outline  → white
BOX_FACE = "#F5F5F5"   # scale box fill     → very light grey (readable on white)

FAULT_COLORS = {
    "Normal":               "#2ECC71",
    "Single_Winding_Fault": "#E9A03A",
    "Double_Winding_Fault": "#E74C3C",
    "Bearing_Fault":        "#3498DB",
}
MODEL_COLORS = {
    "ANN":           "#9B59B6",
    "CNN":           "#3498DB",
    "Random Forest": "#E67E22",
    "DQN-RL":        "#1ABC9C",
}

PARAM_LABELS = ["Voltage\n(V)", "Current\n(A)", "Temp\n(°C)",
                "Speed\n(RPM)", "Vibration\n(g)", "Torque\n(Nm)"]
UNITS        = ["V", "A", "°C", "RPM", "g", "Nm"]
PLOT_FEATS   = ["Voltage_V","Current_A","Temperature_C",
                "Speed_RPM","Vibration_g","Torque_Nm"]
MAX_SCALES   = np.array([55.0, 14.0, 100.0, 3200.0, 1.70, 4.50])

def pct(arr):
    return np.array(arr) / MAX_SCALES * 100.0

def mean_vals_for(pred_col, label):
    sub = df[df[pred_col] == label]
    return sub[PLOT_FEATS].mean().values if len(sub) > 0 else np.zeros(len(PLOT_FEATS))

def apply_dark(fig, ax, title):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=15, fontweight="extra bold", pad=12)
    ax.tick_params(colors=TEXT_CLR, labelsize=11, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.yaxis.grid(color=GRID_CLR, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)
        sp.set_linewidth(1.5)

def annotate_bar(ax, bar, raw, unit):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
            f"{raw:.1f}{unit}", ha="center", va="bottom",
            color=TEXT_CLR, fontsize=9, fontweight="extra bold")

def save_fig(fig, path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[PLOT] → {path}")

def info_box(ax, text):
    ax.text(0.995, 0.995, text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=TEXT_CLR, fontfamily="monospace",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=BOX_FACE,
                      edgecolor=BOX_EDGE,   # ← white outline
                      alpha=0.95))

def smooth(arr, w=30):
    out = np.convolve(arr, np.ones(w)/w, mode="same")
    out[:w//2]  = arr[:w//2]
    out[-w//2:] = arr[-w//2:]
    return out

# =============================================================================
# SECTION 9 – GRAPHS 01-04  Normal Condition
# =============================================================================

MODEL_CFG = [
    ("ANN",           "ANN_Pred"),
    ("CNN",           "CNN_Pred"),
    ("Random Forest", "RF_Pred"),
    ("DQN-RL",        "RL_Pred"),
]

for g_num, (model_name, pred_col) in enumerate(MODEL_CFG, start=1):
    vals = mean_vals_for(pred_col, "Normal")
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_dark(fig, ax, f"Graph {g_num:02d} – Normal Condition  [{model_name}]")
    x    = np.arange(len(PARAM_LABELS))
    bars = ax.bar(x, pct(vals), color=FAULT_COLORS["Normal"],
                  alpha=0.85, width=0.55, zorder=3)
    for bar, raw, unit in zip(bars, vals, UNITS):
        annotate_bar(ax, bar, raw, unit)
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_LABELS, color=TEXT_CLR, fontsize=11, fontweight="bold")
    ax.set_ylabel("% of per-parameter max scale", color=TEXT_CLR, fontsize=12, fontweight="bold")
    ax.set_xlabel("Sensor Parameters",            color=TEXT_CLR, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.legend(handles=[mpatches.Patch(color=FAULT_COLORS["Normal"], alpha=0.85,
                                      label="Normal Condition")],
              facecolor=AX_BG, labelcolor=TEXT_CLR, edgecolor=GRID_CLR, fontsize=11,
              prop={"weight": "bold", "size": 11})
    save_fig(fig, f"plots/Graph{g_num:02d}_Normal_{model_name.replace(' ','_')}.png")

# =============================================================================
# SECTION 10 – GRAPHS 05-16  Fault vs Normal
# =============================================================================

FAULTS = [
    ("Single_Winding_Fault",  5),
    ("Double_Winding_Fault",  9),
    ("Bearing_Fault",        13),
]

for fault_name, base_g in FAULTS:
    for offset, (model_name, pred_col) in enumerate(MODEL_CFG):
        g_num = base_g + offset
        nv = mean_vals_for(pred_col, "Normal")
        fv = mean_vals_for(pred_col, fault_name)
        x  = np.arange(len(PARAM_LABELS));  w = 0.38
        fig, ax = plt.subplots(figsize=(13, 6))
        apply_dark(fig, ax,
            f"Graph {g_num:02d} – {fault_name.replace('_',' ')}  [{model_name}]")
        bn = ax.bar(x - w/2, pct(nv), w, color=FAULT_COLORS["Normal"],
                    alpha=0.80, label="Normal", zorder=3)
        bf = ax.bar(x + w/2, pct(fv), w, color=FAULT_COLORS[fault_name],
                    alpha=0.85, label=fault_name.replace("_"," "), zorder=3)
        for bar, raw, unit in zip(bn, nv, UNITS): annotate_bar(ax, bar, raw, unit)
        for bar, raw, unit in zip(bf, fv, UNITS): annotate_bar(ax, bar, raw, unit)
        ax.set_xticks(x)
        ax.set_xticklabels(PARAM_LABELS, color=TEXT_CLR, fontsize=11, fontweight="bold")
        ax.set_ylabel("% of per-parameter max scale", color=TEXT_CLR, fontsize=12, fontweight="bold")
        ax.set_xlabel("Sensor Parameters",            color=TEXT_CLR, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 125)
        ax.legend(facecolor=AX_BG, labelcolor=TEXT_CLR,
                  edgecolor=GRID_CLR, fontsize=11,
                  prop={"weight": "bold", "size": 11})
        save_fig(fig,
            f"plots/Graph{g_num:02d}_{fault_name}_{model_name.replace(' ','_')}.png")

# =============================================================================
# SECTION 11 – GRAPH 17  Accuracy Comparison
# =============================================================================

all_accs    = [ann_acc*100, cnn_acc*100, rf_acc*100, rl_acc*100]
model_names = ["ANN", "CNN", "Random\nForest", "DQN-RL"]
bar_colors  = [MODEL_COLORS["ANN"], MODEL_COLORS["CNN"],
               MODEL_COLORS["Random Forest"], MODEL_COLORS["DQN-RL"]]

fig, ax = plt.subplots(figsize=(11, 6))
apply_dark(fig, ax,
    "Graph 17 – Accuracy Comparison : ANN | CNN | Random Forest | DQN-RL")
x17 = np.arange(len(model_names))
b17 = ax.bar(x17, all_accs, color=bar_colors, alpha=0.88, width=0.5, zorder=3)
for bar, acc in zip(b17, all_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{acc:.2f}%", ha="center", va="bottom",
            color=TEXT_CLR, fontsize=14, fontweight="extra bold")
ax.set_xticks(x17)
ax.set_xticklabels(model_names, color=TEXT_CLR, fontsize=13, fontweight="bold")
ax.set_ylabel("Accuracy (%)", color=TEXT_CLR, fontsize=13, fontweight="bold")
ax.set_xlabel("Algorithm",    color=TEXT_CLR, fontsize=13, fontweight="bold")
ax.set_ylim(0, 115)
save_fig(fig, "plots/Graph17_AccuracyComparison.png")

# =============================================================================
# SECTION 12 – GRAPH 18  Confusion Matrix
# =============================================================================

accs_dict = {"ANN":ann_acc, "CNN":cnn_acc, "RF":rf_acc, "DQN-RL":rl_acc}
best_name  = max(accs_dict, key=accs_dict.get)
preds_map  = {"ANN":ann_pred_te,"CNN":cnn_pred_te,
              "RF":rf_pred_te,"DQN-RL":rl_pred_te}
cm = confusion_matrix(y_te, preds_map[best_name])

fig, ax = plt.subplots(figsize=(8, 7))
apply_dark(fig, ax,
    f"Graph 18 – Confusion Matrix  [{best_name}  |  Acc={accs_dict[best_name]*100:.2f}%]")
im = ax.imshow(cm, cmap="Blues")
cb = plt.colorbar(im, ax=ax);  cb.ax.tick_params(colors=TEXT_CLR, labelsize=11)
for t in cb.ax.get_yticklabels():
    t.set_fontweight("bold")
short = [c.replace("_","\n") for c in CLASS_NAMES]
ax.set_xticks(range(N_CLASSES));  ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(short, color=TEXT_CLR, fontsize=10, fontweight="bold")
ax.set_yticklabels(short, color=TEXT_CLR, fontsize=10, fontweight="bold")
ax.set_xlabel("Predicted Label", color=TEXT_CLR, fontsize=13, fontweight="bold")
ax.set_ylabel("True Label",      color=TEXT_CLR, fontsize=13, fontweight="bold")
thresh = cm.max() / 2.0
for r in range(N_CLASSES):
    for c in range(N_CLASSES):
        ax.text(c, r, str(cm[r,c]), ha="center", va="center",
                fontsize=14, fontweight="extra bold",
                color="white" if cm[r,c] > thresh else "black")
save_fig(fig, f"plots/Graph18_ConfusionMatrix_{best_name}.png")

# =============================================================================
# SECTION 13 – GRAPH 19  RL Reward & Punishment
# =============================================================================

eps_arr    = np.arange(1, EPISODES + 1)
rew_smooth = smooth(rl_rewards)
pun_smooth = smooth(rl_punishments)
bar_w      = 0.42

fig, ax = plt.subplots(figsize=(14, 6))
apply_dark(fig, ax,
    "Graph 19 – DQN-RL Training : Reward Rate & Punishment Rate  [Vertical Bars]")

rew_bars = ax.bar(eps_arr - bar_w/2, rew_smooth, bar_w,
                  color="#2ECC71", alpha=0.82,
                  label="Reward rate  (correct / total steps)", zorder=3)

pun_bars = ax.bar(eps_arr + bar_w/2, pun_smooth, bar_w,
                  color="#E74C3C", alpha=0.82,
                  label="Punishment rate  (wrong / total steps)", zorder=3)

ax.axhline(0.50, color="#E67E22", linewidth=1.2,
           linestyle="--", alpha=0.85, label="50% threshold", zorder=4)

final_rew = float(rew_smooth[-1]);  final_pun = float(pun_smooth[-1])
ax.annotate(f"Final\n{final_rew*100:.1f}%",
            xy=(EPISODES, final_rew), xytext=(EPISODES - 180, final_rew + 0.09),
            color="#27AE60", fontsize=10, fontweight="extra bold",
            arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2))
ax.annotate(f"Final\n{final_pun*100:.1f}%",
            xy=(EPISODES, final_pun), xytext=(EPISODES - 220, final_pun + 0.11),
            color="#C0392B", fontsize=10, fontweight="extra bold",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2))

ax.set_xlabel("Episode",                            color=TEXT_CLR, fontsize=13, fontweight="bold")
ax.set_ylabel("Rate (fraction of steps / episode)", color=TEXT_CLR, fontsize=13, fontweight="bold")
ax.set_xlim(0, EPISODES + 15)
ax.set_ylim(-0.02, 1.12)
ax.legend(facecolor=AX_BG, labelcolor=TEXT_CLR,
          edgecolor=GRID_CLR, fontsize=11, loc="center left",
          prop={"weight": "bold", "size": 11})
info_box(ax,
    f"REWARD SHAPING\n"
    f"Correct:      +10 + sev×5\n"
    f"Missed fault: -25 × sev\n"
    f"False alarm:  -5\n"
    f"Wrong type:   -10\n"
    f"Batch={BATCH_SIZE}  γ={GAMMA}\n"
    f"Double DQN + PER")
save_fig(fig, "plots/Graph19_RL_Reward_Punishment.png")

# =============================================================================
# SECTION 14 – GRAPH 20  RL Epsilon Decay
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 6))
apply_dark(fig, ax,
    "Graph 20 – DQN-RL Training : Epsilon (ε) Decay over Episodes  [Vertical Bars]")

ax.bar(eps_arr, rl_epsilons,
       color="#F39C12", alpha=0.80, width=1.0,
       label="Epsilon (ε) per episode", zorder=3)

ax.axhline(EPSILON_MIN, color="#E74C3C", linewidth=1.5,
           linestyle="--", alpha=0.85,
           label=f"ε_min = {EPSILON_MIN}", zorder=4)

for m, ep_num in [(1,1),(300,300),(600,600),(900,900),(1200,1200),(1500,1500)]:
    if ep_num <= EPISODES:
        ep_val = rl_epsilons[ep_num - 1]
        ax.text(ep_num, ep_val + 0.03,
                f"Ep{ep_num}\nε={ep_val:.3f}",
                ha="center", va="bottom",
                color=TEXT_CLR, fontsize=8, fontweight="bold",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=BOX_FACE,
                          edgecolor=BOX_EDGE,  # ← white outline
                          alpha=0.90))

ax.set_xlabel("Episode",    color=TEXT_CLR, fontsize=13, fontweight="bold")
ax.set_ylabel("Epsilon (ε)", color=TEXT_CLR, fontsize=13, fontweight="bold")
ax.set_xlim(0, EPISODES + 15)
ax.set_ylim(-0.02, 1.12)
ax.legend(facecolor=AX_BG, labelcolor=TEXT_CLR,
          edgecolor=GRID_CLR, fontsize=11,
          prop={"weight": "bold", "size": 11})
info_box(ax,
    f"EPSILON SCHEDULE\n"
    f"ε_start  = 1.0\n"
    f"ε_min    = {EPSILON_MIN}\n"
    f"ε_decay  = {EPSILON_DECAY}\n"
    f"Episodes = {EPISODES}\n"
    f"Strategy : ε-greedy\n"
    f"  explore → exploit")
save_fig(fig, "plots/Graph20_RL_Epsilon_Decay.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*72)
print("  ✅  ALL 20 GRAPHS GENERATED  →  plots/")
print("="*72)
print(f"""
  NORMAL CONDITION (4 graphs)
    01 Graph01_Normal_ANN.png
    02 Graph02_Normal_CNN.png
    03 Graph03_Normal_Random_Forest.png
    04 Graph04_Normal_DQN-RL.png

  SINGLE WINDING FAULT vs NORMAL (4 graphs)
    05 Graph05_Single_Winding_Fault_ANN.png
    06 Graph06_Single_Winding_Fault_CNN.png
    07 Graph07_Single_Winding_Fault_Random_Forest.png
    08 Graph08_Single_Winding_Fault_DQN-RL.png

  DOUBLE WINDING FAULT vs NORMAL (4 graphs)
    09 Graph09_Double_Winding_Fault_ANN.png
    10 Graph10_Double_Winding_Fault_CNN.png
    11 Graph11_Double_Winding_Fault_Random_Forest.png
    12 Graph12_Double_Winding_Fault_DQN-RL.png

  BEARING FAULT vs NORMAL (4 graphs)
    13 Graph13_Bearing_Fault_ANN.png
    14 Graph14_Bearing_Fault_CNN.png
    15 Graph15_Bearing_Fault_Random_Forest.png
    16 Graph16_Bearing_Fault_DQN-RL.png

  SUMMARY GRAPHS (4 graphs)
    17 Graph17_AccuracyComparison.png
    18 Graph18_ConfusionMatrix_{best_name}.png
    19 Graph19_RL_Reward_Punishment.png
    20 Graph20_RL_Epsilon_Decay.png
""")
print("  MODEL ACCURACIES")
print(f"    ANN            : {ann_acc*100:.2f}%")
print(f"    CNN            : {cnn_acc*100:.2f}%")
print(f"    Random Forest  : {rf_acc*100:.2f}%")
print(f"    DQN-RL (Pure)  : {rl_acc*100:.2f}%")
print(f"    Best Model     : {best_name} ({accs_dict[best_name]*100:.2f}%)")
print("="*72 + "\n")