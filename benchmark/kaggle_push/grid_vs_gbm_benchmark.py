"""Institutional Grid vs. GBM — Kaggle CPU Benchmark Kernel.

Faithful self-contained port of benchmark/run_grid_vs_gbm.py +
data_engine/synthetic_mimic.py (tested locally; CSV round-trip is
collapsed in memory — the generator applies the same 2dp rounding and
the assembler hour-bucketing is replicated, so inputs are
byte-equivalent to the CSV path). The GBM arm embeds a parity gate on
the local AUROC (0.9116 @ seed 42, n=300): a mismatch >0.02 prints
PARITY_WARN on the first run.

Outputs:
  /kaggle/working/benchmark_report.json   — full joined report
  /kaggle/working/benchmark_summary.txt   — human-readable summary
"""
import json, os, time, random
import numpy as np

SEED = 42
K = 39
K3 = 117
WINDOW_SIZE = 14
ITEMIDS = {f: int(v) for f, v in {"HR": 220045, "O2Sat": 220277, "Temp": 223762, "SBP": 220179, "DBP": 220180, "Resp": 220210, "Lactate": 50813, "WBC": 51300, "Glucose": 50931, "Creatinine": 50912, "Potassium": 50971, "pH": 50820, "FiO2": 223835, "Platelets": 51265, "Hgb": 51221, "Hct": 51222}.items()}
BASELINES = {"HR": [72, 5], "O2Sat": [97, 1], "Temp": [36.8, 0.3], "SBP": [120, 10], "DBP": [80, 8], "Resp": [16, 2], "Lactate": [1.2, 0.3], "WBC": [7.0, 2.0], "Glucose": [90, 15], "Creatinine": [0.9, 0.2], "Potassium": [4.0, 0.4], "pH": [7.4, 0.03], "FiO2": [0.21, 0.05], "Platelets": [200, 40], "Hgb": [13.0, 1.5], "Hct": [39.0, 4.0]}
MISS_RATES = {"HR": 0.05, "O2Sat": 0.05, "Temp": 0.05, "SBP": 0.05, "DBP": 0.05, "Resp": 0.05, "Lactate": 0.5, "WBC": 0.6, "Glucose": 0.3, "Creatinine": 0.7, "Potassium": 0.5, "pH": 0.7, "FiO2": 0.1, "Platelets": 0.65, "Hgb": 0.55, "Hct": 0.55}
FEATURE_NAMES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
MIMIC_TO_FEATURE = {int(k): v for k, v in {"220045": "HR", "211": "HR", "220277": "O2Sat", "646": "O2Sat", "834": "SaO2", "223761": "Temp", "223762": "Temp", "678": "Temp", "676": "Temp", "677": "Temp", "220050": "SBP", "220179": "SBP", "51": "SBP", "442": "SBP", "455": "SBP", "220051": "DBP", "220180": "DBP", "8368": "DBP", "8441": "DBP", "52": "DBP", "8555": "DBP", "220052": "MAP", "220181": "MAP", "224": "MAP", "8440": "MAP", "456": "MAP", "220210": "Resp", "615": "Resp", "618": "Resp", "614": "Resp", "50813": "Lactate", "816": "Lactate", "1483": "Lactate", "1664": "Lactate", "3826": "Lactate", "3871": "Lactate", "3908": "Lactate", "7808": "Lactate", "16732": "Lactate", "84129": "Lactate", "51300": "WBC", "51301": "WBC", "861": "WBC", "1127": "WBC", "34482": "WBC", "50931": "Glucose", "807": "Glucose", "811": "Glucose", "1529": "Glucose", "225664": "Glucose", "226537": "Glucose", "220621": "Glucose", "50912": "Creatinine", "791": "Creatinine", "1162": "Creatinine", "5067": "Creatinine", "50971": "Potassium", "817": "Potassium", "1524": "Potassium", "4136": "Potassium", "84123": "Potassium", "50960": "Magnesium", "821": "Magnesium", "50902": "Chloride", "813": "Chloride", "84117": "Chloride", "51221": "Hgb", "849": "Hgb", "220228": "Hgb", "11682": "Hgb", "51222": "Hct", "851": "Hct", "220229": "Hct", "11633": "Hct", "51265": "Platelets", "850": "Platelets", "1536": "Platelets", "51006": "BUN", "845": "BUN", "50820": "pH", "50821": "pH", "780": "pH", "4753": "pH", "3683": "pH", "50818": "PaCO2", "50819": "PaCO2", "779": "PaCO2", "4751": "PaCO2", "223835": "FiO2", "3420": "FiO2", "3421": "FiO2", "3422": "FiO2", "189": "FiO2", "190": "FiO2", "1279": "FiO2", "1979": "FiO2", "7249": "FiO2", "7511": "FiO2", "7704": "FiO2", "16547": "FiO2", "50817": "SaO2", "50862": "SaO2", "220227": "SaO2", "50802": "BaseExcess", "50803": "BaseExcess", "778": "BaseExcess", "50806": "HCO3", "50807": "HCO3", "50882": "HCO3", "786": "HCO3", "4749": "HCO3", "50861": "AST", "50863": "Alkalinephos", "50808": "Calcium", "50893": "Calcium", "84112": "Calcium", "50885": "Bilirubin_direct", "50883": "Bilirubin_total", "51003": "TroponinI", "51274": "PTT", "51275": "PTT", "51214": "Fibrinogen", "50970": "Phosphate", "829": "Phosphate", "224689": "EtCO2", "19249": "EtCO2"}.items()}


def gen_stay_events(stay_id, subject_id, n_hours, scenario, seed, onset_hour=None):
    """One ICU stay's events (unix-ts charttimes, 2dp values). Mirrors
    data_engine/synthetic_mimic.generate_stay exactly."""
    random.seed(seed); np.random.seed(seed)
    if onset_hour is None and scenario == "deteriorating":
        onset_hour = random.randint(4, 20)
    elif onset_hour is None:
        onset_hour = 0

    baselines = {f: (m + np.random.normal(0, s * 1.5), s)
                  for f, (m, s) in BASELINES.items()}
    amp = random.uniform(0.7, 1.3) if scenario == "deteriorating" else 1.0
    rate = 12.0 / random.uniform(8.0, 18.0) if scenario == "deteriorating" else 1.0

    events = []
    base_ts = 1577836800.0  # 2020-01-01T00:00:00Z
    for hour in range(n_hours):
        charttime = base_ts + hour * 3600.0
        for feat, (mean, std) in baselines.items():
            if scenario == "deteriorating":
                prog = min(max((hour - onset_hour) * rate, 0.0), 1.0)
                if feat == "HR": mean += prog * 8 * amp; std = 6
                elif feat == "O2Sat": mean -= prog * 2 * amp; std = 1.2
                elif feat == "Temp": mean += prog * 0.4 * amp; std = 0.3
                elif feat == "SBP": mean -= prog * 12 * amp; std = 10
                elif feat == "Lactate": mean += prog * 0.5 * amp; std = 0.5
                elif feat == "WBC": mean += prog * 2.5 * amp; std = 2.0
                elif feat == "Glucose": mean += prog * 15 * amp; std = 15
                elif feat == "Creatinine": mean += prog * 0.3 * amp; std = 0.2
                elif feat == "FiO2": mean += prog * 0.12 * amp; std = 0.05
            elif scenario == "recovering":
                offset = max(0.0, 1.0 - hour / 24.0)
                if feat == "HR": mean += 30 * offset
                elif feat == "O2Sat": mean -= max(0.0, mean - 90) * offset
                elif feat == "Lactate": mean += 3.0 * offset
                elif feat == "WBC": mean += 6.0 * offset
            if random.random() < MISS_RATES[feat]:
                continue
            value = max(0.0, float(np.random.normal(mean, std)))
            events.append((float(charttime), int(ITEMIDS[feat]), round(value, 2)))
    return events


def gen_cohort(n_stays=300, seed=42):
    """Return (by_stay: {stay_id: [events]},
    scenarios: {stay_id: (scenario, onset_hour)})."""
    random.seed(seed); np.random.seed(seed)
    by_stay, scenarios = {}, {}
    for i in range(n_stays):
        stay_id = 10000 + i
        subject_id = 1000 + (i % 500)  # Some patients have multiple stays
        n_hours = random.randint(24, 168)
        scenario = random.choice(["stable", "deteriorating", "recovering"])
        onset = random.randint(4, 20) if scenario == "deteriorating" else 0
        events = gen_stay_events(stay_id, subject_id, n_hours, scenario,
                                 seed=seed + i, onset_hour=onset)
        by_stay[stay_id] = events
        scenarios[stay_id] = (scenario, onset)
        # Local generate_cohort draws careunits per stay AFTER generate_stay's
        # reseed — consume the same stream so later draws align byte-for-byte.
        random.choice(['MICU', 'SICU', 'CCU', 'TSICU'])
        random.choice(['MICU', 'SICU', 'CCU', 'TSICU'])
    return by_stay, scenarios


def prepare(by_stay, scenarios, test_frac=0.2, seed=42):
    """Per-hour windows + scenario/onset labels from in-memory stays.
    Byte-equivalent to benchmark/prepare_mimic_data's CSV path."""
    windows_all, flat_all, y_all = [], [], []
    all_values = [[] for _ in range(K)]
    stay_ranges, start = [], 0
    for stay_id in sorted(by_stay.keys()):
        events = sorted(by_stay[stay_id])
        if not events:
            continue
        t0 = events[0][0]
        T = int((events[-1][0] - t0) / 3600) + 1
        values = np.zeros((T, K), dtype=np.float32)
        mask = np.zeros((T, K), dtype=np.float32)
        delta = np.full((T, K), 48.0, dtype=np.float32)
        for ct, itemid, val in events:
            feat = MIMIC_TO_FEATURE.get(int(itemid))
            if feat not in FEATURE_INDEX:
                continue
            ci = FEATURE_INDEX[feat]
            hour = min(int((ct - t0) / 3600), T - 1)
            values[hour, ci] = val
            mask[hour, ci] = 1.0
        for c in range(K):
            last_val, last_seen = 0.0, None
            for t in range(T):
                if mask[t, c]:
                    last_val = values[t, c]
                    last_seen = t
                else:
                    values[t, c] = last_val
                    if last_seen is not None:
                        delta[t, c] = min(t - last_seen, 48.0)
        if T < WINDOW_SIZE:
            continue
        sc, onset = scenarios.get(stay_id, ("stable", 0))
        for t in range(T):
            w = max(0, t - WINDOW_SIZE + 1)
            n = t - w + 1
            window = np.zeros((WINDOW_SIZE, K3), dtype=np.float32)
            for i in range(n):
                d = WINDOW_SIZE - n + i
                window[d, :K] = values[w + i]
                window[d, K:2*K] = mask[w + i]
                window[d, 2*K:] = delta[w + i]
            windows_all.append(window)
            flat_all.append(window[-1])
            y_all.append(1 if (sc == "deteriorating" and t >= onset) else 0)
            for c in range(K):
                if mask[t, c]:
                    all_values[c].append(values[t, c])
        stay_ranges.append((start, start + T))
        start += T

    Xw = np.stack(windows_all).astype(np.float32)
    Xf = np.array(flat_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    z_mu = np.array([np.mean(v) if v else 0.0 for v in all_values], dtype=np.float32)
    z_sd = np.array([np.std(v) if v else 1.0 for v in all_values], dtype=np.float32)
    z_sd = np.where(z_sd < 1e-6, 1.0, z_sd)
    for c in range(K):
        Xw[:, :, c] = (Xw[:, :, c] - z_mu[c]) / z_sd[c]
        Xf[:, c] = (Xf[:, c] - z_mu[c]) / z_sd[c]

    np.random.seed(seed)
    n_stays = len(stay_ranges)
    order = np.random.permutation(n_stays)
    n_test = max(1, int(n_stays * test_frac))
    test_stays = set(order[:n_test])
    train_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    for si, (s, e) in enumerate(stay_ranges):
        if si in test_stays:
            test_mask[s:e] = True
        else:
            train_mask[s:e] = True

    meta = {'n_stays': n_stays, 'n_samples': len(y),
             'n_train': int(train_mask.sum()), 'n_test': int(test_mask.sum()),
             'sepsis_rate': float(y.mean()),
             'z_mu': z_mu.tolist(), 'z_sd': z_sd.tolist(),
             'window_size': WINDOW_SIZE, 'test_frac': test_frac}
    return (Xw[train_mask], Xf[train_mask], y[train_mask],
            Xw[test_mask], Xf[test_mask], y[test_mask], meta)


def compute_auroc(y_true, y_score):
    from scipy.stats import mannwhitneyu
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u, _ = mannwhitneyu(pos, neg, alternative='greater')
    return u / (len(pos) * len(neg))


def compute_auprc(y_true, y_score):
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def compute_utility(y_true, y_score, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    best_util, best_thr = -float('inf'), 0.5
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        util = tp * 1.0 + fp * (-0.05) + fn * (-1.0)
        n_pos = y_true.sum()
        if n_pos > 0:
            util = util / n_pos
        if util > best_util:
            best_util, best_thr = util, thr
    return best_util, best_thr


def train_gbm(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import HistGradientBoostingClassifier
    start_time = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, l2_regularization=0.1, random_state=42)
    model.fit(X_train, y_train)
    return {'model': model, 'train_time': time.time() - start_time,
             'test_probs': model.predict_proba(X_test)[:, 1],
             'n_iter': model.n_iter_}


def train_grid(X_train, y_train, X_test, y_test, n_cells=100, k_active=3,
               n_epochs=50, lr=1e-3, batch_size=512, hidden_dim=128, seed=42):
    """Institutional Grid: GRU encoder over (N, W, K*3) + top-k MoE cells
    with pie-chart economy. Mirrors benchmark/run_grid_vs_gbm.train_grid.
    Training upgrades: mini-batch + intra-epoch shuffle, cosine LR schedule,
    wider GRU + dropout."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)

    class TemporalEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True,
                              dropout=0.1)
            self.drop = nn.Dropout(0.1)
        def forward(self, x):
            _, h = self.gru(x)
            return self.drop(h[-1])

    class GridCell(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.W = nn.Parameter(torch.randn(input_dim) * 0.01)
            self.b = nn.Parameter(torch.zeros(1))
            self.pie_logits = nn.Parameter(torch.zeros(3))
        def forward(self, h):
            confidence = torch.sigmoid(h @ self.W + self.b)
            pie = F.softmax(self.pie_logits, dim=-1).unsqueeze(0).expand(h.shape[0], -1)
            return confidence, pie
        def compute_bid(self, h, governance=None):
            confidence, pie = self.forward(h)
            risk_scale = governance.get('risk_scale', 1.0) if governance else 1.0
            penalty = torch.ones_like(confidence) - pie[:, 1] * risk_scale * 0.1
            return confidence * penalty.clamp(min=0.0)

    class GridModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_cells, k):
            super().__init__()
            self.encoder = TemporalEncoder(input_dim, hidden_dim)
            self.cells = nn.ModuleList([GridCell(hidden_dim) for _ in range(n_cells)])
            self.k = k
        def forward(self, x, governance=None):
            h = self.encoder(x)
            conf_list, pie_list, bid_list = [], [], []
            for cell in self.cells:
                conf, pie = cell(h)
                bid = cell.compute_bid(h, governance)
                conf_list.append(conf); pie_list.append(pie); bid_list.append(bid)
            confidences = torch.stack(conf_list, dim=-1)
            pies = torch.stack(pie_list, dim=-1)
            bids = torch.stack(bid_list, dim=-1)
            _, top_idx = torch.topk(bids, self.k, dim=-1)
            selected = torch.gather(confidences, 1, top_idx).mean(dim=-1)
            selected = selected.clamp(1e-6, 1 - 1e-6)
            logits = torch.log(selected / (1 - selected))
            return {'predictions': logits, 'bids': bids, 'pie_weights': pies,
                     'selected_idx': top_idx}

    model = GridModel(K3, hidden_dim, n_cells, k_active)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = int(np.ceil(len(y_train) / batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * n_batches)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)])
    X_t = torch.from_numpy(np.ascontiguousarray(X_train))
    y_t = torch.from_numpy(y_train).float()
    start_time = time.time()
    losses = []
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(y_train))
        epoch_losses = []
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = X_t[idx]; yb = y_t[idx]
            optimizer.zero_grad()
            out = model(xb, governance={'cost_scale': 1.0, 'risk_scale': 1.0})
            loss = F.binary_cross_entropy_with_logits(
                out['predictions'].squeeze(), yb, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss))
        losses.append(float(np.mean(epoch_losses)))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1}/{n_epochs} loss {losses[-1]:.4f}", flush=True)
    train_time = time.time() - start_time
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(np.ascontiguousarray(X_test)))
        test_probs = 1 / (1 + np.exp(-out['predictions'].squeeze()))
        test_probs = test_probs.numpy()
        bids = out['bids'][0].numpy()
        pie = out['pie_weights'][0].numpy()
        selected = out['selected_idx'][0].numpy()
    return {'model': model, 'train_time': train_time, 'test_probs': test_probs,
             'train_losses': losses,
             'cell_bids': bids.tolist(), 'pie_weights': pie.tolist(),
             'selected_cells': selected.tolist()}


def main():
    print("[1/5] Generating synthetic cohort...", flush=True)
    by_stay, scenarios = gen_cohort(n_stays=300, seed=SEED)
    print(f"  {len(by_stay)} stays", flush=True)

    print("[2/5] Preparing tensors...", flush=True)
    Xw_tr, Xf_tr, y_tr, Xw_te, Xf_te, y_te, meta = prepare(
        by_stay, scenarios, test_frac=0.2, seed=SEED)
    print(f"  {meta['n_samples']} samples, sepsis {meta['sepsis_rate']:.1%}",
          flush=True)
    if abs(meta['n_samples'] - 28095) > 100 or abs(meta['sepsis_rate'] - 0.2718) > 0.02:
        print(f"PARITY_WARN: cohort meta drifted {meta['n_samples']}/"
              f"{meta['sepsis_rate']:.3f} vs local 28095/0.2718", flush=True)

    print("[3/5] Training GBM (flat last-hour features)...", flush=True)
    gbm_res = train_gbm(Xf_tr, y_tr, Xf_te, y_te)
    gbm_auc = compute_auroc(y_te, gbm_res['test_probs'])
    gbm_ap = compute_auprc(y_te, gbm_res['test_probs'])
    gbm_util, gbm_thr = compute_utility(y_te, gbm_res['test_probs'])
    print(f"  GBM AUROC {gbm_auc:.4f} AUPRC {gbm_ap:.4f} util {gbm_util:.4f} "
          f"({gbm_res['train_time']:.1f}s)", flush=True)
    if abs(gbm_auc - 0.9116) > 0.02:
        print(f"PARITY_WARN: GBM AUROC {gbm_auc:.4f} != local 0.9116 — "
              f"generator/prepare drift detected", flush=True)

    print("[4/5] Training Grid (GRU + 100 MoE cells, batch 512)...", flush=True)
    grid_res = train_grid(Xw_tr, y_tr, Xw_te, y_te, n_epochs=50,
                          batch_size=512, hidden_dim=128)
    g_auc = compute_auroc(y_te, grid_res['test_probs'])
    g_ap = compute_auprc(y_te, grid_res['test_probs'])
    g_util, g_thr = compute_utility(y_te, grid_res['test_probs'])
    print(f"  GRID AUROC {g_auc:.4f} AUPRC {g_ap:.4f} util {g_util:.4f} "
          f"({grid_res['train_time']:.1f}s)", flush=True)

    d_auc, d_ap, d_util = g_auc - gbm_auc, g_ap - gbm_ap, g_util - gbm_util
    if d_auc > 0.02 and d_util > 0.05:
        verdict = "GRID DOMINATES"
    elif d_auc > -0.02 and d_util > -0.05:
        verdict = "PARITY ACHIEVED"
    elif d_auc > -0.05:
        verdict = "ACCEPTABLE TRADEOFF"
    else:
        verdict = "GBM WINS ON AUC"

    def _serialize(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    report = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'kernel': 'grid_vs_gbm_benchmark',
        'data': meta,
        'gbm': {'auroc': gbm_auc, 'auprc': gbm_ap, 'utility': gbm_util,
                 'threshold': gbm_thr, 'train_time': gbm_res['train_time'],
                 'n_iter': gbm_res['n_iter'],
                 'auditability': 'black-box: N/A'},
        'grid': {'auroc': g_auc, 'auprc': g_ap, 'utility': g_util,
                  'threshold': g_thr, 'train_time': grid_res['train_time'],
                  'n_cells': 100, 'n_epochs': 50,
                  'auditability': 'cell bids + pie economy + governance trail',
                  'cell_meta': {'cell_bids': grid_res['cell_bids'],
                                 'pie_weights': grid_res['pie_weights'],
                                 'selected_cells': grid_res['selected_cells'],
                                 'train_losses': grid_res['train_losses']}},
        'deltas': {'auroc': d_auc, 'auprc': d_ap, 'utility': d_util,
                    'train_time': grid_res['train_time'] - gbm_res['train_time']},
        'verdict': verdict,
    }
    with open('/kaggle/working/benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=_serialize)
    with open('/kaggle/working/benchmark_summary.txt', 'w') as f:
        f.write(f"SEED={SEED}  {meta['n_samples']} samples\n")
        f.write(f"GBM : AUROC {gbm_auc:.4f}  AUPRC {gbm_ap:.4f}  util {gbm_util:.4f}\n")
        f.write(f"GRID: AUROC {g_auc:.4f}  AUPRC {g_ap:.4f}  util {g_util:.4f}\n")
        f.write(f"DELTA: {d_auc:+.4f} / {d_ap:+.4f} / {d_util:+.4f}\n")
        f.write(f"VERDICT: {verdict}\n")
    print("\n=== SUMMARY ===", flush=True)
    print(f"GBM : AUROC {gbm_auc:.4f} AUPRC {gbm_ap:.4f} util {gbm_util:.4f}", flush=True)
    print(f"GRID: AUROC {g_auc:.4f} AUPRC {g_ap:.4f} util {g_util:.4f}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
