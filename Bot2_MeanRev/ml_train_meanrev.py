"""
ml_train_meanrev.py
===================
Обучение ML модели для Bot2 MeanRev.
Запускай раз в неделю когда накопятся новые сделки.

Запуск:
  cd C:\\TradingBots\\Bot2_MeanRev
  python ml_train_meanrev.py

Результат:
  ml_model_meanrev.pkl — модель для фильтрации сигналов LONG и SHORT
"""

import os, sys, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

DATASET_PATH = "C:\\TradingBots\\ML\\ml_dataset.csv"
MODEL_PATH   = "C:\\TradingBots\\Bot2_MeanRev\\ml_model_meanrev.pkl"
MIN_TRADES   = 60

FEATURES = [
    'rsi',        'adx',        'bb_width',
    'vol_ratio',  'hour',       'day_of_week',
    'is_london',  'is_newyork', 'regime_conf',
    'dir_enc',    'regime_enc',
]


def main():
    print("=" * 55)
    print("  ML ОБУЧЕНИЕ — Bot2 MeanRev")
    print("=" * 55)

    if not os.path.exists(DATASET_PATH):
        print(f"[!] Файл не найден: {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    b2 = df[df['bot'] == 'MeanRev'].copy()
    print(f"\n  Сделок: {len(b2)}  WR={b2.target.mean()*100:.1f}%")

    if len(b2) < MIN_TRADES:
        print(f"[!] Мало данных ({len(b2)}), нужно {MIN_TRADES}+")
        sys.exit(0)

    le_dir = LabelEncoder()
    le_reg = LabelEncoder()
    b2['dir_enc']    = le_dir.fit_transform(b2['direction'])
    b2['regime_enc'] = le_reg.fit_transform(b2['regime'].fillna('?'))
    available = [f for f in FEATURES if f in b2.columns]
    X, y = b2[available].fillna(0), b2['target']

    # CV
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf   = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   random_state=42, max_depth=4, min_samples_leaf=6)
    aucs = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
    print(f"  CV AUC: {aucs.mean():.3f} ±{aucs.std():.3f}")

    # Walk-forward
    split = int(len(X) * 0.70)
    cal_wf = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=42, max_depth=4, min_samples_leaf=6),
        cv=5, method='isotonic')
    cal_wf.fit(X.iloc[:split], y.iloc[:split])
    proba_test = cal_wf.predict_proba(X.iloc[split:])[:,1]
    auc_oos = roc_auc_score(y.iloc[split:], proba_test) if len(y.iloc[split:].unique())>1 else 0.5
    base_wr = y.iloc[split:].mean()*100
    print(f"  OOS AUC: {auc_oos:.3f}  базовый WR: {base_wr:.1f}%\n")
    print(f"  {'Порог':<8}{'Сделок':<8}{'WR':<8}{'vs база':<10}{'Покр.'}")
    print(f"  {'-'*42}")
    for t in [0.35,0.38,0.40,0.42,0.45,0.48,0.50]:
        mask = proba_test >= t
        n = mask.sum()
        if n >= 3:
            wr  = y.iloc[split:][mask].mean()*100
            cov = n/len(y.iloc[split:])*100
            sgn = '+' if wr-base_wr>=0 else ''
            print(f"  {t:<8.2f}{n:<8}{wr:<8.1f}{sgn}{wr-base_wr:.1f}%      {cov:.0f}%")

    # Финальная модель
    final = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=42, max_depth=4, min_samples_leaf=6),
        cv=5, method='isotonic')
    final.fit(X, y)
    proba_all = final.predict_proba(X)[:,1]

    # Оптимальный порог (покрытие >= 35%)
    best_t, best_wr = 0.40, 0
    for t in np.arange(0.30, 0.60, 0.01):
        mask = proba_all >= t
        if mask.sum()/len(y) >= 0.35:
            wr = y[mask].mean()*100
            if wr > best_wr: best_wr, best_t = wr, t

    print(f"\n  Порог: {best_t:.2f}  WR={best_wr:.1f}%  покрытие={(proba_all>=best_t).sum()/len(y)*100:.0f}%")

    # По направлениям
    b2c = b2.copy(); b2c['prob'] = proba_all
    print(f"\n  {'Dir':<8}{'Одобрено':<10}{'WR одобр.':<12}{'WR отклон.'}")
    print(f"  {'-'*40}")
    for d in ['LONG','SHORT']:
        g  = b2c[b2c['direction']==d]
        ap = g[g['prob']>=best_t]; rj = g[g['prob']<best_t]
        wa = (ap.result=='WIN').mean()*100 if len(ap) else 0
        wr2 = (rj.result=='WIN').mean()*100 if len(rj) else 0
        print(f"  {d:<8}{len(ap)}/{len(g):<8}{wa:<12.0f}{wr2:.0f}%")

    # Важность
    rf_i = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   random_state=42, max_depth=4, min_samples_leaf=6)
    rf_i.fit(X, y)
    imp = pd.Series(rf_i.feature_importances_, index=available).sort_values(ascending=False)
    print(f"\n  Важность признаков:")
    for feat, val in imp.head(6).items():
        print(f"    {feat:<15} {val:.3f}  {'█'*int(val*50)}")

    payload = {
        'model': final, 'features': available,
        'le_dir': le_dir, 'le_reg': le_reg,
        'threshold': round(best_t,2),
        'auc_cv': round(aucs.mean(),3),
        'auc_oos': round(auc_oos,3),
        'base_wr': round(b2.target.mean()*100,1),
        'wr_at_thresh': round(best_wr,1),
        'trained_on': len(b2),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)

    print(f"\n  [+] Сохранено: {MODEL_PATH}")
    print(f"  Базовый WR: {b2.target.mean()*100:.1f}%  →  WR с ML: {best_wr:.1f}%")
    print("=" * 55)
    input("\n  Нажми Enter...")


if __name__ == "__main__":
    main()
