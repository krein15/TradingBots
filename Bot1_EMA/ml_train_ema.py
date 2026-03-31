"""
ml_train_ema.py
===============
Переобучение ML модели для бота #1 EMA.
Запускай раз в неделю когда накопятся новые сделки.

Запуск:
  python ml_train_ema.py

Результат:
  ml_model_ema.pkl — обновлённая модель
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import pickle, os, warnings
warnings.filterwarnings("ignore")

# Путь к датасету
DATASET = os.path.join("C:\\TradingBots", "ML", "ml_dataset.csv")
MODEL   = os.path.join("C:\\TradingBots", "Bot1_EMA", "ml_model_ema.pkl")

def train():
    if not os.path.exists(DATASET):
        print(f"[!] Датасет не найден: {DATASET}")
        print(f"    Запусти сначала: python ml_data_collector.py")
        return

    df  = pd.read_csv(DATASET)
    ema = df[df["bot"] == "EMA"].copy()
    print(f"EMA сделок: {len(ema)}  WR={round(ema["target"].mean()*100,1)}%")

    if len(ema) < 50:
        print("[!] Мало данных — нужно минимум 50 EMA сделок")
        return

    features = [
        # Риск-менеджмент
        "risk_pct",           # размер стопа — главный признак
        # Объём
        "vol_ratio",          # объём на входе
        # Время
        "hour",
        "day_of_week",
        "is_weekend",
        "is_london",
        "is_newyork",
        "is_night",
        "is_asia",
        # Режим рынка
        "regime_conf",
        "dir_enc",
        "regime_enc",
        # Качество импульса
        "body_ratio",         # чистота свечи (0=доджи, 1=чистая)
        "impulse_strength",   # сила в единицах ATR
        "consec_candles",     # серия свечей в одну сторону
        "dist_ema50",         # расстояние от EMA50
        "atr_pct",            # перцентиль ATR (высокая ли волатильность)
        # Рыночный контекст
        "spread_pct",         # спред bid/ask (ликвидность)
        "rel_strength",       # сила монеты vs BTC
        "btc_momentum",       # скорость BTC
    ]

    le_dir = LabelEncoder()
    le_reg = LabelEncoder()
    ema["dir_enc"]    = le_dir.fit_transform(ema["direction"])
    ema["regime_enc"] = le_reg.fit_transform(ema["regime"].fillna("?"))
    X = ema[features].fillna(0)
    y = ema["target"]

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf  = RandomForestClassifier(
        n_estimators=300, max_depth=4,
        min_samples_leaf=8, class_weight="balanced",
        random_state=42)
    auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"ROC-AUC: {auc.mean():.3f} ± {auc.std():.3f}")

    model = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=300, max_depth=4,
            min_samples_leaf=8, class_weight="balanced",
            random_state=42),
        cv=5, method="isotonic")
    model.fit(X, y)

    # Оптимальный порог
    proba = model.predict_proba(X)[:, 1]
    ema["prob"] = proba
    best_t, best_wr = 0.35, 0
    for t in np.arange(0.25, 0.55, 0.01):
        f = ema[ema["prob"] >= t]
        if len(f) >= 15:
            wr = f["target"].mean()
            if wr > best_wr:
                best_wr, best_t = wr, t

    model_data = {
        "model":      model,
        "features":   features,
        "le_dir":     le_dir,
        "le_reg":     le_reg,
        "trained_on": len(ema),
        "wr_base":    round(y.mean()*100,1),
        "auc":        round(auc.mean(),3),
        "threshold":  round(best_t, 2),
        "threshold_wr": round(best_wr*100,1),
    }

    with open(MODEL, "wb") as f:
        pickle.dump(model_data, f)

    print(f"[+] Модель сохранена: {MODEL}")
    print(f"    Порог: p>={round(best_t,2)}  WR={round(best_wr*100,1)}%")
    input("Нажми Enter...")

if __name__ == "__main__":
    train()
