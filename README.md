# MLB Edge — Setup & Run Guide

## What This Is

A multi-tab Streamlit web app that displays:
- **Games & Moneylines** — Today's game cards with model picks, EV, odds; tap a card to drill into the full matchup (pitching lines + lineup projections)
- **Pitcher Projections** — Sortable table of projected IP, K, BB, H, ER for each starter
- **Hitter Projections** — Lineup cards per matchup or full sortable table (PA, H, HR, K, BB, R)
- **Season Accuracy** — Cumulative pick accuracy chart + recent pick log

---

## Folder Structure

```
mlb_edge/
├── app.py                      ← Main Streamlit app (run this)
├── daily_mlb_model_runner.py   ← Your betting model runner
├── hitterspitchers_today.py    ← Your projection runner
├── hitterspitchers_data.py     ← Data helpers
├── hitterspitchers_train.py    ← Training scripts
├── daily_update.py             ← Full daily pipeline
├── betting_model.pkl           ← Trained model file
├── 2025_model_data.csv         ← Historical training data
├── 2026_picks_accuracy.csv     ← Season pick log (auto-created)
├── requirements.txt
├── .streamlit/
│   └── config.toml             ← Theme + server config
├── pages/
│   ├── games_tab.py
│   ├── pitchers_tab.py
│   ├── hitters_tab.py
│   └── accuracy_tab.py
├── components/
│   └── data_loader.py
├── outputs/                    ← Auto-created; model saves CSVs here
│   ├── today_predictions_with_ev.csv
│   └── hitterspitchers_today.csv
├── data/                       ← Place your pitcher/hitter game CSVs here
└── models/                     ← Place your hitter/pitcher pkl models here
```

---

## Step 1 — Install on Your Computer

### Requirements
- Python 3.11+
- pip

```bash
# Clone or copy the mlb_edge folder to your computer
cd ~/Documents/mlb_edge       # or wherever you put it

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate      # Mac/Linux
# OR on Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2 — Copy Your Files In

Copy these files INTO the `mlb_edge/` root folder:
```
betting_model.pkl
2025_model_data.csv
daily_mlb_model_runner.py
hitterspitchers_today.py
hitterspitchers_data.py
hitterspitchers_train.py
daily_update.py
```

If you have pitcher/hitter data CSVs (pitcher_game_data.csv, hitter_game_data.csv, etc.), 
put them in `mlb_edge/data/`.

If you have your hitter/pitcher ML models (pitcher_K_rate.pkl, etc.),
put them in `mlb_edge/models/`.

---

## Step 3 — Run Locally

```bash
cd ~/Documents/mlb_edge
source .venv/bin/activate      # activate your venv
streamlit run app.py
```

Open your browser to: **http://localhost:8501**

---

## Step 4 — Daily Update Workflow

### Option A — From the app (click buttons)
Each tab has a "Run Model" / "Run Projections" / "Grade Picks" button that calls your scripts directly.

### Option B — From terminal (recommended for automation)
```bash
cd ~/Documents/mlb_edge
source .venv/bin/activate

# Run everything: backfill + today's model + grade picks
python daily_update.py

# OR run only projections (pitchers + hitters)
python hitterspitchers_today.py

# OR run only the betting model
python daily_mlb_model_runner.py
```

### Option C — Schedule it automatically (Mac/Linux cron)
Run at 10am every day:
```bash
crontab -e
# Add this line:
0 10 * * * cd /Users/yourname/Documents/mlb_edge && .venv/bin/python daily_update.py >> logs/daily.log 2>&1
```

---

## Step 5 — Publish to the Web (Streamlit Community Cloud)

This is 100% free for public apps.

### 5a — Put your code on GitHub

1. Create a free account at [github.com](https://github.com)
2. Create a new **private** repository (so your API key stays private)
3. Push your code:
```bash
cd ~/Documents/mlb_edge
git init
git add .
git commit -m "MLB Edge initial commit"
git remote add origin https://github.com/YOUR_USERNAME/mlb-edge.git
git push -u origin main
```

> ⚠️ Before pushing, replace your ODDS_API_KEY in app.py with:
> `ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")`
> Then set it as a secret in Streamlit Cloud (step 5c).

### 5b — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - Repository: `YOUR_USERNAME/mlb-edge`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy**

### 5c — Add your secrets (API key)

In Streamlit Cloud → your app → **Settings → Secrets**, add:
```toml
ODDS_API_KEY = "afa28350c34fba9f318ecd7ae4e21b63"
```

### 5d — Add large files (models + data)

GitHub won't accept files over 100MB. Options:
- If your `.pkl` and `.csv` are under 100MB, they'll push fine
- If they're bigger, use [Git LFS](https://git-lfs.com) or store them in Google Drive and load via URL in your code

---

## Environment Variable Reference

| Variable | Description |
|---|---|
| `ODDS_API_KEY` | The Odds API key (set as Streamlit secret in prod) |

---

## Troubleshooting

**"No games found"** — The MLB Stats API is public; check your internet. Games appear around 10am ET.

**"No pitcher projections"** — Run `python hitterspitchers_today.py` first. Check that your `data/` folder has the required CSVs.

**"ModuleNotFoundError: hitterspitchers_today"** — Make sure `hitterspitchers_today.py` is in the same folder as `app.py`.

**App is slow on first load** — Normal. Streamlit caches API responses for 5 minutes (TTL=300s in data_loader.py). Subsequent loads are instant.

---

## Retraining After the HR/BB Improvements

These changes are now applied:
- `hitterspitchers_data.py` — rolling windows expanded to 5, 10, **20, 50** games
- `hitterspitchers_train.py` — last20/last50 features added; shallower+larger-leaf RF and stronger-regularized XGBoost for `hr_rate`, `bb_rate`, `BB_rate`, `HR_rate`; log1p target transform for sparse rates instead of logit

**You must re-run data build + retrain to get improved predictions:**

```bash
# Step 1: Rebuild the game-level CSVs with the new 20/50-game windows
python hitterspitchers_data.py --input data/pitch_data_2025.csv

# Step 2: Retrain all models (picks best of RF / XGBoost / NN per target)
python hitterspitchers_train.py \
  --pitcher-data data/pitcher_game_data.csv \
  --hitter-data  data/hitter_game_data.csv \
  --model-dir    models/

# Step 3: Run today's projections with the new models
python hitterspitchers_today.py
```

After Step 2 you'll see per-target RMSE printed. For `hr_rate` and `BB_rate` you should
see test RMSE drop and `pred_mean` get closer to `actual_mean` vs before.
