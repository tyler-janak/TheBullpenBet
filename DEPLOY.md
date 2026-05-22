# TheBullpenBet — Deploy Guide (V1, no paywall)

Get the site live in **under an hour**. Free, fully automated.

---

## How it works

```
   ┌──────────────────────┐
   │  GitHub Actions      │   Twice a day (4 AM ET + 12 PM ET):
   │  Daily Cron          │   1. checkout repo (with LFS for data/)
   │                      │   2. pip install -r requirements.txt
   │                      │   3. python daily_update.py
   │                      │   4. python hitterspitchers_today.py
   │                      │   5. commit + push fresh CSVs
   └──────────┬───────────┘
              │ (pushes to main)
              ▼
   ┌──────────────────────┐
   │  Render Web Service  │   Auto-redeploys on every push.
   │  uvicorn server:app  │   Serves index.html + /api/* endpoints
   │                      │   that read the freshly-pushed CSVs.
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │  thebullpenbet.       │
   │  onrender.com        │   Public URL — share with friends.
   └──────────────────────┘
```

You do this once. After that the site updates itself forever.

---

## Step 1 — Test the server locally (5 min)

Make sure the server works on your PC first.

```powershell
# from the project folder
cd "C:\Users\tyler\OneDrive\Documents\Machine Learning\MLB Betting Website"
.\.venv\Scripts\Activate.ps1
pip install -r requirements-server.txt
uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000** in your browser. You should see the dark TheBullpenBet site populated with whatever's in your `outputs/` CSVs.

If you see "No games today" everywhere, run your existing pipeline once first:

```powershell
python daily_update.py
python hitterspitchers_today.py
```

Then refresh the browser.

When it's working, kill the server (Ctrl+C).

---

## Step 2 — Install Git LFS (5 min)

LFS lets you commit the big `data/*.csv` files to GitHub without bloating the repo.

```powershell
# Install Git LFS (one-time on your machine)
winget install GitHub.GitLFS
# or download from: https://git-lfs.com

# Initialize in the repo
cd "C:\Users\tyler\OneDrive\Documents\Machine Learning\MLB Betting Website"
git lfs install
git lfs track "data/*.csv"
git add .gitattributes
```

---

## Step 3 — Push to GitHub (10 min)

### 3a. Create a GitHub account
https://github.com/signup — free, takes 2 min.

### 3b. Create a new repo
https://github.com/new
- Name: `thebullpenbet` (anything works)
- Visibility: **Private** is fine (Render can still deploy it)
- Don't initialize with README/gitignore (you already have them)
- Click **Create repository**

### 3c. Push your code

```powershell
cd "C:\Users\tyler\OneDrive\Documents\Machine Learning\MLB Betting Website"

# First push
git init
git branch -M main
git add .
git commit -m "Initial commit — TheBullpenBet v1"
git remote add origin https://github.com/YOUR_USERNAME/thebullpenbet.git
git push -u origin main
```

> If `git push` is slow or fails on the LFS files: the `data/` folder is ~106MB. GitHub LFS free tier allows 1GB storage and 1GB/mo bandwidth, which is fine for the cron's checkout traffic. If you ever hit the cap, GitHub will warn you and you can buy a $5/mo data pack.

---

## Step 4 — Set the Odds API key as a secret (2 min)

The daily cron needs your Odds API key. **Don't put it in code.**

1. Go to your repo on GitHub → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `ODDS_API_KEY`
4. Value: `afa28350c34fba9f318ecd7ae4e21b63` *(or whatever your key is)*
5. Save.

`daily_update.py` already reads it via `os.environ.get("ODDS_API_KEY")`.

---

## Step 5 — Test the cron once manually (3 min)

Before relying on the schedule, trigger the workflow once by hand to make sure it works.

1. Go to your repo → **Actions** tab
2. Click **Daily MLB Update** in the left sidebar
3. Click **Run workflow** → **Run workflow** (the green button)
4. Wait ~5–10 min for it to finish
5. If it goes green ✅, check the latest commit on `main` — you should see a "Daily update — …" commit from `github-actions[bot]`.

If it fails, click into the failed run, expand the failing step, and the error log tells you exactly what went wrong (usually a missing data file or an Odds API rate-limit).

---

## Step 6 — Deploy to Render (10 min)

### 6a. Create a Render account
https://render.com/register — free, sign in with GitHub so it can see your repo.

### 6b. Deploy via Blueprint

1. Click **New +** → **Blueprint**
2. Connect your `thebullpenbet` repo
3. Render reads `render.yaml` and creates the web service automatically
4. Click **Apply**
5. Wait ~3–5 min for the first build

When it's done you'll get a URL like `https://thebullpenbet.onrender.com`. Open it.

### 6c. Confirm autodeploy is on

Render → your service → **Settings** → **Build & Deploy** → make sure **Auto-Deploy** is **Yes**. (It should already be — `render.yaml` sets it.)

That means every time the daily cron commits new CSVs, Render redeploys automatically with the fresh data.

---

## Step 7 — Share with friends (0 min)

That's it. Send them the Render URL.

The site stays in sync because:
- Cron runs at 4 AM ET → commits CSVs → Render redeploys → fresh data live by ~4:10 AM
- Cron runs at 12 PM ET → commits CSVs → Render redeploys → fresh data live by ~12:10 PM

---

## Notes & gotchas

**Render free tier spins down after 15 min idle.** First visit after a spin-down takes ~30 seconds to wake up. If that's a problem, upgrade to the Starter plan ($7/mo, always-on).

**GitHub Actions cron timing.** GitHub doesn't guarantee exact scheduled times — runs can be delayed by 5–15 min during peak load. So 4 AM ET means "around 4 AM ET, give or take." Fine for this use case.

**Daylight Savings.** The workflow has 4 cron lines (DST and standard time variants for both 4 AM and 12 PM). The script is idempotent so the extra fires during DST transition weeks just no-op. Don't worry about it.

**Rerunning from earlier in the day.** Use the **Run workflow** button on the Actions tab any time — it reruns the full pipeline immediately.

**Cost summary.**
- GitHub Actions: free (2000 min/mo, you'll use ~30 min/day = ~900 min/mo)
- GitHub LFS: free (1GB / 1GB) — should be enough; bandwidth is the variable cost
- Render: free for always-on-when-someone-visits, $7/mo for always-on-period
- Total for v1: **$0/mo**

---

## When you're ready for the paywall (v2)

The HTML and server have stubs for Stripe + Supabase already. To enable:

1. Sign up for Supabase (free) — create a `user_profiles` table per the schema in the `index.html` header comment
2. Sign up for Stripe — create the monthly + yearly products, get publishable key + price IDs
3. Replace placeholder values in `index.html` and `server.py`
4. Flip `PAYWALL_ENABLED = true` in `index.html`
5. Add the `/api/create-checkout` Stripe handler in `server.py`
6. Push — Render auto-deploys

Estimated time when you're ready: ~45 min.

---

## Troubleshooting

**Site shows "No games today" forever** → The daily cron hasn't run yet, or `outputs/today_predictions_with_ev.csv` is empty. Check Actions tab → latest run.

**Cron says "ODDS_API_KEY not set"** → You skipped step 4. Set the secret.

**Cron fails with `ModuleNotFoundError`** → Some package is missing from `requirements.txt`. Add it and push.

**Render service won't start** → Check the deploy logs. Usually means `requirements-server.txt` is missing a dep, or the Python version mismatch.

**LFS bandwidth exceeded** → You hit the 1GB/mo cap. Options: buy a $5 data pack, or move data files to Cloudflare R2 / Backblaze B2 (free tiers, ~$0/mo for this size).

**Site loads but tabs are empty** → Open browser DevTools → Network tab → reload. Check the `/api/games` request status. If 500, check Render logs. If 200 with empty array, the cron hasn't produced data yet.
