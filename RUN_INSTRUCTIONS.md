# Running changed.py / changed.ipynb on your machine

## 1. Install (once)
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # PyTorch: if you have an NVIDIA GPU install the CUDA build from pytorch.org instead
```

## 2. Check that everything runs (30-60 min, tiny sample sizes -- the numbers mean nothing)
```bash
python run_quick_test.py
```

## 3. Real run in the background (survives closing the terminal)

Pick a profile with `QKD_PROFILE`:

| profile | what it is |
|---|---|
| `quick` | tiny, only proves the code runs |
| `standard` | the sizes of the earlier drafts (hours) |
| `large` (default) | many more runs and seeds so classifier / DL differences are resolved (many hours; a GPU helps Sections 18 and 24) |
| `xl` | `large` plus 1,000+1,000 deep-learning sessions per protocol in Section 24, 10 seeds, 30 epochs |

**Linux / macOS**
```bash
QKD_PROFILE=large nohup python -u changed.py > run.log 2>&1 &
tail -f run.log                      # watch progress (Ctrl-C stops watching, not the run)
```
or inside `tmux` / `screen`: `QKD_PROFILE=large python -u changed.py 2>&1 | tee run.log`

**Windows (PowerShell)**
```powershell
$env:QKD_PROFILE = 'large'
Start-Process python -ArgumentList '-u','changed.py' -RedirectStandardOutput run.log -RedirectStandardError run.err -WindowStyle Hidden
Get-Content run.log -Wait            # watch progress
```

**Windows (cmd)**
```bat
set QKD_PROFILE=large
start /B python -u changed.py > run.log 2>&1
```

Stop it: `pkill -f changed.py` (Linux/macOS) or end the python process in Task Manager.

## 4. Change a single setting without editing the file
```bash
QKD_PROFILE=large QKD_SC_OVERRIDE='{"s24_dl_clean": 300, "s24_dl_att": 300, "s24_dl_seeds": [0,1,2]}' nohup python -u changed.py > run.log 2>&1 &
```
All keys are in the `SC = dict(...)` block in Section 0.

## 5. Outputs
`data/` (CSVs, `changed_manifest.json`, `ledger_holm.csv`) and `plots/`. The last lines of `run.log` list any file that this run did not regenerate. Datasets and deep-learning data are cached under a content fingerprint, so an interrupted run reuses finished work; delete `data/` for a completely fresh run.

## 6. Notebook instead of script
`jupyter lab changed.ipynb` -> Kernel > Restart & Run All (same profiles via `QKD_PROFILE` set before starting Jupyter). For a long run the script above is more robust.

## 7. Ubuntu: one block, background run, results pushed to the `v2_results` branch
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git tmux && git clone -b v2_results https://github.com/himeshsoni704/QuantumKeyDistribution.git && cd QuantumKeyDistribution && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && (git config user.name >/dev/null || git config user.name "qkd-runner") && (git config user.email >/dev/null || git config user.email "qkd@example.com") && tmux new -d -s qkd "./run_bg.sh large; sleep infinity" && echo "running in tmux session 'qkd' -- watch: tail -f run.log | pushes: tail -f autopush.log | stop: ./stop_bg.sh"
```
Results appear on `origin/v2_results` under `results/<profile>-<date>/` (data CSV/JSON, plots, log tail) every 10 minutes and at the end. `git push` must work non-interactively on that machine (SSH key or saved token). Profile: replace `large` by `quick|standard|xl`; interval: `PUSH_EVERY_SECONDS=300`.
