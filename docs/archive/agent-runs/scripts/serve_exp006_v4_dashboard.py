from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[2]
CAMPAIGN=ROOT/"var/agent_runs/burgers-rad-exp006-v4"
os.environ["AF_PIKAN_CAMPAIGN"]=str(CAMPAIGN)
os.environ["AF_PIKAN_BUNDLE"]=str(CAMPAIGN/"bundle-v6-gpu-20260926")
os.environ["AF_PIKAN_STATUS_SCRIPT"]=str(ROOT/"var/agent_runs/status_exp006_v4.py")
sys.path.insert(0,str(ROOT))
import src.research_pipeline.remote.dashboard as dashboard
dashboard.PAGE=dashboard.PAGE.replace(
    "Monitor de campaña · Heat + Burgers · Kaggle",
    "Monitor de campaña · Burgers · Sobol vs RAD · Kaggle"
)
dashboard.main()
