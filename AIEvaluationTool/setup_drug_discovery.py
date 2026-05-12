"""
setup_drug_discovery.py
Sets up the CeRAI SQLite database with:
  - Drug Discovery Assistant target (API type, LOCAL/Claude proxy)
  - Drug discovery test cases
  - All required strategies, plans, metrics
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from lib.data import (Prompt, TestCase, Response, TestPlan, Metric, 
                      LLMJudgePrompt, Target, Run, RunDetail)
from lib.orm import DB

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "data", "AIEvaluationData.db")
DB_URL = f"sqlite:///{DB_PATH}"
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# ── Connect ───────────────────────────────────────────────────────────────────
db = DB(db_url=DB_URL, debug=False)
print(f"Connected to {DB_PATH}")

# ── Strategies ────────────────────────────────────────────────────────────────
strategies_raw = json.load(open(os.path.join(BASE_DIR, "data/strategy_id.json")))
for sid, sname in strategies_raw.items():
    db.add_or_get_strategy_id(strategy_name=sname)
print(f"Imported {len(strategies_raw)} strategies")

# ── Plans & Metrics from plans.json ──────────────────────────────────────────
plans = json.load(open(os.path.join(BASE_DIR, "data/plans.json")))
domain_general = db.add_or_get_domain_id(domain_name="general")
lang_auto = db.add_or_get_language_id(language_name="auto")
domain_dd = db.add_or_get_domain_id(domain_name="drug_discovery")

metrics_lookup = {}  # metric_key_in_json -> metric_name
all_metrics_set = set()

for plan_key, record in plans.items():
    plan_name = record["TestPlan_name"]
    tp = TestPlan(plan_name=plan_name)
    metrics_list = []
    for met_id, met_name in record["metrics"].items():
        mk_lower = met_name.lower()
        if mk_lower not in all_metrics_set:
            all_metrics_set.add(mk_lower)
            metrics_lookup[met_id] = met_name
            metrics_list.append(Metric(metric_name=met_name, domain_id=domain_general))
    db.add_testplan_and_metrics(plan=tp, metrics=metrics_list)

print(f"Imported {len(metrics_lookup)} metrics across {len(plans)} test plans")

# ── Test Cases ────────────────────────────────────────────────────────────────
prompts = json.load(open(os.path.join(BASE_DIR, "data/drug_discovery_datapoints.json")))

total_cases = 0
for met_id, data in prompts.items():
    if met_id not in metrics_lookup:
        print(f"  WARN: metric {met_id} not in plans, skipping")
        continue
    parent_metric_name = metrics_lookup[met_id]
    parent_cases = []

    for case in data.get("cases", []):
        lang_id = db.add_or_get_language_id(language_name="auto")
        domain_id = db.add_or_get_domain_id(domain_name=case.get("DOMAIN", "general").lower())
        
        prompt = Prompt(
            system_prompt=case["SYSTEM_PROMPT"],
            user_prompt=case["PROMPT"],
            domain_id=domain_id,
            lang_id=lang_id,
        )
        
        judge_prompt = None
        if case.get("LLM_AS_JUDGE", "No") != "No":
            judge_prompt = LLMJudgePrompt(prompt=case["LLM_AS_JUDGE"])
        
        response = None
        if case.get("EXPECTED_OUTPUT"):
            response = Response(
                response_text=case["EXPECTED_OUTPUT"],
                response_type="GT",
                lang_id=lang_id,
            )
        
        strategy_ids = case.get("STRATEGY", ["42"])
        strategy_name = strategies_raw.get(strategy_ids[0], "similarity_match").lower()
        
        tc = TestCase(
            name=case["PROMPT_ID"],
            metric=parent_metric_name,
            prompt=prompt,
            strategy=strategy_name,
            response=response,
            judge_prompt=judge_prompt,
        )
        parent_cases.append(tc)
        total_cases += 1
    
    metric_obj = Metric(metric_name=parent_metric_name, domain_id=domain_general)
    db.add_metric_and_testcases(testcases=parent_cases, metric=metric_obj)
    print(f"  Metric '{parent_metric_name}': {len(parent_cases)} test cases")

print(f"Total test cases imported: {total_cases}")

# ── Target: Drug Discovery Assistant ─────────────────────────────────────────
tgt = Target(
    target_name="Drug-Discovery-Assistant",
    target_type="API",
    target_url="http://localhost:11434",
    target_description=(
        "Llama 3.2 via Ollama configured as a drug discovery assistant "
        "with expertise in computational chemistry, ADMET, binding affinity, "
        "and AI-driven drug development approaches."
    ),
    target_domain="drug_discovery",
    target_languages=["english"],
)
target_id = db.add_or_get_target(target=tgt)
print(f"\nTarget 'Drug-Discovery-Assistant' registered with ID: {target_id}")
print("Setup complete.")