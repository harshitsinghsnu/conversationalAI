"""
One-time fix: update DD018 and DD019 from llm_judge_negative -> llm_judge_positive
in the SQLite DB so the analyzer uses the correct (non-inverted) scoring.
"""
import sqlite3, os, sys

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "AIEvaluationData.db")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Get strategy IDs
cur.execute("SELECT strategy_id, strategy_name FROM Strategies WHERE strategy_name IN ('llm_judge_positive', 'llm_judge_negative')")
rows = cur.fetchall()
strat = {name: sid for sid, name in rows}
print(f"Strategies found: {strat}")

pos_id = strat.get("llm_judge_positive")
neg_id = strat.get("llm_judge_negative")
if not pos_id or not neg_id:
    print("ERROR: could not find required strategy IDs"); sys.exit(1)

# Check current state for DD018 and DD019
cur.execute("SELECT testcase_id, testcase_name, strategy_id FROM TestCases WHERE testcase_name IN ('DD018', 'DD019')")
before = cur.fetchall()
print(f"Before: {before}")

# Update
cur.execute(
    "UPDATE TestCases SET strategy_id = ? WHERE testcase_name IN ('DD018', 'DD019') AND strategy_id = ?",
    (pos_id, neg_id)
)
print(f"Rows updated: {cur.rowcount}")
conn.commit()

cur.execute("SELECT testcase_id, testcase_name, strategy_id FROM TestCases WHERE testcase_name IN ('DD018', 'DD019')")
after = cur.fetchall()
print(f"After:  {after}")
conn.close()
print("Done. Re-run the analyzer with --force for smolder-bordeaux-berry-users to re-score DD018/DD019.")
