# Usage

```bash
uv run .\validate_pddl_solutions.py \
  --domains-dir .\domains\ \
  --problems-dir .\problems\ \
  --answer-key .\answer_key\answer_key_1.csv \
  --output-csv .\results\result_1.csv \
  --fast-downward .\fast-downward\fast-downward.py \
  --plans-dir .\plans
```