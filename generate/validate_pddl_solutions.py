#!/usr/bin/env python3
"""
PDDL Plan Validator

Runs Fast Downward on all domain/problem pairs and validates the results
against an answer key CSV containing expected number of steps.
"""

import os
import csv
import re
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse


def check_failed_marker(file_path: Path) -> bool:
    """Check if a PDDL file is marked as FAILED TO PARSE PDDL"""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            return "FAILED TO PARSE PDDL" in first_line
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return True  # Treat as failed if we can't read it


def count_steps_in_solution(solution_file: Path) -> Optional[int]:
    """Count the number of steps in a Fast Downward solution file"""
    try:
        with open(solution_file, 'r') as f:
            lines = f.readlines()
        
        # Count non-comment, non-empty lines (excluding cost line)
        steps = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith(';') and not line.lower().startswith('cost'):
                steps += 1
        
        return steps
    except Exception as e:
        print(f"Error reading solution file {solution_file}: {e}")
        return None


def run_fastdownward(domain_file: Path, problem_file: Path, fast_downward_path: str, instruction_id: int, plans_dir: Optional[Path] = None) -> Optional[int]:
    """
    Run Fast Downward on a domain/problem pair using fast-downward.py
    Returns the number of steps in the solution, or None if no solution found
    """
    try:
        print(f"[{instruction_id}] Running Fast Downward...")
        
        # Use absolute paths for domain and problem files
        domain_abs = domain_file.resolve()
        problem_abs = problem_file.resolve()
        
        # Single command approach like in the ROS2 example
        cmd = [
            "uv",
            "run",
            fast_downward_path,
            str(domain_abs),
            str(problem_abs),
            "--search", "astar(lmcut())"
        ]
        
        print(f"[{instruction_id}] Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Look for sas_plan in the current working directory
        solution_file = Path.cwd() / "sas_plan"
        
        if not solution_file.exists():
            print(f"[{instruction_id}] No solution found (sas_plan not created)")
            if result.returncode != 0:
                print(f"[{instruction_id}] Fast Downward failed with return code {result.returncode}")
                print(f"[{instruction_id}] Error output: {result.stderr[:500]}")
            return None
        
        # Count steps in solution
        steps = count_steps_in_solution(solution_file)
        print(f"[{instruction_id}] Solution found with {steps} steps")
        
        # Save the plan file if plans_dir is specified
        if plans_dir:
            try:
                plans_dir.mkdir(parents=True, exist_ok=True)
                plan_dest = plans_dir / f"sas_plan_{instruction_id:04d}"
                shutil.copy(solution_file, plan_dest)
                print(f"[{instruction_id}] Plan saved to {plan_dest}")
            except Exception as e:
                print(f"[{instruction_id}] Warning: Could not save plan file: {e}")
        
        # Clean up the solution file for next run
        try:
            solution_file.unlink()
        except:
            pass
        
        return steps
        
    except subprocess.TimeoutExpired:
        print(f"[{instruction_id}] Timeout")
        return None
    except Exception as e:
        print(f"[{instruction_id}] Error running Fast Downward: {e}")
        return None


def load_answer_key(csv_path: Path) -> Dict[int, int]:
    """Load answer key CSV with expected number of steps for each ID"""
    answer_key = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instruction_id = int(row.get('id', 0))
            expected_steps = int(row.get('steps', 0))
            answer_key[instruction_id] = expected_steps
    
    print(f"Loaded answer key with {len(answer_key)} entries")
    return answer_key


def main():
    parser = argparse.ArgumentParser(description='Validate PDDL solutions using Fast Downward')
    parser.add_argument('--domains-dir', type=str, required=True,
                        help='Directory containing domain PDDL files')
    parser.add_argument('--problems-dir', type=str, required=True,
                        help='Directory containing problem PDDL files')
    parser.add_argument('--answer-key', type=str, required=True,
                        help='Path to answer key CSV (columns: id, steps)')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Path to output results CSV')
    parser.add_argument('--fast-downward', type=str, default='fast-downward.py',
                        help='Path to fast-downward.py script')
    parser.add_argument('--plans-dir', type=str, default='./plans',
                        help='Directory to save plan files (default: ./plans)')
    
    args = parser.parse_args()
    
    # Setup paths
    domains_dir = Path(args.domains_dir)
    problems_dir = Path(args.problems_dir)
    answer_key_path = Path(args.answer_key)
    output_csv_path = Path(args.output_csv)
    plans_dir = Path(args.plans_dir)
    
    # Resolve fast-downward path (convert relative to absolute)
    fast_downward_path = Path(args.fast_downward).resolve()
    
    # Check if fast-downward.py exists or is in PATH
    if not fast_downward_path.exists():
        # Try to find it in PATH
        fd_in_path = shutil.which('fast-downward.py')
        if fd_in_path:
            fast_downward_path = Path(fd_in_path).resolve()
        else:
            print(f"ERROR: fast-downward.py not found at {fast_downward_path} or in PATH")
            print("Please specify the path using --fast-downward argument")
            return
    
    # Convert to string for subprocess
    fast_downward_path = str(fast_downward_path)
    
    # Validate input directories
    if not domains_dir.exists():
        print(f"ERROR: Domains directory not found: {domains_dir}")
        return
    
    if not problems_dir.exists():
        print(f"ERROR: Problems directory not found: {problems_dir}")
        return
    
    if not answer_key_path.exists():
        print(f"ERROR: Answer key file not found: {answer_key_path}")
        return
    
    # Load answer key
    try:
        answer_key = load_answer_key(answer_key_path)
    except Exception as e:
        print(f"ERROR loading answer key: {e}")
        return
    
    # Find all domain/problem pairs
    domain_files = sorted(domains_dir.glob("domain_*.pddl"))
    
    print(f"\nStarting PDDL validation...")
    print(f"Found {len(domain_files)} domain files")
    print(f"Output: {output_csv_path}")
    print("=" * 60)
    
    # Results storage
    results = []
    
    for domain_file in domain_files:
        # Extract ID from filename (e.g., domain_0001.pddl -> 1)
        match = re.search(r'domain_(\d+)\.pddl', domain_file.name)
        if not match:
            continue
        
        instruction_id = int(match.group(1))
        problem_file = problems_dir / f"problem_{instruction_id:04d}.pddl"
        
        print(f"\n[{instruction_id}] Processing...")
        
        # Check if problem file exists
        if not problem_file.exists():
            print(f"[{instruction_id}] Problem file not found - False")
            results.append({'id': instruction_id, 'correct_step': False})
            continue
        
        # Check for FAILED markers
        if check_failed_marker(domain_file) or check_failed_marker(problem_file):
            print(f"[{instruction_id}] Marked as FAILED TO PARSE PDDL - False")
            results.append({'id': instruction_id, 'correct_step': False})
            continue
        
        # Run Fast Downward
        actual_steps = run_fastdownward(domain_file, problem_file, fast_downward_path, instruction_id, plans_dir)
        
        # Check if we have expected steps for this ID
        if instruction_id not in answer_key:
            print(f"[{instruction_id}] No answer key entry - False")
            results.append({'id': instruction_id, 'correct_step': False})
            continue
        
        expected_steps = answer_key[instruction_id]
        
        # Compare results
        if actual_steps is None:
            print(f"[{instruction_id}] No solution found - False")
            correct = False
        elif actual_steps == expected_steps:
            print(f"[{instruction_id}] Steps match ({actual_steps}) - True")
            correct = True
        else:
            print(f"[{instruction_id}] Steps mismatch (expected: {expected_steps}, got: {actual_steps}) - False")
            correct = False
        
        results.append({'id': instruction_id, 'correct_step': correct})
    
    # Write results to CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'correct_step'])
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Validation Complete!")
    total = len(results)
    correct = sum(1 for r in results if r['correct_step'])
    print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Incorrect: {total - correct}/{total}")
    print(f"Results saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
