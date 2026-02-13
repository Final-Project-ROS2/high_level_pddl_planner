# PDDL Training Data Generator

This script generates PDDL domain and problem files from natural language instructions using Google Gemini API with automatic API key rotation to handle rate limits.

## Features

- Uses the same prompt and domain template as `high_level_pddl_planner`
- Automatic API key rotation when rate limits are hit
- **Automatic model rotation** across multiple Gemini models to distribute load
- **Smart error handling** - automatically disables models that fail (policy violations, content generation errors, etc.)
- Configurable delay between API calls
- Saves numbered domain and problem files matching instruction IDs
- Retry logic for failed generations
- Progress tracking and summary statistics

## Available Models

The script rotates through these Gemini models:
- `gemini-2.0-flash-exp`
- `gemini-2.0-flash-lite`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-3`
- `gemini-3-flash`

Models that encounter errors (policy violations, content generation failures, etc.) are automatically marked as unusable and removed from rotation.

## Setup

1. Create your input CSV files (see examples below)
2. Install required dependencies:
   ```bash
   pip install langchain-google-genai langchain-core
   ```

## CSV File Formats

### Instructions CSV (`instructions.csv`)
Must have columns: `id`, `instruction`

Example:
```csv
id,instruction
1,Move the robot to the ready position
2,Pick up the red cup from the table
3,Place the object at the handover location
```

### API Keys CSV (`api_keys.csv`)
Must have column: `api_key`

Example:
```csv
api_key
YOUR_GEMINI_API_KEY_1
YOUR_GEMINI_API_KEY_2
YOUR_GEMINI_API_KEY_3
```

## Usage

Basic usage:
```bash
python generate_training_data.py \
    --instructions instructions.csv \
    --api-keys api_keys.csv \
    --output-dir .
```

With custom delay:
```bash
python generate_training_data.py \
    --instructions instructions.csv \
    --api-keys api_keys.csv \
    --output-dir . \
    --delay 2.0
```

## Command-Line Arguments

- `--instructions`: Path to CSV file containing instructions (required)
- `--api-keys`: Path to CSV file containing API keys (required)
- `--output-dir`: Output directory for generated PDDL files (default: current directory)
- `--delay`: Delay between API calls in seconds (default: 1.0)

## Output

The script creates two directories in the output directory:
- `domains/`: Contains generated domain files named `domain_XXXX.pddl`
- `problems/`: Contains generated problem files named `problem_XXXX.pddl`

Where `XXXX` is the zero-padded instruction ID (e.g., `domain_0001.pddl`)

## Rate Limit Handling

When a rate limit is detected:
1. The current API key is marked as rate-limited with a 60-second cooldown
2. The script automatically rotates to the next available API key
3. If all keys are rate-limited, it waits 60 seconds before retrying
4. Failed generations are retried up to 3 times

## Model Error Handling

When a model encounters errors (content generation failures, policy violations, unsupported operations):
1. The model is automatically marked as unusable
2. The script rotates to the next available model
3. Unusable models are removed from rotation permanently for the session
4. The script continues with remaining usable models

This ensures the script can handle different model capabilities and restrictions gracefully.

## Example

```bash
# Run with example files
python generate_training_data.py \
    --instructions instructions_example.csv \
    --api-keys api_keys_example.csv \
    --output-dir . \
    --delay 1.5

# Output:
# Loaded 3 API keys
# Loaded 5 instructions
# 
# Starting PDDL generation for 5 instructions...
# Output directory: /path/to/generate
# Domains: /path/to/generate/domains
# Problems: /path/to/generate/problems
# ============================================================
# 
# [1] Processing: Move the robot to the ready position
# [1] Using model: gemini-2.0-flash-exp
# [1] ✓ Successfully generated PDDL files
# Rotated to model: gemini-2.0-flash-lite
# ...
# ⚠️  Model 'gemini-3' marked as UNUSABLE. Reason: Content generation blocked by policy
#    Remaining usable models: 5/6
# ...
# ============================================================
# PDDL Generation Complete!
# Successful: 5/5
# Failed: 0/5
# 
# Model Statistics:
#   Usable models: 5/6
#   Active models: gemini-2.0-flash-exp, gemini-2.0-flash-lite, gemini-2.5-flash, ...
#   Unusable models: gemini-3
```

## Notes

- The script uses the same PDDL domain template and prompt as defined in `high_level_pddl_planner.py`
- Default initial state assumes: robot at home, gripper open
- Generated files are numbered to match instruction IDs for easy pairing
- Progress is printed to console with status indicators (✓ for success, ✗ for failure)
