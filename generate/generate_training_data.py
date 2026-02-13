#!/usr/bin/env python3
"""
PDDL Training Data Generator

Generates PDDL domain and problem files from natural language instructions
using Google Gemini API with multiple API key rotation to handle rate limits.
"""

import os
import csv
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


# Available Gemini models to rotate through
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


# Fixed PDDL Domain Template (same as in high_level_pddl_planner)
FIXED_DOMAIN = """
(define (domain robot-manipulation)
  (:requirements :strips :typing :conditional-effects)
  
  (:types
    location direction object
  )
  
  (:predicates
    (robot-at-location ?loc - location)
    (robot-at-object ?obj - object)
    (robot-have ?obj - object)
    (gripper-open)
    (gripper-closed)
  )
  
  (:action move_to_home
    :parameters ()
    :precondition (and)
    :effect (and
      (robot-at-location home)
      (not (robot-at-location ready))
      (not (robot-at-location handover))
      (forall (?obj - object) (not (robot-at-object ?obj)))
    )
  )
  
  (:action move_to_ready
    :parameters ()
    :precondition (and)
    :effect (and
      (robot-at-location ready)
      (not (robot-at-location home))
      (not (robot-at-location handover))
      (forall (?obj - object) (not (robot-at-object ?obj)))
    )
  )

  (:action move_to_handover
    :parameters ()
    :precondition (and)
    :effect (and
      (robot-at-location handover)
      (not (robot-at-location home))
      (not (robot-at-location ready))
      (forall (?obj - object) (not (robot-at-object ?obj)))
    )
  )
  
  (:action open_gripper
    :parameters ()
    :precondition (gripper-closed)
    :effect (and
      (gripper-open)
      (not (gripper-closed))
    )
  )
  
  (:action close_gripper
    :parameters ()
    :precondition (gripper-open)
    :effect (and
      (gripper-closed)
      (not (gripper-open))
    )
  )
  
  (:action move_to_direction
    :parameters (?dir - direction)
    :precondition (and)
    :effect (and
      (not (robot-at-location ready))
      (not (robot-at-location handover))
      (forall (?obj - object) (not (robot-at-object ?obj)))
      (not (robot-at-location home))
      (not (robot-at-location ready))
    )
  )

  (:action move_to_object
    :parameters (?obj - object)
    :precondition (and
      (robot-at-location ready)
    )
    :effect (and
      (robot-at-object ?obj)
      (not (robot-at-location home))
      (not (robot-at-location ready))
      (not (robot-at-location handover))
    )
  )

  (:action pick-object
    :parameters (?obj - object)
    :precondition (and
      (robot-at-location ready)
    )
    :effect (and
      (robot-at-object ?obj)
      (robot-have ?obj)
      (not (robot-at-location ready))
      (not (robot-at-location handover))
      (forall (?obj2 - object) (when (not (= ?obj ?obj2)) (not (robot-at-object ?obj2))))
      (not (robot-at-location home))
      (not (robot-at-location ready))
      (not (robot-at-location handover))
    )
  )
  
)
"""


class PDDLGenerationResult:
    """Container for PDDL generation results"""
    def __init__(self, domain_pddl: str, problem_pddl: str, reasoning: str = ""):
        self.domain_pddl = domain_pddl
        self.problem_pddl = problem_pddl
        self.reasoning = reasoning


class APIKeyManager:
    """Manages multiple API keys with rotation and rate limit handling"""
    
    def __init__(self, api_keys: List[str], models: List[str] = None):
        self.api_keys = api_keys
        self.models = models if models else AVAILABLE_MODELS.copy()
        self.usable_models = self.models.copy()  # Track which models are usable (use list for consistent ordering)
        self.current_key_index = 0
        self.current_model_index = 0
        self.rate_limit_count = {key: 0 for key in api_keys}
        self.cooldown_until = {key: 0 for key in api_keys}
        self.model_errors = {model: 0 for model in self.models}
        
        # Track which (key, model) combinations are rate-limited
        self.key_model_cooldown = {key: {} for key in api_keys}  # {key: {model: cooldown_time}}
        
    def get_current_key(self) -> str:
        """Get the current API key"""
        return self.api_keys[self.current_key_index]
    
    def get_current_model(self) -> Optional[str]:
        """Get the current model (only from usable models)"""
        if not self.usable_models:
            return None
        
        # Ensure current index is within bounds
        self.current_model_index = self.current_model_index % len(self.usable_models)
        return self.usable_models[self.current_model_index]
    
    def rotate_key(self):
        """Rotate to the next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"Rotated to API key #{self.current_key_index + 1}")
    
    def rotate_model(self):
        """Rotate to the next usable model"""
        if len(self.usable_models) <= 1:
            return
        
        self.current_model_index = (self.current_model_index + 1) % len(self.usable_models)
        print(f"Rotated to model: {self.usable_models[self.current_model_index]}")
    
    def mark_model_unusable(self, model: str, error_msg: str = ""):
        """Mark a model as unusable and remove it from rotation"""
        if model in self.usable_models:
            removed_index = self.usable_models.index(model)
            self.usable_models.remove(model)
            self.model_errors[model] += 1
            print(f"⚠️  Model '{model}' marked as UNUSABLE. Reason: {error_msg}")
            print(f"   Remaining usable models: {len(self.usable_models)}/{len(self.models)}")
            
            # Adjust current_model_index if necessary
            if self.usable_models:
                # If we removed a model before the current index, decrement the index
                if removed_index < self.current_model_index:
                    self.current_model_index -= 1
                # If we removed the current model, keep the index (which now points to next model)
                elif removed_index == self.current_model_index:
                    # Wrap around if we're at the end
                    if self.current_model_index >= len(self.usable_models):
                        self.current_model_index = 0
    
    def mark_rate_limited(self, api_key: str, cooldown_seconds: int = 60):
        """Mark an API key as rate limited"""
        self.rate_limit_count[api_key] += 1
        self.cooldown_until[api_key] = time.time() + cooldown_seconds
        print(f"API key marked as rate limited. Cooldown: {cooldown_seconds}s")
    
    def mark_key_model_rate_limited(self, api_key: str, model: str, cooldown_seconds: int = 200):
        """Mark a specific (key, model) combination as rate limited"""
        if api_key not in self.key_model_cooldown:
            self.key_model_cooldown[api_key] = {}
        self.key_model_cooldown[api_key][model] = time.time() + cooldown_seconds
        self.rate_limit_count[api_key] += 1
        key_num = self.api_keys.index(api_key) + 1
        print(f"API key #{key_num} + model '{model}' marked as rate limited. Cooldown: {cooldown_seconds}s")
    
    def is_model_available_for_key(self, api_key: str, model: str) -> bool:
        """Check if a model is available for a specific API key (not rate limited)"""
        if model not in self.usable_models:
            return False
        if api_key not in self.key_model_cooldown:
            return True
        if model not in self.key_model_cooldown[api_key]:
            return True
        return time.time() >= self.key_model_cooldown[api_key][model]
    
    def get_next_available_model_for_current_key(self) -> Optional[str]:
        """Get next available model for the current API key"""
        current_key = self.get_current_key()
        
        if not self.usable_models:
            return None
        
        # Try all models starting from next index
        for i in range(len(self.usable_models)):
            idx = (self.current_model_index + 1 + i) % len(self.usable_models)
            model = self.usable_models[idx]
            if self.is_model_available_for_key(current_key, model):
                self.current_model_index = idx
                return model
        
        return None
    
    def all_models_exhausted_for_current_key(self) -> bool:
        """Check if all models are exhausted (rate limited) for current API key"""
        current_key = self.get_current_key()
        for model in self.usable_models:
            if self.is_model_available_for_key(current_key, model):
                return False
        return True
    
    def has_available_models(self, api_key: str) -> bool:
        """Check if an API key has any available models (not rate limited)"""
        for model in self.usable_models:
            if self.is_model_available_for_key(api_key, model):
                return True
        return False
    
    def is_available(self, api_key: str) -> bool:
        """Check if an API key is available (not in cooldown)"""
        return time.time() >= self.cooldown_until[api_key]
    
    def get_next_available_key(self) -> Optional[str]:
        """Get the next available API key that has at least one available model"""
        for _ in range(len(self.api_keys)):
            key = self.get_current_key()
            # Check both cooldown and model availability
            if self.is_available(key) and self.has_available_models(key):
                return key
            self.rotate_key()
        return None


class PDDLGenerator:
    """Generates PDDL files from natural language instructions"""
    
    # Default fallback problem when parsing fails
    FALLBACK_PROBLEM = """(define (problem move-open-home)
  (:domain robot-manipulation)
  
  (:objects
    home ready handover - location
    left right up down forward backward - direction
  )
  
  (:init
    (robot-at home)
    (gripper-open)
    ; (not (robot-at ready)) is implicit
    ; (not (robot-at handover)) is implicit
    ; (not (gripper-closed)) is implicit
  )
  
  (:goal
    (and
      (robot-at home)
      (gripper-open)
    )
  )
)"""
    
    def __init__(self, api_key_manager: APIKeyManager, output_dir: Path):
        self.api_key_manager = api_key_manager
        self.output_dir = output_dir
        self.domains_dir = output_dir / "domains"
        self.problems_dir = output_dir / "problems"
        
        # Create output directories
        self.domains_dir.mkdir(parents=True, exist_ok=True)
        self.problems_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = None
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the LLM with current API key and model"""
        api_key = self.api_key_manager.get_current_key()
        model = self.api_key_manager.get_current_model()
        
        if not model:
            raise ValueError("No usable models available!")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.0,
        )
        self.current_model = model
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the PDDL generation prompt (same as in high_level_pddl_planner)"""
        system_message = f"""You are a PDDL domain and problem generator for a robot planning system.

Below is a TEMPLATE DOMAIN with predefined actions. You should use this as a starting point:

{FIXED_DOMAIN}

Your task is to:
1. Review the template domain above
2. Modify it (e.g., if you need additional predicates, types, or action parameters)
3. Generate a corresponding PROBLEM file with:
   - Object definitions (include directions: left, right, up, down, forward, backward as direction objects)
   - Initial state (assume robot starts at home with gripper open)
   - Goal state that achieves the user's instruction

IMPORTANT GUIDELINES:
- Prefer using a modified domain whenever possible
- You can create new actions if necessary
- The robot can move to a given object, generate move_to_<object> actions as needed
- DO NOT modify the actions name or parameters unless absolutely necessary
- Always ensure domain and problem are compatible

Follow this format exactly:
REASONING:
[Explain your approach and any domain modifications]

DOMAIN:
```pddl
[domain content - use template or modified version]
```

PROBLEM:
```pddl
[problem content]
```
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}"),
        ])
    
    def _parse_pddl_from_text(self, text: str) -> Optional[PDDLGenerationResult]:
        """Extract DOMAIN and PROBLEM PDDL from LLM output"""
        try:
            # Try code fence format first
            domain_match = re.search(r"DOMAIN:\s*```pddl\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            problem_match = re.search(r"PROBLEM:\s*```pddl\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            
            if domain_match and problem_match:
                domain = domain_match.group(1).strip()
                problem = problem_match.group(1).strip()
                reasoning_match = re.search(r"REASONING:\s*(.*?)(?=DOMAIN:|PROBLEM:|$)", text, re.DOTALL | re.IGNORECASE)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                return PDDLGenerationResult(domain_pddl=domain, problem_pddl=problem, reasoning=reasoning)

            # Try simple split
            if "DOMAIN:" in text and "PROBLEM:" in text:
                domain_part = text.split("DOMAIN:")[1].split("PROBLEM:")[0].strip()
                problem_part = text.split("PROBLEM:")[1].strip()
                domain = domain_part.strip("` \n")
                problem = problem_part.strip("` \n")
                reasoning_match = re.search(r"REASONING:\s*(.*?)(?=DOMAIN:|PROBLEM:|$)", text, re.DOTALL | re.IGNORECASE)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                return PDDLGenerationResult(domain_pddl=domain, problem_pddl=problem, reasoning=reasoning)

            # Try finding define blocks
            domain_paren = re.search(r"\(define\s*\(domain.*?\)\s*\)", text, re.DOTALL | re.IGNORECASE)
            problem_paren = re.search(r"\(define\s*\(problem.*?\)\s*\)", text, re.DOTALL | re.IGNORECASE)
            if domain_paren and problem_paren:
                domain = domain_paren.group(0).strip()
                problem = problem_paren.group(0).strip()
                return PDDLGenerationResult(domain_pddl=domain, problem_pddl=problem, reasoning="")
                
        except Exception as e:
            print(f"Error extracting PDDL from text: {e}")
        return None
    
    def _save_fallback_pddl(self, instruction_id: int):
        """Save fallback PDDL files with FAILED marker when parsing fails"""
        domain_file = self.domains_dir / f"domain_{instruction_id:04d}.pddl"
        problem_file = self.problems_dir / f"problem_{instruction_id:04d}.pddl"
        
        # Add marker to first line
        domain_content = "; FAILED TO PARSE PDDL\n" + FIXED_DOMAIN
        problem_content = "; FAILED TO PARSE PDDL\n" + self.FALLBACK_PROBLEM
        
        domain_file.write_text(domain_content)
        problem_file.write_text(problem_content)
        
        print(f"[{instruction_id}] ⚠️  Saved fallback PDDL files with FAILED marker")
    
    def generate_pddl(self, instruction: str, instruction_id: int, max_retries: int = 3) -> bool:
        """Generate PDDL files for a single instruction"""
        print(f"\n[{instruction_id}] Processing: {instruction}")
        
        # Add default state information
        state_description = """
Current Robot State:
- Robot at home: True
- Robot at ready: False
- Gripper open: True
- Gripper closed: False
"""
        augmented_instruction = f"{instruction}\n\n{state_description}"
        
        prompt = self._create_prompt()
        
        for attempt in range(max_retries):
            try:
                # Check if we need to wait for cooldown
                available_key = self.api_key_manager.get_next_available_key()
                if not available_key:
                    print(f"All API keys are rate limited. Waiting 60 seconds...")
                    time.sleep(60)
                    available_key = self.api_key_manager.get_next_available_key()
                
                if not available_key:
                    print(f"ERROR: No API keys available after waiting")
                    return False
                
                current_model = self.api_key_manager.get_current_model()
                if not current_model:
                    print(f"ERROR: No usable models available")
                    return False
                
                # Re-initialize LLM if key or model changed
                if (self.llm.google_api_key != available_key or 
                    self.current_model != current_model):
                    self._initialize_llm()
                
                print(f"[{instruction_id}] Using model: {self.current_model}")
                
                # Invoke LLM
                messages = prompt.format_messages(input=augmented_instruction)
                response = self.llm.invoke(messages)
                output_text = response.content
                
                # Parse PDDL from output
                pddl_result = self._parse_pddl_from_text(output_text)
                
                if pddl_result is None:
                    print(f"[{instruction_id}] Attempt {attempt + 1}: Failed to parse PDDL")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        # After max retries, save fallback files
                        self._save_fallback_pddl(instruction_id)
                        return False
                
                # Save PDDL files
                domain_file = self.domains_dir / f"domain_{instruction_id:04d}.pddl"
                problem_file = self.problems_dir / f"problem_{instruction_id:04d}.pddl"
                
                domain_file.write_text(pddl_result.domain_pddl)
                problem_file.write_text(pddl_result.problem_pddl)
                
                print(f"[{instruction_id}] ✓ Successfully generated PDDL files")
                
                # Keep using same key and model until rate limited
                # (removed automatic rotation)
                
                return True
                
            except Exception as e:
                error_msg = str(e)
                
                # Check for model-specific errors (content generation, policy violations, etc.)
                if any(keyword in error_msg.lower() for keyword in [
                    "content generation", "policy", "safety", "blocked",
                    "not supported", "invalid model", "model not found"
                ]):
                    print(f"[{instruction_id}] Model error detected: {error_msg}")
                    current_model = self.api_key_manager.get_current_model()
                    if current_model:
                        self.api_key_manager.mark_model_unusable(current_model, error_msg[:100])
                        # Try with next model
                        if attempt < max_retries - 1:
                            self.api_key_manager.rotate_model()
                            time.sleep(2)
                            continue
                
                # Check for rate limit errors
                elif "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    print(f"[{instruction_id}] Rate limit hit.")
                    current_key = self.api_key_manager.get_current_key()
                    current_model = self.api_key_manager.get_current_model()
                    
                    # Mark this (key, model) combination as rate limited
                    if current_model:
                        self.api_key_manager.mark_key_model_rate_limited(current_key, current_model, cooldown_seconds=200)
                    
                    # Try next available model with same key
                    next_model = self.api_key_manager.get_next_available_model_for_current_key()
                    
                    if next_model:
                        print(f"[{instruction_id}] Switching to next model: {next_model} (keeping same API key)")
                    elif self.api_key_manager.all_models_exhausted_for_current_key():
                        # All models exhausted for this key, move to next key
                        print(f"[{instruction_id}] All models exhausted for current API key. Moving to next API key...")
                        self.api_key_manager.rotate_key()
                        self.api_key_manager.current_model_index = 0  # Reset to first model
                    else:
                        # Some models available but in cooldown, wait a bit
                        print(f"[{instruction_id}] Waiting for model cooldown...")
                        time.sleep(30)
                    
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                else:
                    print(f"[{instruction_id}] Error: {error_msg}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
        
        return False


def load_api_keys(csv_path: Path) -> List[str]:
    """Load API keys from CSV file"""
    api_keys = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Assume CSV has a column named 'api_key'
            key = row.get('api_key', '').strip()
            if key:
                api_keys.append(key)
    
    if not api_keys:
        raise ValueError(f"No API keys found in {csv_path}")
    
    print(f"Loaded {len(api_keys)} API keys")
    return api_keys


def load_instructions(csv_path: Path) -> List[Tuple[int, str]]:
    """Load instructions from CSV file"""
    instructions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Assume CSV has columns 'id' and 'instruction'
            instruction_id = int(row.get('id', 0))
            instruction = row.get('instruction', '').strip()
            if instruction:
                instructions.append((instruction_id, instruction))
    
    print(f"Loaded {len(instructions)} instructions")
    return instructions


def main():
    parser = argparse.ArgumentParser(description='Generate PDDL training data from natural language instructions')
    parser.add_argument('--instructions', type=str, required=True,
                        help='Path to CSV file containing instructions (columns: id, instruction)')
    parser.add_argument('--api-keys', type=str, required=True,
                        help='Path to CSV file containing API keys (column: api_key)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for generated PDDL files (default: current directory)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between API calls in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    # Setup paths
    instructions_path = Path(args.instructions)
    api_keys_path = Path(args.api_keys)
    output_dir = Path(args.output_dir)
    
    # Validate input files
    if not instructions_path.exists():
        print(f"ERROR: Instructions file not found: {instructions_path}")
        return
    
    if not api_keys_path.exists():
        print(f"ERROR: API keys file not found: {api_keys_path}")
        return
    
    # Load data
    try:
        api_keys = load_api_keys(api_keys_path)
        instructions = load_instructions(instructions_path)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return
    
    # Initialize generator
    api_key_manager = APIKeyManager(api_keys)
    generator = PDDLGenerator(api_key_manager, output_dir)
    
    # Generate PDDL files
    print(f"\nStarting PDDL generation for {len(instructions)} instructions...")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Domains: {generator.domains_dir}")
    print(f"Problems: {generator.problems_dir}")
    print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for instruction_id, instruction in instructions:
        success = generator.generate_pddl(instruction, instruction_id)
        
        if success:
            successful += 1
        else:
            failed += 1
            print(f"[{instruction_id}] ✗ Failed to generate PDDL")
        
        # Delay between requests
        time.sleep(args.delay)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"PDDL Generation Complete!")
    print(f"Successful: {successful}/{len(instructions)}")
    print(f"Failed: {failed}/{len(instructions)}")
    print(f"Output: {output_dir.absolute()}")
    print("\nModel Statistics:")
    print(f"  Usable models: {len(api_key_manager.usable_models)}/{len(AVAILABLE_MODELS)}")
    if api_key_manager.usable_models:
        print(f"  Active models: {', '.join(sorted(api_key_manager.usable_models))}")
    unusable = set(AVAILABLE_MODELS) - api_key_manager.usable_models
    if unusable:
        print(f"  Unusable models: {', '.join(sorted(unusable))}")


if __name__ == "__main__":
    main()
