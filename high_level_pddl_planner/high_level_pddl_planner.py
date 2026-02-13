"""
ROS2 High-Level Agent Node (PDDL-based) - Fixed Domain Version

Modified to use a fixed PDDL domain with predetermined actions,
while generating the problem file dynamically based on runtime state queries.
"""
import os
import re
import tempfile
import threading
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.task import Future as RclpyFuture

from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import String
from geometry_msgs.msg import Pose

from custom_interfaces.action import Prompt
from custom_interfaces.srv import (
    DetectObjects,
    ClassifyBBox,
    DetectGrasps,
    DetectGraspBBox,
    UnderstandScene,
    GetSetBool,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama


from dotenv import load_dotenv

ENV_PATH = '/home/group11/final_project_ws/src/high_level_pddl_planner/.env'
load_dotenv(dotenv_path=ENV_PATH)

FAST_DOWNWARD_PY = os.getenv("FAST_DOWNWARD_PY", "./fastdownward/fast-downward.py")
SAS_PATH_PLAN = "/home/group11/final_project_ws/src/high_level_pddl_planner/sas_plan"

# Fixed PDDL Domain
FIXED_DOMAIN = """(define (domain robot-manipulation)
  (:requirements :strips :typing)
  
  (:types
    location direction object
  )
  
  (:predicates
    (robot-at ?loc - location)
    (robot-at ?obj - object)
    (gripper-open)
    (gripper-closed)
  )
  
  (:action move_to_home
    :parameters ()
    :precondition (and)
    :effect (and
      (robot-at home)
      (not (robot-at ready))
      (not (robot-at handover))
    )
  )
  
  (:action move_to_ready
    :parameters ()
    :precondition (robot-at home)
    :effect (and
      (robot-at ready)
      (not (robot-at home))
      (not (robot-at handover))
    )
  )

  (:action move_to_handover
    :parameters ()
    :precondition (robot-at home)
    :effect (and
      (robot-at handover)
      (not (robot-at home))
      (not (robot-at ready))
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
      (not (robot-at home))
      (not (robot-at ready))
    )
  )
"""


class PDDLGenerationResult:
    def __init__(self, domain_pddl: str, problem_pddl: str, reasoning: str = ""):
        self.domain_pddl = domain_pddl
        self.problem_pddl = problem_pddl
        self.reasoning = reasoning


class PlanningResult:
    def __init__(self, status: str, return_code: int, stdout: str, stderr: str, plan: str = ""):
        self.status = status
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.plan = plan
        self.plan_length = len([l for l in plan.splitlines() if l.strip()])


class Ros2HighLevelAgentNode(Node):
    def __init__(self):
        super().__init__("ros2_high_level_agent_pddl")
        self.get_logger().info("Initializing Ros2 High-Level Agent Node (Fixed PDDL Domain)...")

        self.declare_parameter("real_hardware", False)
        self.real_hardware: bool = self.get_parameter("real_hardware").get_parameter_value().bool_value
        self.declare_parameter("use_ollama", False)
        self.use_ollama: bool = self.get_parameter("use_ollama").get_parameter_value().bool_value
        self.declare_parameter("confirm", True)
        self.confirm: bool = self.get_parameter("confirm").get_parameter_value().bool_value

        # -----------------------------
        # LLM Selection: Gemini or Ollama
        # -----------------------------
        if self.use_ollama:
            self.get_logger().info("Using local LLM via Ollama.")
            # Example: using llama3.1 or any model installed in `ollama list`
            self.llm = ChatOllama(
                model="gpt-oss:20b",   # <--- change to any local model you want
                temperature=0.0
            )
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                self.get_logger().warn("No LLM API key found in environment variables GEMINI_API_KEY.")
            self.get_logger().info("Using Google Gemini API LLM.")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.0,
            )

        # Transcript subscription
        self.transcript_sub = self.create_subscription(String, "/transcript", self.transcript_callback, 10)
        self._last_transcript_lock = threading.Lock()
        self._last_transcript: Optional[str] = None

        # Medium-level action client
        self.medium_level_client = ActionClient(self, Prompt, "/medium_level")

        # Vision service clients
        self.vision_detect_objects_client = self.create_client(DetectObjects, "/vision/detect_objects")
        self.vision_classify_all_client = self.create_client(Trigger, "/vision/classify_all")
        self.vision_classify_bb_client = self.create_client(ClassifyBBox, "/vision/classify_bb")
        self.vision_detect_grasp_client = self.create_client(DetectGrasps, "/vision/detect_grasp")
        self.vision_detect_grasp_bb_client = self.create_client(DetectGraspBBox, "/vision/detect_grasp_bb")
        self.vision_understand_scene_client = self.create_client(UnderstandScene, "/vision/understand_scene")

        # State query service clients
        self.is_home_client = self.create_client(GetSetBool, "/is_home")
        self.is_ready_client = self.create_client(GetSetBool, "/is_ready")
        self.is_handover_client = self.create_client(GetSetBool, "/is_handover")
        self.gripper_is_open_client = self.create_client(GetSetBool, "/gripper_is_open")

        self._tools_called: List[str] = []
        self._tools_called_lock = threading.Lock()

        self.tools = self._initialize_tools()
        self.agent_executor = self._create_pddl_agent_executor()

        self.chat_history: List[Dict[str, str]] = []
        self.latest_plan: Optional[List[str]] = None
        self.latest_pddl: Optional[PDDLGenerationResult] = None

        self.confirm_srv = self.create_service(Trigger, "/confirm", self.confirm_service_callback)

        self._action_server = ActionServer(
            self,
            Prompt,
            "prompt_high_level",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.response_pub = self.create_publisher(String, "/response", 10)
        self.benchmark_pub = self.create_publisher(String, "/benchmark_logs", 10)

        self.get_logger().info("Ros2 High-Level Agent Node (Fixed PDDL Domain) ready.")

    def _benchmark_log(self, label: str):
        t = self.get_clock().now()
        t_sec = t.nanoseconds * 1e-9
        self.benchmark_pub.publish(
            String(data=f"{label},{t_sec:.9f}")
        )

    # -----------------------
    # State query helpers
    # -----------------------
    def _query_state(self, client, service_name: str) -> Optional[bool]:
        """Query a state service and return the current value."""
        try:
            if not client.wait_for_service(timeout_sec=3.0):
                self.get_logger().warn(f"Service {service_name} unavailable")
                return None
            
            req = GetSetBool.Request()
            req.set = False  # Just query, don't set
            req.value = False
            
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            
            resp = future.result()
            if resp is None or not resp.success:
                self.get_logger().warn(f"Failed to query {service_name}")
                return None
            
            return resp.value
        except Exception as e:
            self.get_logger().error(f"Error querying {service_name}: {e}")
            return None

    def _get_current_state(self) -> Dict[str, bool]:
        """Query all state services and return current state."""
        state = {}
        
        is_home = self._query_state(self.is_home_client, "/is_home")
        is_ready = self._query_state(self.is_ready_client, "/is_ready")
        is_handover = self._query_state(self.is_handover_client, "/is_handover")
        gripper_open = self._query_state(self.gripper_is_open_client, "/gripper_is_open")
        
        # Default to reasonable values if services fail
        state['robot_at_home'] = is_home if is_home is not None else False
        state['robot_at_ready'] = is_ready if is_ready is not None else False
        state['robot_in_handover'] = is_handover if is_handover is not None else False
        state['gripper_open'] = gripper_open if gripper_open is not None else True
        state['gripper_closed'] = not state['gripper_open']
        
        self.get_logger().info(f"Current state: {state}")
        return state

    # -----------------------
    # Transcript handling
    # -----------------------
    def transcript_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        start_time = time.perf_counter()
        self._benchmark_log("transcript_received")

        with self._last_transcript_lock:
            self._last_transcript = text
        self.get_logger().info(f"Received transcript: {text}")
        plan_thread = threading.Thread(
            target=self._generate_plan, 
            args=(text, start_time), 
            daemon=True
        )
        plan_thread.start()

    def _generate_plan(self, instruction_text: str, start_time: Optional[float] = None) -> List[str]:
        """Generate a plan using fixed domain and dynamic problem generation."""
        self.chat_history.append({"role": "user", "content": instruction_text})

        with self._tools_called_lock:
            self._tools_called = []

        try:
            self.get_logger().info("High-level agent (PDDL): generating plan with fixed domain...")
            self.response_pub.publish(String(data="Got it! Let me think through that..."))
            self.response_pub.publish(String(data="Analyzing scene and determining current state"))

            # Get current state from services
            current_state = self._get_current_state()
            
            # Build state description for LLM
            state_description = f"""
Current Robot State:
- Robot at home: {current_state['robot_at_home']}
- Robot at ready: {current_state['robot_at_ready']}
- Gripper open: {current_state['gripper_open']}
- Gripper closed: {current_state['gripper_closed']}
"""

            # Build LangChain chat history
            langchain_history = []
            for msg in self.chat_history[:-1]:
                if msg["role"] == "user":
                    langchain_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_history.append(AIMessage(content=msg["content"]))

            # Create augmented instruction with state info
            augmented_instruction = f"{instruction_text}\n\n{state_description}"

            # Invoke agent
            agent_resp = self.agent_executor.invoke({
                "input": augmented_instruction,
                "chat_history": langchain_history
            })

            final_text = agent_resp.get("output") if isinstance(agent_resp, dict) else str(agent_resp)
            self.get_logger().info(f"Agent output (raw):\n{final_text}")

            self.chat_history.append({"role": "assistant", "content": final_text})

            self.response_pub.publish(String(data="Generating PDDL problem file"))

            # Parse DOMAIN and PROBLEM from LLM output
            pddl_gen = self._parse_domain_and_problem_from_text(final_text)
            if pddl_gen is None:
                msg = "Hmm... I couldn't generate valid PDDL files. Could you try rephrasing that?"
                self.get_logger().error("Failed to parse PDDL domain/problem from LLM output.")
                self.response_pub.publish(String(data=msg))
                return []

            # Use LLM-generated domain (which may be modified from the template)
            domain_pddl = pddl_gen.domain_pddl
            problem_pddl = pddl_gen.problem_pddl
            
            # Store PDDL for reference
            self.latest_pddl = PDDLGenerationResult(domain_pddl=domain_pddl, problem_pddl=problem_pddl)

            # Save PDDL to temporary directory
            tmpdir = tempfile.mkdtemp(prefix="pddl_")
            domain_path = Path(tmpdir) / "domain.pddl"
            problem_path = Path(tmpdir) / "problem.pddl"
            domain_path.write_text(domain_pddl)
            problem_path.write_text(problem_pddl)
            self.get_logger().info(f"PDDL files saved to {tmpdir}")
            self.get_logger().debug(f"Domain:\n{domain_pddl}")
            self.get_logger().debug(f"Problem:\n{problem_pddl}")

            # Run Fast Downward
            self.response_pub.publish(String(data="Solving the PDDL problem using Fast Downward"))
            plan_result = self._run_fast_downward(str(domain_path), str(problem_path))

            if start_time is not None:
                end_time = time.perf_counter()
                self._benchmark_log("plan_generated")
                self.benchmark_pub.publish(String(data=f"Plan generated in: ,{end_time - start_time:.2f}"))

            if plan_result.status != "success" or not plan_result.plan.strip():
                msg = f"I couldn't find a valid plan. The planner returned: {plan_result.status}"
                self.get_logger().warn(f"Planner returned no plan. status={plan_result.status} rc={plan_result.return_code}")
                self.response_pub.publish(String(data=msg))
                return []

            self.get_logger().info("Plan obtained from Fast Downward. Parsing to steps...")
            plan_lines = self._parse_plan_text(plan_result.plan)
            if not plan_lines:
                msg = "No actionable steps could be parsed from the plan."
                self.get_logger().warn("No actionable plan lines parsed.")
                self.response_pub.publish(String(data=msg))
                return []

            self.latest_plan = plan_lines

            readable_plan = "\n".join([f"{i+1}. {s}" for i, s in enumerate(plan_lines)])
            if self.confirm:
                self.response_pub.publish(String(
                    data=f"Here's what I plan to do:\n{readable_plan}\n\nPlease review and confirm if this looks good!"
                ))
                self.get_logger().info(f"Generated plan with {len(plan_lines)} steps, waiting for /confirm.")
            else:
                def execute_plan():
                    self.response_pub.publish(String(data="Got it! Executing your approved plan now..."))
                    self.get_logger().info("Executing confirmed plan...")

                    for i, step in enumerate(self.latest_plan, start=1):
                        self.response_pub.publish(String(data=f"Starting step {i}: {step}"))
                        result = self.send_step_to_medium_async(step)

                        if result is None or not result.success:
                            msg = f"Step {i} failed: {step}. Stopping execution."
                            self.response_pub.publish(String(data=msg))
                            self.get_logger().error(msg)
                            break
                        else:
                            done_msg = f"Step {i} completed successfully."
                            self.response_pub.publish(String(data=done_msg))
                            self.get_logger().info(done_msg)

                    end_time = time.perf_counter()
                    benchmark_info = f"High-level action completed in {end_time - self.start_time:.2f} seconds."
                    self.benchmark_pub.publish(String(data=benchmark_info))
                    self.response_pub.publish(String(data="Plan execution finished."))
                    self.get_logger().info("All steps done. Clearing chat history and plan.")
                    self.chat_history.clear()
                    self.latest_plan = None
                    self.latest_pddl = None

                execution_thread = threading.Thread(target=execute_plan, daemon=True)
                execution_thread.start()

            return plan_lines

        except Exception as e:
            self.get_logger().error(f"Error generating plan: {e}")
            self.response_pub.publish(String(data="Sorry, something went wrong while planning."))
            return []

    def confirm_service_callback(self, request, response):
        """Execute the latest plan when confirmed."""
        if not self.latest_plan:
            response.success = False
            response.message = "No plan to confirm. Please give a new instruction first."
            self.response_pub.publish(String(data=response.message))
            return response

        def execute_plan():
            self.response_pub.publish(String(data="Got it! Executing your approved plan now..."))
            self.get_logger().info("Executing confirmed plan...")

            for i, step in enumerate(self.latest_plan, start=1):
                self.response_pub.publish(String(data=f"Starting step {i}: {step}"))
                result = self.send_step_to_medium_async(step)

                if result is None or not result.success:
                    msg = f"Step {i} failed: {step}. Stopping execution."
                    self.response_pub.publish(String(data=msg))
                    self.get_logger().error(msg)
                    break
                else:
                    done_msg = f"Step {i} completed successfully."
                    self.response_pub.publish(String(data=done_msg))
                    self.get_logger().info(done_msg)

            self.response_pub.publish(String(data="Plan execution finished."))
            self.get_logger().info("All steps done. Clearing chat history and plan.")
            self.chat_history.clear()
            self.latest_plan = None
            self.latest_pddl = None

        execution_thread = threading.Thread(target=execute_plan, daemon=True)
        execution_thread.start()

        response.success = True
        response.message = "Plan execution started."
        return response

    # -----------------------
    # PDDL parsing helpers
    # -----------------------
    def _parse_domain_and_problem_from_text(self, text: str) -> Optional[PDDLGenerationResult]:
        """Extract DOMAIN and PROBLEM PDDL from LLM output."""
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
            domain_paren = re.search(r"\(define\s*\(domain.*?\)\)", text, re.DOTALL | re.IGNORECASE)
            problem_paren = re.search(r"\(define\s*\(problem.*?\)\)", text, re.DOTALL | re.IGNORECASE)
            if domain_paren and problem_paren:
                domain = domain_paren.group(0).strip()
                problem = problem_paren.group(0).strip()
                return PDDLGenerationResult(domain_pddl=domain, problem_pddl=problem, reasoning="")
                
        except Exception as e:
            self.get_logger().error(f"Error extracting PDDL from text: {e}")
        return None

    def _run_fast_downward(self, domain_file: str, problem_file: str, timeout: int = 300) -> PlanningResult:
        """Call Fast Downward to produce a plan."""
        workdir = str(Path(domain_file).parent)
        cmd = ["python3", FAST_DOWNWARD_PY, domain_file, problem_file, "--search", "astar(lmcut())"]
        self.get_logger().info(f"Calling Fast Downward: {' '.join(cmd)} (workdir={workdir})")
        try:
            result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)
            plan_text = ""
            sas_plan_path = Path(workdir) / "sas_plan"
            self.get_logger().info(f"Fast Downward stdout:\n{result.stdout}")
            if sas_plan_path.exists():
                plan_text = sas_plan_path.read_text()
            status = "success" if result.returncode == 0 else "failed"
            return PlanningResult(status=status, return_code=result.returncode, stdout=result.stdout, stderr=result.stderr, plan=plan_text)
        except subprocess.TimeoutExpired as e:
            return PlanningResult(status="timeout", return_code=-1, stdout=str(e.stdout or ""), stderr=str(e.stderr or ""))
        except Exception as e:
            return PlanningResult(status="error", return_code=-1, stdout="", stderr=str(e))

    def _parse_plan_text(self, plan_text: str) -> List[str]:
        """Parse plan text into textual steps."""
        lines = []
        for ln in plan_text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith(";"):
                continue
            ln = re.sub(r"\s*\[.*?\]\s*$", "", ln)
            ln = re.sub(r"\s*\(cost.*?\)\s*$", "", ln, flags=re.IGNORECASE)
            if ln.startswith("(") and ln.endswith(")"):
                ln = ln[1:-1].strip()
            ln = " ".join(ln.split())
            if ln:
                lines.append(ln)
        return lines

    # -----------------------
    # Tools
    # -----------------------
    def _initialize_tools(self) -> List[BaseTool]:
        tools: List[BaseTool] = []

        @tool
        def detect_objects(image_hint: Optional[str] = "") -> str:
            """Call /vision/detect_objects to get bounding boxes."""
            tool_name = "detect_objects"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                if not self.vision_detect_objects_client.wait_for_service(timeout_sec=5.0):
                    return "Service /vision/detect_objects unavailable"
                req = DetectObjects.Request()
                future = self.vision_detect_objects_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                resp = future.result()
                if resp is None:
                    return "No response from /vision/detect_objects"
                if not resp.success:
                    return f"detect_objects failed: {resp.error_message or 'unknown error'}"
                total = int(resp.total_detections)
                items = []
                N = min(total, 4)
                for i in range(N):
                    oid = resp.object_ids[i] if i < len(resp.object_ids) else f"obj_{i}"
                    x1 = resp.bbox_x1[i] if i < len(resp.bbox_x1) else -1
                    y1 = resp.bbox_y1[i] if i < len(resp.bbox_y1) else -1
                    x2 = resp.bbox_x2[i] if i < len(resp.bbox_x2) else -1
                    y2 = resp.bbox_y2[i] if i < len(resp.bbox_y2) else -1
                    conf = resp.confidences[i] if i < len(resp.confidences) else 0.0
                    dist = resp.distances_cm[i] if i < len(resp.distances_cm) else -1.0
                    items.append(f"{oid} bbox=[{x1},{y1},{x2},{y2}] conf={conf:.2f} dist_cm={dist:.1f}")
                summary = f"Detected {total} objects. Examples: " + "; ".join(items) if items else f"Detected {total} objects."
                return summary
            except Exception as e:
                return f"ERROR in detect_objects: {e}"

        tools.append(detect_objects)

        @tool
        def classify_all() -> str:
            """Trigger /vision/classify_all."""
            tool_name = "classify_all"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_classify_all_client.wait_for_service(timeout_sec=5.0):
                    return "Service /vision/classify_all unavailable"
                req = Trigger.Request()
                future = self.vision_classify_all_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                resp = future.result()
                if resp is None:
                    return "No response from /vision/classify_all"
                return f"classify_all: success={resp.success}, message={resp.message}"
            except Exception as e:
                return f"ERROR in classify_all: {e}"

        tools.append(classify_all)

        @tool
        def understand_scene() -> str:
            """Call /vision/understand_scene for scene understanding."""
            tool_name = "understand_scene"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_understand_scene_client.wait_for_service(timeout_sec=5.0):
                    return "Service /vision/understand_scene unavailable"
                req = UnderstandScene.Request()
                future = self.vision_understand_scene_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                resp = future.result()
                if resp is None:
                    return "No response from /vision/understand_scene"
                if not resp.success:
                    return f"understand_scene failed: {resp.error_message or 'unknown'}"
                summary = getattr(resp.scene, "scene_description", None)
                if summary:
                    return f"scene_summary: {summary}"
                total_objects = getattr(resp.scene, "total_objects", None)
                labels = getattr(resp.scene, "object_labels", None)
                return f"scene_summary: total_objects={total_objects}, labels={labels}"
            except Exception as e:
                return f"ERROR in understand_scene: {e}"

        tools.append(understand_scene)

        return tools

    # -----------------------
    # Create PDDL agent
    # -----------------------
    def _create_pddl_agent_executor(self) -> AgentExecutor:
        """Create an agent that generates PDDL domain and problem files."""
        system_message = f"""You are a PDDL domain and problem generator for a robot planning system.

        You have access to vision tools to understand the scene (detect_objects, classify_all, understand_scene).
        The current robot state will be provided in the user message.

        Below is a TEMPLATE DOMAIN with predefined actions. You should use this as a starting point:

        {FIXED_DOMAIN}

        Your task is to:
        1. Review the template domain above
        2. Modify it (e.g., if you need additional predicates, types, or action parameters)
        3. Generate a corresponding PROBLEM file with:
        - Object definitions (include directions: left, right, up, down, forward, backward as direction objects)
        - Initial state based on the provided current robot state
        - Goal state that achieves the user's instruction

        IMPORTANT GUIDELINES:
        - Prefer using a modified domain whenever possible
        - You can create new actions if necessary
        - The robot can move to a given object, generate move_to_<object> actions as needed
        - DO NOT modify the actions name or parameters
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

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=12)

    # -----------------------
    # Action server callbacks
    # -----------------------
    def goal_callback(self, goal_request) -> GoalResponse:
        self.get_logger().info(f"[high-level action] Received goal: {getattr(goal_request, 'prompt', '')}")
        with self._tools_called_lock:
            self._tools_called = []
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("[high-level action] Cancel request received.")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute incoming high-level Prompt action."""
        self.start_time = time.perf_counter()
        self._benchmark_log("action_goal_received")

        prompt_text = goal_handle.request.prompt
        self.get_logger().info(f"[high-level action] Executing prompt: {prompt_text}")

        feedback_msg = Prompt.Feedback()
        result_container: Dict[str, Any] = {"success": False, "final_response": ""}

        def run_pipeline():
            try:
                goal_text = goal_handle.request.prompt.strip()
                if not goal_text:
                    result_container["success"] = False
                    result_container["final_response"] = "Empty prompt"
                    return

                self.get_logger().info(f"High-level Prompt action received: {goal_text}")

                steps = self._generate_plan(goal_text, start_time=self.start_time)
                if not steps:
                    result_container["success"] = False
                    result_container["final_response"] = "Failed to generate plan"
                    return

                msg = f"Generated {len(steps)} step(s). Please review and confirm via /confirm to execute."
                result_container["success"] = True
                result_container["final_response"] = msg
            except Exception as e:
                self.get_logger().error(f"Exception in action pipeline: {e}")
                result_container["success"] = False
                result_container["final_response"] = f"Error: {e}"

        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()

        while thread.is_alive():
            with self._tools_called_lock:
                tools_snapshot = list(self._tools_called)
            feedback_msg.tools_called = tools_snapshot
            try:
                goal_handle.publish_feedback(feedback_msg)
            except Exception:
                pass
            time.sleep(0.5)

        with self._tools_called_lock:
            tools_snapshot = list(self._tools_called)
        feedback_msg.tools_called = tools_snapshot
        try:
            goal_handle.publish_feedback(feedback_msg)
        except Exception:
            pass

        result_msg = Prompt.Result()
        result_msg.success = bool(result_container.get("success", False))
        result_msg.final_response = str(result_container.get("final_response", ""))

        goal_handle.succeed()
        self.get_logger().info(f"[high-level action] Goal finished. success={result_msg.success}")
        return result_msg

    # -----------------------
    # Medium-level communication
    # -----------------------
    def send_step_to_medium_async(self, step_text: str, timeout: float = 60.0) -> Optional[Prompt.Result]:
        """Send a step to medium-level action server."""
        try:
            if not self.medium_level_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("/medium_level action server unavailable")
                return None
            
            goal = Prompt.Goal()
            goal.prompt = step_text
            
            goal_event = threading.Event()
            result_event = threading.Event()
            goal_handle_container = [None]
            result_container = [None]
            
            def goal_response_callback(future):
                goal_handle_container[0] = future.result()
                goal_event.set()
            
            def result_callback(future):
                result_container[0] = future.result()
                result_event.set()
            
            send_future = self.medium_level_client.send_goal_async(goal)
            send_future.add_done_callback(goal_response_callback)
            
            if not goal_event.wait(timeout=5.0):
                self.get_logger().error("Timeout waiting for goal acceptance")
                return None
            
            goal_handle = goal_handle_container[0]
            if not goal_handle.accepted:
                self.get_logger().error("Medium-level goal rejected")
                return None
            
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(result_callback)
            
            if not result_event.wait(timeout=timeout):
                self.get_logger().error("Timeout waiting for result")
                return None
            
            result = result_container[0].result
            return result
            
        except Exception as e:
            self.get_logger().error(f"Exception when sending to medium: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = Ros2HighLevelAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Ros2 High-Level Agent Node (Fixed PDDL Domain).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()