"""
ROS2 High-Level Agent Node (PDDL-based) - Fixed Domain Version

Modified to use a fixed PDDL domain with predetermined actions,
while generating the problem file dynamically based on runtime state queries.
"""
import os
import re
import json
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

from custom_interfaces.action import Prompt, PromptScene
from custom_interfaces.srv import (
    DetectObjects,
    ClassifyBBox,
    DetectGrasps,
    DetectGraspBBox,
    UnderstandScene,
    GetSetBool,
)

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama


from dotenv import load_dotenv


# ==================== Structured Output Models ====================
class Predicate(BaseModel):
    """Represents a single PDDL predicate with arguments."""
    predicate: str = Field(description="Name of the predicate")
    args: List[str] = Field(default_factory=list, description="Arguments to the predicate")


class DefaultDomainOutput(BaseModel):
    """Structured output for the default domain."""
    locations: List[str] = Field(default_factory=list, description="Task-specific locations needed")
    objects: List[str] = Field(default_factory=list, description="Objects present in the workspace")
    goals: List[Predicate] = Field(default_factory=list, description="Desired final state predicates")


class DefaultDomainData(BaseModel):
    """Wrapper for default domain output."""
    data: DefaultDomainOutput


class BlocksworldDomainOutput(BaseModel):
    """Structured output for the blocksworld domain."""
    objects: List[str] = Field(description="Objects (blocks) present in the workspace")
    init: List[Predicate] = Field(default_factory=list, description="Current state predicates")
    goals: List[Predicate] = Field(description="Desired final state predicates")


class BlocksworldDomainData(BaseModel):
    """Wrapper for blocksworld domain output."""
    data: BlocksworldDomainOutput


class GripperDomainOutput(BaseModel):
    """Structured output for the gripper domain."""
    objects: List[str] = Field(description="Objects (rooms, balls, grippers)")
    init: List[Predicate] = Field(default_factory=list, description="Current state predicates")
    goals: List[Predicate] = Field(description="Desired final state predicates")


class GripperDomainData(BaseModel):
    """Wrapper for gripper domain output."""
    data: GripperDomainOutput


# ===================================================================

PROJECT_ROOT = Path("/home/group11/final_project_ws/src/high_level_pddl_planner")

ENV_PATH = Path(os.getenv("ENV_PATH", str(PROJECT_ROOT / ".env")))
load_dotenv(dotenv_path=ENV_PATH)

FAST_DOWNWARD_PY = os.getenv("FAST_DOWNWARD_PY", str(PROJECT_ROOT / "fast-downward" / "fast-downward.py"))
TRANSFORM_PY = os.getenv("TRANSFORM_PY", str(PROJECT_ROOT / "transform" / "transform.py"))
TRANSFORM_BLOCKSWORLD_PY = os.getenv("TRANSFORM_BLOCKSWORLD_PY", str(PROJECT_ROOT / "transform" / "transform_blocksworld.py"))
TRANSFORM_GRIPPER_PY = os.getenv("TRANSFORM_GRIPPER_PY", str(PROJECT_ROOT / "transform" / "transform_gripper.py"))

TEMPLATE_PROBLEM = os.getenv("TEMPLATE_PROBLEM", str(PROJECT_ROOT / "problems" / "template_problem.pddl"))
TEMPLATE_PROBLEM_BLOCKSWORLD = os.getenv(
    "TEMPLATE_PROBLEM_BLOCKSWORLD",
    str(PROJECT_ROOT / "problems" / "template_problem_blocksworld.pddl"),
)
TEMPLATE_PROBLEM_GRIPPER = os.getenv(
    "TEMPLATE_PROBLEM_GRIPPER",
    str(PROJECT_ROOT / "problems" / "template_problem_gripper.pddl"),
)

DOMAIN_PDDL = os.getenv("DOMAIN_PDDL", str(PROJECT_ROOT / "domains" / "domain.pddl"))
DOMAIN_PDDL_BLOCKSWORLD = os.getenv("DOMAIN_PDDL_BLOCKSWORLD", str(PROJECT_ROOT / "domains" / "domain_blocksworld.pddl"))
DOMAIN_PDDL_GRIPPER = os.getenv("DOMAIN_PDDL_GRIPPER", str(PROJECT_ROOT / "domains" / "domain_gripper.pddl"))

SCENE_DESC_MODES = {"default", "custom", "disabled"}
DOMAIN_MODES = {"default", "blocksworld", "gripper"}

class PDDLGenerationResult:
    def __init__(self, json_data: Dict[str, Any]):
        self.json_data = json_data


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

        # Initialization flag: will be set to True after scene description is obtained
        self.initialized = False
        self.scene_description: Optional[str] = None
        self._init_lock = threading.Lock()

        self.declare_parameter("real_hardware", False)
        self.real_hardware: bool = self.get_parameter("real_hardware").get_parameter_value().bool_value
        self.declare_parameter("use_ollama", False)
        self.use_ollama: bool = self.get_parameter("use_ollama").get_parameter_value().bool_value
        self.declare_parameter("confirm", True)
        self.confirm: bool = self.get_parameter("confirm").get_parameter_value().bool_value

        self.declare_parameter("scene_desc", "default")
        raw_scene_desc_mode = self.get_parameter("scene_desc").get_parameter_value().string_value
        self.scene_desc_mode = self._validate_enum_parameter("scene_desc", raw_scene_desc_mode, SCENE_DESC_MODES)

        self.declare_parameter("domain", "default")
        raw_domain_mode = self.get_parameter("domain").get_parameter_value().string_value
        self.domain_mode = self._validate_enum_parameter("domain", raw_domain_mode, DOMAIN_MODES)

        self.get_logger().info(
            f"Planner configuration: scene_desc={self.scene_desc_mode}, domain={self.domain_mode}"
        )

        self.declare_parameter("ollama_model", "gpt-oss:20b")
        self.ollama_model: str = self.get_parameter("ollama_model").get_parameter_value().string_value
        # -----------------------------
        # LLM Selection: Gemini or Ollama
        # -----------------------------
        if self.use_ollama:
            self.get_logger().info("Using local LLM via Ollama.")
            # Example: using llama3.1 or any model installed in `ollama list`
            self.llm = ChatOllama(
                model=self.ollama_model,
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

        # Vision clients
        self.vision_vqa_client = ActionClient(self, Prompt, "/vqa")

        # State query service clients
        self.is_home_client = self.create_client(GetSetBool, "/is_home")
        self.is_ready_client = self.create_client(GetSetBool, "/is_ready")
        self.is_handover_client = self.create_client(GetSetBool, "/is_handover")
        self.gripper_is_open_client = self.create_client(GetSetBool, "/gripper_is_open")

        self._tools_called: List[str] = []
        self._tools_called_lock = threading.Lock()

        self.tools = self._initialize_tools()
        self.agent_executor: Optional[AgentExecutor] = None  # Will be created after scene description is ready

        self.chat_history: List[Dict[str, str]] = []
        self.latest_plan: Optional[List[str]] = None
        self.latest_pddl: Optional[PDDLGenerationResult] = None
        self.last_goal_state: Optional[List[Dict[str, Any]]] = None  # Store goal state from previous successful execution
        self.initial_state_summary: Optional[str] = None

        self.confirm_srv = self.create_service(Trigger, "/confirm", self.confirm_service_callback)
        self.reset_state_srv = self.create_service(Trigger, "/reset_state", self.reset_state_callback)

        self.high_level_action_type = PromptScene if self.scene_desc_mode == "custom" else Prompt

        self._action_server = ActionServer(
            self,
            self.high_level_action_type,
            "prompt_high_level",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.response_pub = self.create_publisher(String, "/response", 10)
        self.tts_pub = self.create_publisher(String, "/tts", 10)
        self.benchmark_pub = self.create_publisher(String, "/benchmark_logs", 10)

        if self.scene_desc_mode == "default":
            self.get_logger().info("Ros2 High-Level Agent Node initialized. Fetching scene description...")
            init_thread = threading.Thread(target=self._initialize_scene_description, daemon=False)
            init_thread.start()
            self.get_logger().info("Ros2 High-Level Agent Node ready (waiting for scene description before accepting requests).")
        else:
            if self.scene_desc_mode == "custom":
                self.scene_description = "Scene description provided via /prompt_high_level requests"
            else:
                self.scene_description = None

            self.agent_executor = self._create_pddl_agent_executor(
                scene_desc=self.scene_description if self.scene_desc_mode != "disabled" else None
            )
            with self._init_lock:
                self.initialized = True
            self.get_logger().info(
                "Ros2 High-Level Agent Node ready (scene initialization via /vqa is disabled by configuration)."
            )

    def _validate_enum_parameter(self, name: str, value: str, valid_values: set) -> str:
        normalized = (value or "").strip().lower()
        if normalized in valid_values:
            return normalized
        fallback = "default" if "default" in valid_values else sorted(valid_values)[0]
        self.get_logger().warn(
            f"Invalid value '{value}' for parameter '{name}'. Falling back to '{fallback}'. "
            f"Valid values: {sorted(valid_values)}"
        )
        return fallback

    def _format_for_tts(self, text: str) -> str:
        """
        Convert text to a format suitable for text-to-speech.
        - Removes excessive formatting characters
        - Converts numbered lists to natural language
        - Simplifies technical jargon
        - Makes text more conversational
        """
        if not text:
            return ""
        
        # Replace multiple newlines with single space
        tts_text = text.replace("\n", " ").replace("\r", " ")
        
        # Replace multiple spaces with single space
        tts_text = " ".join(tts_text.split())
        
        # Convert numbered list items (e.g., "1. Do X" -> "Step one, do X")
        tts_text = re.sub(r'^\d+\.\s+', lambda m: f"Step {self._number_to_word(int(m.group(0).split('.')[0]))}, ", tts_text)
        
        # Remove trailing punctuation that doesn't help TTS (multiple exclamation marks, question marks)
        tts_text = re.sub(r'([!?])\1+', r'\1', tts_text)
        
        # Simplify "PDDL" references in speech
        tts_text = tts_text.replace(" PDDL ", " ")
        tts_text = tts_text.replace("PDDL planning", "planning")
        
        return tts_text.strip()
    
    def _number_to_word(self, num: int) -> str:
        """Convert a number to its word representation (1-20, and tens)"""
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        if num < 20:
            return ones[num]
        elif num < 100:
            return tens[num // 10] + ("" if num % 10 == 0 else " " + ones[num % 10])
        else:
            return str(num)  # For larger numbers, just use str representation

    def _benchmark_log(self, label: str):
        t = self.get_clock().now()
        t_sec = t.nanoseconds * 1e-9
        self.benchmark_pub.publish(
            String(data=f"{label},{t_sec:.9f}")
        )
    
    def _initialize_scene_description(self):
        """
        Fetch the initial scene description from /vqa action.
        Sets self.initialized to True once complete.
        """
        try:
            self.get_logger().info("Waiting for /vqa action server...")
            if not self.vision_vqa_client.wait_for_server(timeout_sec=120.0):
                self.get_logger().error("/vqa action server unavailable after 120 seconds. Node will not initialize.")
                with self._init_lock:
                    self.scene_description = "Scene not available"
                    scene_desc = self.scene_description
                self.agent_executor = self._create_pddl_agent_executor(scene_desc=scene_desc)
                with self._init_lock:
                    self.initialized = True
                return
            
            self.get_logger().info("Calling /vqa to describe the scene...")
            goal = Prompt.Goal()
            goal.prompt = "Describe the scene, including what object exists and how many of each object"

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
            
            send_future = self.vision_vqa_client.send_goal_async(goal)
            send_future.add_done_callback(goal_response_callback)

            if not goal_event.wait(timeout=30.0):
                self.get_logger().error("Timeout waiting for VQA goal acceptance")
                with self._init_lock:
                    self.scene_description = "Scene description unavailable"
                    scene_desc = self.scene_description
                self.agent_executor = self._create_pddl_agent_executor(scene_desc=scene_desc)
                with self._init_lock:
                    self.initialized = True
                return
            
            goal_handle = goal_handle_container[0]
            if not goal_handle.accepted:
                self.get_logger().error("VQA goal rejected during scene initialization")
                with self._init_lock:
                    self.scene_description = "Scene description unavailable"
                    scene_desc = self.scene_description
                self.agent_executor = self._create_pddl_agent_executor(scene_desc=scene_desc)
                with self._init_lock:
                    self.initialized = True
                return
            
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(result_callback)

            if not result_event.wait(timeout=120.0):
                self.get_logger().error("Timeout waiting for VQA result")
                with self._init_lock:
                    self.scene_description = "Scene description unavailable"
                    scene_desc = self.scene_description
                self.agent_executor = self._create_pddl_agent_executor(scene_desc=scene_desc)
                with self._init_lock:
                    self.initialized = True
                return

            result = result_container[0].result
            scene_response = None
            if result is not None:
                # Extract only the textual response from the action result
                scene_response = getattr(result, "final_response", None) or str(result)

            with self._init_lock:
                self.scene_description = scene_response if scene_response else "Scene description unavailable"
                scene_desc = self.scene_description
            self.agent_executor = self._create_pddl_agent_executor(scene_desc=scene_desc)
            with self._init_lock:
                self.initialized = True
            
            self.get_logger().info(f"Scene description obtained: {self.scene_description}")
            self.response_pub.publish(String(data=f"Scene analysis: {self.scene_description}"))
            self.tts_pub.publish(String(data=self._format_for_tts(f"I've analyzed the scene. {self.scene_description}")))
            
        except Exception as e:
            self.get_logger().error(f"Exception during scene initialization: {e}")
            with self._init_lock:
                self.scene_description = "Scene description unavailable"
                scene_desc = self.scene_description
            self.agent_executor = self._create_pddl_agent_executor(scene_desc=scene_desc)
            with self._init_lock:
                self.initialized = True


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
        state['gripper_close'] = not state['gripper_open']
        
        self.get_logger().info(f"Current state: {state}")
        return state

    def _state_to_init_predicates(self, current_state: Dict[str, bool]) -> List[Dict[str, Any]]:
        """Convert current state dict to PDDL init predicates for template."""
        init_predicates = []
        
        # Add location predicates
        if current_state.get('robot_at_home'):
            init_predicates.append({"predicate": "robot-at-location", "args": ["home"]})
        if current_state.get('robot_at_ready'):
            init_predicates.append({"predicate": "robot-at-location", "args": ["ready"]})
        if current_state.get('robot_in_handover'):
            init_predicates.append({"predicate": "robot-at-location", "args": ["handover"]})
        
        # Add gripper predicates
        if current_state.get('gripper_open'):
            init_predicates.append({"predicate": "gripper-open", "args": []})
        if current_state.get('gripper_close'):
            init_predicates.append({"predicate": "gripper-close", "args": []})
        
        return init_predicates

    # -----------------------
    # Transcript handling
    # -----------------------
    def transcript_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        if self.scene_desc_mode == "custom":
            warn_msg = (
                "scene_desc=custom requires /prompt_high_level action requests with scene_desc. "
                "Transcript input is ignored in this mode."
            )
            self.get_logger().warn(warn_msg)
            self.response_pub.publish(String(data=warn_msg))
            return

        if not self.initialized:
            self.get_logger().warn("Node not fullly initialized yet. Ignoring transcript.")
            self.response_pub.publish(String(data="Still initializing, please wait a moment..."))
            self.tts_pub.publish(String(data="Still initializing, please wait a moment."))

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

    def _generate_plan(
        self,
        instruction_text: str,
        start_time: Optional[float] = None,
        request_scene_desc: Optional[str] = None,
    ) -> List[str]:
        """Generate a plan using fixed domain and dynamic problem generation."""
        self.chat_history.append({"role": "user", "content": instruction_text})

        with self._tools_called_lock:
            self._tools_called = []

        try:
            self.get_logger().info("High-level agent (PDDL): generating plan with fixed domain...")
            self.response_pub.publish(String(data="Got it! Let me think through that..."))
            self.tts_pub.publish(String(data="Got it. Let me think through that."))
            self.response_pub.publish(String(data="Analyzing scene and determining current state"))
            self.tts_pub.publish(String(data="Analyzing the scene and determining the current state."))

            if self.scene_desc_mode == "custom":
                scene_from_request = (request_scene_desc or "").strip()
                if not scene_from_request:
                    self.get_logger().warn(
                        "scene_desc=custom but request scene_desc is empty. Falling back to placeholder."
                    )
                    scene_from_request = "Scene description unavailable"
                with self._init_lock:
                    self.scene_description = scene_from_request
                self.get_logger().info(f"scene_desc=custom: using scene description from request: {scene_from_request}")
                self.agent_executor = self._create_pddl_agent_executor(scene_desc=scene_from_request)
            elif self.scene_desc_mode == "disabled":
                self.scene_description = None
                if self.agent_executor is None:
                    self.agent_executor = self._create_pddl_agent_executor(scene_desc=None)
            elif self.agent_executor is None:
                with self._init_lock:
                    current_scene = self.scene_description
                self.agent_executor = self._create_pddl_agent_executor(scene_desc=current_scene)

            if self.agent_executor is None:
                raise RuntimeError("Agent executor is not initialized")

            init_predicates: List[Dict[str, Any]] = []
            if self.domain_mode == "default":
                # Get current state from services
                current_state = self._get_current_state()

                # Convert current state to init predicates for PDDL template
                init_predicates = self._state_to_init_predicates(current_state)

                # Merge with last goal state if available (continuous state tracking)
                if self.last_goal_state:
                    self.get_logger().info(f"Merging previous goal state into init: {self.last_goal_state}")
                    # Add previous goals to init, avoiding duplicates
                    existing_predicates = {(p["predicate"], tuple(p.get("args", []))): p for p in init_predicates}
                    for goal in self.last_goal_state:
                        key = (goal["predicate"], tuple(goal.get("args", [])))
                        if key not in existing_predicates:
                            init_predicates.append(goal)
                    self.get_logger().info(f"Combined init predicates: {init_predicates}")
            else:
                self.get_logger().info(
                    f"domain={self.domain_mode}: skipping state service init predicates; expecting init from LLM output."
                )

            # Build LangChain chat history
            langchain_history = []
            for msg in self.chat_history[:-1]:
                if msg["role"] == "user":
                    langchain_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_history.append(AIMessage(content=msg["content"]))

            # Invoke agent with instruction only (state will be in init block)
            agent_resp = self.agent_executor.invoke({
                "input": instruction_text,
                "chat_history": langchain_history
            })

            # Extract structured response from agent state
            # According to Langchain docs, structured response is in 'structured_response' key
            agent_output = agent_resp.get("structured_response")
            
            if agent_output is None:
                # Fallback: check for direct output key
                agent_output = agent_resp.get("output")
            
            self.get_logger().info(f"Agent output (raw): {agent_output}")

            # Check if this is a clarifying question (string response)
            if isinstance(agent_output, str):
                final_text_stripped = agent_output.strip()
                if final_text_stripped.upper().startswith("NORMAL"):
                    clarification = final_text_stripped[len("NORMAL"):].lstrip(" :")
                    clarification = clarification or final_text_stripped
                    self.get_logger().info(f"Agent requests clarification: {clarification}")
                    self.chat_history.append({"role": "assistant", "content": clarification})
                    self.response_pub.publish(String(data=clarification))
                    self.tts_pub.publish(String(data=self._format_for_tts(clarification)))
                    return []
                # If it's a string but not a clarification, treat as fallback JSON
                final_text = final_text_stripped
                self.chat_history.append({"role": "assistant", "content": final_text})
            else:
                # Structured output - convert Pydantic model to JSON for logging
                final_text = json.dumps(agent_output.model_dump() if hasattr(agent_output, 'model_dump') else agent_output.dict(), indent=2)
                self.chat_history.append({"role": "assistant", "content": final_text})

            self.response_pub.publish(String(data="Generating planning data and PDDL problem file"))
            self.tts_pub.publish(String(data="Generating planning data."))

            # Parse the planning data - handle both structured output and fallback JSON string parsing
            if isinstance(agent_output, str):
                # Fallback: parse as JSON string
                planning_payload = self._parse_planning_json(final_text)
            else:
                # Structured output: convert Pydantic model directly
                planning_payload = self._structured_output_to_dict(agent_output)

            if not planning_payload:
                msg = "Hmm... I couldn't generate valid planning data. Could you try rephrasing that?"
                self.get_logger().error("LLM response could not be parsed into planning data.")
                self.response_pub.publish(String(data=msg))
                self.tts_pub.publish(String(data=self._format_for_tts(msg)))
                return []

            if self.domain_mode != "default":
                init_predicates = planning_payload.get("init", [])

            self.initial_state_summary = self._format_predicates(init_predicates)

            json_gen_data = {
                "locations": planning_payload.get("locations", []),
                "objects": planning_payload.get("objects", []),
                "goals": planning_payload.get("goals", []),
                "init": init_predicates,
            }

            # Store JSON for reference
            json_result = PDDLGenerationResult(json_data=json_gen_data)
            self.latest_pddl = json_result

            # Save JSON and generate PDDL using transform.py
            tmpdir = tempfile.mkdtemp(prefix="pddl_")
            json_path = Path(tmpdir) / "data.json"
            problem_path = Path(tmpdir) / "problem.pddl"
            
            # Write JSON data to file
            json_data = {"data": json_gen_data}
            json_path.write_text(json.dumps(json_data, indent=2))
            self.get_logger().info(f"JSON data saved to {json_path}")
            self.get_logger().debug(f"JSON data:\n{json.dumps(json_data, indent=2)}")

            template_problem = self._get_template_problem_path()
            domain_pddl = self._get_domain_pddl_path()

            # Call transform.py to generate problem PDDL
            self.get_logger().info(
                f"Calling transform.py to generate problem file with template: {template_problem}"
            )
            transform_result = self._run_transform(template_problem, str(json_path), str(problem_path))
            
            if not transform_result or not problem_path.exists():
                msg = "Failed to generate PDDL problem file. Transform failed."
                self.get_logger().error("Transform failed or problem file not created.")
                self.response_pub.publish(String(data=msg))
                self.tts_pub.publish(String(data="Failed to generate the planning file. Please try again."))
                return []

            self.get_logger().debug(f"Problem:\n{problem_path.read_text()}")

            # Run Fast Downward with domain.pddl and generated problem.pddl
            self.response_pub.publish(String(data="Solving the PDDL problem using Fast Downward"))
            self.tts_pub.publish(String(data="Now solving the planning problem."))
            plan_result = self._run_fast_downward(domain_pddl, str(problem_path))

            if start_time is not None:
                end_time = time.perf_counter()
                self._benchmark_log("plan_generated")
                self.benchmark_pub.publish(String(data=f"Plan generated in: ,{end_time - start_time:.2f}"))

            if plan_result.status != "success" or not plan_result.plan.strip():
                msg = f"I couldn't find a valid plan. The planner returned: {plan_result.status}"
                self.get_logger().warn(f"Planner returned no plan. status={plan_result.status} rc={plan_result.return_code}")
                self.response_pub.publish(String(data=msg))
                self.tts_pub.publish(String(data="I couldn't find a valid plan for that request. Could you try rephrasing it?"))
                # Keep last_goal_state intact - planning failed but state hasn't changed
                return []

            self.get_logger().info("Plan obtained from Fast Downward. Parsing to steps...")
            plan_lines = self._parse_plan_text(plan_result.plan)
            if not plan_lines:
                msg = "No actionable steps could be parsed from the plan."
                self.get_logger().warn("No actionable plan lines parsed.")
                self.response_pub.publish(String(data=msg))
                self.tts_pub.publish(String(data="I couldn't parse the plan into steps. Please try again."))
                # Keep last_goal_state intact - parsing failed but state hasn't changed
                return []

            self.latest_plan = plan_lines

            readable_plan = "\n".join([f"{i+1}. {s}" for i, s in enumerate(plan_lines)])
            if self.confirm:
                self.response_pub.publish(String(
                    data=f"Here's what I plan to do:\n{readable_plan}\n\nPlease review and confirm if this looks good!"
                ))
                # Create a more natural TTS version of the plan
                tts_plan = ". ".join([s for s in plan_lines])
                self.tts_pub.publish(String(data=f"Here is my plan: {tts_plan}. Please confirm if this looks good."))
                self.get_logger().info(f"Generated plan with {len(plan_lines)} steps, waiting for /confirm.")
            else:
                def execute_plan():
                    self.response_pub.publish(String(data="Got it! Executing your approved plan now..."))
                    self.tts_pub.publish(String(data="Executing the plan now."))
                    self.get_logger().info("Executing confirmed plan...")

                    for i, step in enumerate(self.latest_plan, start=1):
                        actions_completed = self.latest_plan[: i - 1]
                        step_payload = self._compose_step_instruction(step, actions_completed)

                        self.response_pub.publish(String(data=f"Starting step {i}: {step}"))
                        self.tts_pub.publish(String(data=self._format_for_tts(f"Step {i}: {step}")))
                        result = self.send_step_to_medium_async(step_payload)

                        if result is None or not result.success:
                            msg = f"Step {i} failed: {step}. Stopping execution."
                            self.response_pub.publish(String(data=msg))
                            self.tts_pub.publish(String(data=f"Step {i} failed. Stopping execution."))
                            self.get_logger().error(msg)
                            # Clear last goal state on execution failure - state is now uncertain after partial execution
                            self.last_goal_state = None
                            break
                        else:
                            done_msg = f"Step {i} completed successfully."
                            self.response_pub.publish(String(data=done_msg))
                            self.tts_pub.publish(String(data=f"Step {i} completed."))
                            self.get_logger().info(done_msg)

                    end_time = time.perf_counter()
                    benchmark_info = f"High-level action completed in {end_time - self.start_time:.2f} seconds."
                    self.benchmark_pub.publish(String(data=benchmark_info))
                    self.response_pub.publish(String(data="Plan execution finished."))
                    self.tts_pub.publish(String(data="Plan execution finished."))
                    self.get_logger().info("All steps done. Storing goal state for continuous tracking.")
                    
                    # Store goal state from successfully executed plan for next instruction
                    if self.latest_pddl and self.latest_pddl.json_data:
                        self.last_goal_state = self.latest_pddl.json_data.get("goals", [])
                        self.get_logger().info(f"Stored goal state for next instruction: {self.last_goal_state}")
                    
                    self.chat_history.clear()
                    self.latest_plan = None
                    self.latest_pddl = None
                    self.initial_state_summary = None

                execution_thread = threading.Thread(target=execute_plan, daemon=True)
                execution_thread.start()

            return plan_lines

        except Exception as e:
            self.get_logger().error(f"Error generating plan: {e}")
            self.response_pub.publish(String(data="Sorry, something went wrong while planning."))
            self.tts_pub.publish(String(data="Sorry, something went wrong. Please try again."))
            # Keep last_goal_state intact - planning exception but state hasn't changed
            return []

    def confirm_service_callback(self, request, response):
        """Execute the latest plan when confirmed."""
        if not self.initialized:
            response.success = False
            response.message = "Node not yet initialized. Please wait for scene analysis to complete."
            self.response_pub.publish(String(data=response.message))
            self.tts_pub.publish(String(data="Node is still initializing. Please wait."))
            return response

        if not self.latest_plan:
            response.success = False
            response.message = "No plan to confirm. Please give a new instruction first."
            self.response_pub.publish(String(data=response.message))
            self.tts_pub.publish(String(data="No plan to confirm. Please give me a new instruction."))
            return response

        def execute_plan():
            self.response_pub.publish(String(data="Got it! Executing your approved plan now..."))
            self.tts_pub.publish(String(data="Executing the plan now."))
            self.get_logger().info("Executing confirmed plan...")

            for i, step in enumerate(self.latest_plan, start=1):
                actions_completed = self.latest_plan[: i - 1]
                step_payload = self._compose_step_instruction(step, actions_completed)

                self.response_pub.publish(String(data=f"Starting step {i}: {step}"))
                self.tts_pub.publish(String(data=self._format_for_tts(f"Step {i}: {step}")))
                result = self.send_step_to_medium_async(step_payload)

                if result is None or not result.success:
                    msg = f"Step {i} failed: {step}. Stopping execution."
                    self.response_pub.publish(String(data=msg))
                    self.tts_pub.publish(String(data=f"Step {i} failed. Stopping execution."))
                    self.get_logger().error(msg)
                    # Clear last goal state on execution failure - state is now uncertain after partial execution
                    self.last_goal_state = None
                    break
                else:
                    done_msg = f"Step {i} completed successfully."
                    self.response_pub.publish(String(data=done_msg))
                    self.tts_pub.publish(String(data=f"Step {i} completed."))
                    self.get_logger().info(done_msg)

            self.response_pub.publish(String(data="Plan execution finished."))
            self.tts_pub.publish(String(data="Plan execution finished."))
            self.get_logger().info("All steps done. Storing goal state for continuous tracking.")
            
            # Store goal state from successfully executed plan for next instruction
            if self.latest_pddl and self.latest_pddl.json_data:
                self.last_goal_state = self.latest_pddl.json_data.get("goals", [])
                self.get_logger().info(f"Stored goal state for next instruction: {self.last_goal_state}")
            
            self.chat_history.clear()
            self.latest_plan = None
            self.latest_pddl = None
            self.initial_state_summary = None

        execution_thread = threading.Thread(target=execute_plan, daemon=True)
        execution_thread.start()

        response.success = True
        response.message = "Plan execution started."
        return response

    def reset_state_callback(self, request, response):
        """Reset the stored goal state to start fresh."""
        self.last_goal_state = None
        self.chat_history.clear()
        self.latest_plan = None
        self.latest_pddl = None
        self.initial_state_summary = None
        
        response.success = True
        response.message = "State has been reset. Ready for new instructions."
        self.get_logger().info("State reset via /reset_state service")
        self.response_pub.publish(String(data=response.message))
        self.tts_pub.publish(String(data=response.message))
        return response

    # -----------------------
    # JSON and Transform helpers
    # -----------------------
    def _run_transform(self, template_file: str, data_json_file: str, output_file: str) -> bool:
        """Call the appropriate transform script to generate problem PDDL from template and JSON data."""
        workdir = str(Path(template_file).parent)

        if self.domain_mode == "blocksworld":
            transform_script = TRANSFORM_BLOCKSWORLD_PY
        elif self.domain_mode == "gripper":
            transform_script = TRANSFORM_GRIPPER_PY
        else:
            transform_script = TRANSFORM_PY

        cmd = ["python3", transform_script, template_file, data_json_file, output_file]
        self.get_logger().info(f"Calling transform script ({self.domain_mode}): {' '.join(cmd)} (workdir={workdir})")
        try:
            result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=60)
            self.get_logger().info(f"Transform stdout:\n{result.stdout}")
            if result.returncode != 0:
                self.get_logger().error(f"Transform failed with return code {result.returncode}: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            self.get_logger().error("Transform timed out")
            return False
        except Exception as e:
            self.get_logger().error(f"Error calling transform.py: {e}")
            return False

    def _run_fast_downward(self, domain_file: str, problem_file: str, timeout: int = 300) -> PlanningResult:
        """Call Fast Downward to produce a plan."""
        workdir = str(Path(problem_file).parent)
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

    def _structured_output_to_dict(self, output: Any) -> Optional[Dict[str, Any]]:
        """Convert structured output (Pydantic model) to internal dictionary format."""
        try:
            if hasattr(output, "model_dump"):
                # Pydantic v2
                output_dict = output.model_dump()
            elif hasattr(output, "dict"):
                # Pydantic v1
                output_dict = output.dict()
            else:
                return None

            # Navigate to the data object
            if "data" in output_dict:
                base_data = output_dict["data"]
            else:
                base_data = output_dict

            if not isinstance(base_data, dict):
                self.get_logger().error("Structured output 'data' field is not a dict")
                return None

            # Extract and validate fields
            locations = base_data.get("locations", [])
            objects = base_data.get("objects", [])
            goals = base_data.get("goals", [])
            init = base_data.get("init", [])

            if not isinstance(objects, list) or not isinstance(goals, list):
                self.get_logger().error("Structured output objects/goals are not lists")
                return None

            if locations is not None and not isinstance(locations, list):
                locations = []

            if init is not None and not isinstance(init, list):
                init = []

            # Convert predicates to standardized format
            sanitized_goals: List[Dict[str, Any]] = []
            for goal in goals:
                if isinstance(goal, dict):
                    sanitized_goals.append({
                        "predicate": str(goal.get("predicate", "")).strip(),
                        "args": [str(arg).strip() for arg in goal.get("args", [])]
                    })
                elif hasattr(goal, "predicate"):
                    # Pydantic model
                    sanitized_goals.append({
                        "predicate": str(goal.predicate).strip(),
                        "args": [str(arg).strip() for arg in getattr(goal, "args", [])]
                    })

            sanitized_init: List[Dict[str, Any]] = []
            for init_pred in init or []:
                if isinstance(init_pred, dict):
                    sanitized_init.append({
                        "predicate": str(init_pred.get("predicate", "")).strip(),
                        "args": [str(arg).strip() for arg in init_pred.get("args", [])]
                    })
                elif hasattr(init_pred, "predicate"):
                    # Pydantic model
                    sanitized_init.append({
                        "predicate": str(init_pred.predicate).strip(),
                        "args": [str(arg).strip() for arg in getattr(init_pred, "args", [])]
                    })

            sanitized_locations = [str(loc).strip() for loc in locations if str(loc).strip()]
            sanitized_objects = [str(obj).strip() for obj in objects if str(obj).strip()]

            return {
                "locations": sanitized_locations,
                "objects": sanitized_objects,
                "goals": sanitized_goals,
                "init": sanitized_init,
            }

        except Exception as e:
            self.get_logger().error(f"Error converting structured output: {e}")
            return None

    def _parse_planning_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM output into planning data dict following data.json structure."""
        if not text:
            return None

        def _try_load(candidate: str) -> Optional[Dict[str, Any]]:
            try:
                return json.loads(candidate)
            except Exception:
                return None

        text_stripped = text.strip()
        json_candidates = []

        # Prefer fenced json blocks if present
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_stripped, re.DOTALL)
        if fenced:
            json_candidates.append(fenced.group(1))

        # Raw text as-is
        json_candidates.append(text_stripped)

        # Largest brace-wrapped span fallback
        brace_start = text_stripped.find("{")
        brace_end = text_stripped.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            json_candidates.append(text_stripped[brace_start:brace_end + 1])

        parsed: Optional[Dict[str, Any]] = None
        for candidate in json_candidates:
            parsed = _try_load(candidate)
            if isinstance(parsed, dict):
                break

        if not isinstance(parsed, dict):
            self.get_logger().error("Failed to parse LLM JSON output")
            return None

        base_data = parsed.get("data", parsed)
        if not isinstance(base_data, dict):
            self.get_logger().error("Parsed JSON missing 'data' object")
            return None
        
        locations = base_data.get("locations", [])
        objects = base_data.get("objects", [])
        goals = base_data.get("goals", [])
        init = base_data.get("init", [])

    
        if not isinstance(objects, list) or not isinstance(goals, list) or not isinstance(locations, list):
            self.get_logger().error("JSON does not contain list fields for objects/goals/locations")
            return None

        if init is not None and not isinstance(init, list):
            self.get_logger().error("JSON field 'init' must be a list when provided")
            return None

        sanitized_locations = [str(loc).strip() for loc in locations if str(loc).strip()]
        sanitized_objects = [str(obj).strip() for obj in objects if str(obj).strip()]

        sanitized_goals: List[Dict[str, Any]] = []
        for goal in goals:
            if not isinstance(goal, dict):
                continue
            predicate = goal.get("predicate")
            args = goal.get("args", [])
            if predicate is None or not isinstance(args, list):
                continue
            sanitized_goals.append({
                "predicate": str(predicate).strip(),
                "args": [str(arg).strip() for arg in args]
            })

        sanitized_init: List[Dict[str, Any]] = []
        for init_predicate in init or []:
            if not isinstance(init_predicate, dict):
                continue
            predicate = init_predicate.get("predicate")
            args = init_predicate.get("args", [])
            if predicate is None or not isinstance(args, list):
                continue
            sanitized_init.append({
                "predicate": str(predicate).strip(),
                "args": [str(arg).strip() for arg in args]
            })

        if not sanitized_objects and not sanitized_goals and not sanitized_locations:
            self.get_logger().error("Parsed JSON missing objects, goals, and locations")
            return None

        return {
            "locations": sanitized_locations,
            "objects": sanitized_objects,
            "goals": sanitized_goals,
            "init": sanitized_init,
        }

    def _format_predicates(self, predicates: List[Dict[str, Any]]) -> str:
        """Create a compact string description of predicate list."""
        parts = []
        for pred in predicates:
            name = str(pred.get("predicate", "")).strip()
            args = pred.get("args", []) or []
            if not name:
                continue
            arg_str = " ".join(str(a).strip() for a in args if str(a).strip())
            parts.append(f"{name} {arg_str}".strip())
        return "; ".join(parts) if parts else "unknown"

    def _get_template_problem_path(self) -> str:
        if self.domain_mode == "blocksworld":
            return TEMPLATE_PROBLEM_BLOCKSWORLD
        if self.domain_mode == "gripper":
            return TEMPLATE_PROBLEM_GRIPPER
        return TEMPLATE_PROBLEM

    def _get_domain_pddl_path(self) -> str:
        if self.domain_mode == "blocksworld":
            return DOMAIN_PDDL_BLOCKSWORLD
        if self.domain_mode == "gripper":
            return DOMAIN_PDDL_GRIPPER
        return DOMAIN_PDDL

    def _compose_step_instruction(self, step_text: str, actions_completed: List[str]) -> str:
        """Attach workspace context, initial state, and prior steps to the current instruction."""
        if self.scene_desc_mode == "disabled":
            workspace = "scene description disabled"
        else:
            workspace = self.scene_description or "unknown"
        initial = self.initial_state_summary or "unknown"
        completed = "; ".join(actions_completed) if actions_completed else "none"
        return f"Workspace context: {workspace}\nInitial state: {initial}\nactions completed: {completed}\nInstruction: {step_text}"

    # -----------------------
    # Tools
    # -----------------------
    def _initialize_tools(self) -> List[BaseTool]:
        tools: List[BaseTool] = []
        
        @tool
        def vqa(question: str) -> str:
            """
            Call /vqa which returns an answer to a visual question.
            """
            tool_name = "vqa"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_vqa_client.wait_for_server(timeout_sec=5.0):
                    self.get_logger().error("/vqa action server unavailable")
                    return None
                goal = Prompt.Goal()
                goal.prompt = question
                send_future = self.vision_vqa_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, send_future)
                goal_handle = send_future.result()
                if not goal_handle.accepted:
                    self.get_logger().error("VQA goal rejected")
                    return None
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                result = result_future.result().result
                return result
            except Exception as e:
                self.get_logger().error(f"Exception when sending to VQA: {e}")
                return None

        if self.domain_mode == "default":
            tools.append(vqa)

        return tools

    # -----------------------
    # Create Planning Agent
    # -----------------------
    def _build_system_message(self, scene_desc: Optional[str]) -> str:
        include_scene_desc = self.scene_desc_mode != "disabled"

        if self.domain_mode == "blocksworld":
            return self._build_blocksworld_system_message(scene_desc, include_scene_desc)
        if self.domain_mode == "gripper":
            return self._build_gripper_system_message(scene_desc, include_scene_desc)
        return self._build_default_system_message(scene_desc, include_scene_desc)

    def _build_default_system_message(self, scene_desc: Optional[str], include_scene_desc: bool) -> str:
        scene_section = ""
        if include_scene_desc:
            effective_scene_desc = scene_desc if scene_desc else "Scene not yet analyzed"
            scene_section = f"\nCurrent scene description: {effective_scene_desc}\n"

        system_message = f"""You are a planning assistant for a robot manipulation system.

        {scene_section}

        Your job: analyze the user's instruction and determine the planning data needed to achieve the goal.

        Analyze the instruction to extract:
        - locations: task-specific locations needed to complete the instruction
        - objects: the objects present in the workspace
        - goals: the desired FINAL state (do not include intermediate goals)

        If the user asks to "handover" or "hand me" an object, the requested object should be located at handover.

        Important guidelines:
        - Object names CANNOT contain spaces, use underscores instead
        - You can add modifiers to object names, like screwdriver_leftmost, so you should NOT ask clarifying questions
        - If the instruction is unclear, respond with a clarifying question prefixed with NORMAL
        - You may call tools like vqa if needed, but the final reply must provide the structured output described above

        Available predicates (DO NOT create new ones):
        - gripper-open: args []
        - gripper-close: args []
        - robot-at-location: args [location_name]
        - robot-at-object: args [object_name]
        - object-at-location: args [object_name, location_name]
        - robot-have: args [object_name]
        - Available locations: home, ready, handover
        """

        return system_message


    def _build_blocksworld_system_message(self, scene_desc: Optional[str], include_scene_desc: bool) -> str:
        scene_section = ""
        if include_scene_desc:
            effective_scene_desc = scene_desc if scene_desc else "Scene not yet analyzed"
            scene_section = f"\nCurrent scene description: {effective_scene_desc}\n"

        system_message = f"""You are a planning assistant for a blocksworld system.

        {scene_section}

        Your job: analyze the user's instruction and determine the planning data needed.

        Extract:
        - objects: the blocks present in the workspace (use color initials: r=red, g=green, b=blue, y=yellow, p=purple)
        - init: the CURRENT state predicates
        - goals: the desired FINAL state predicates

        Important guidelines:
        - Follow the instruction even if there are inconsistencies with the initial state
        - You can infer objects from the instruction if not explicitly mentioned in the initial state

        Available predicates (DO NOT create new ones):
        - clear: args [object_name] — the object is clear (nothing on top)
        - on-table: args [object_name] — the object is on the table surface
        - arm-empty: args [] — the robot arm is not holding any object
        - holding: args [object_name] — the robot arm is holding the object
        - on: args [object_name, object_name] — first object is on top of second
        """

        return system_message

    def _build_gripper_system_message(self, scene_desc: Optional[str], include_scene_desc: bool) -> str:
        scene_section = ""
        if include_scene_desc:
            effective_scene_desc = scene_desc if scene_desc else "Scene not yet analyzed"
            scene_section = f"\nCurrent scene description: {effective_scene_desc}\n"

        system_message = f"""You are a planning assistant for a gripper robot system.

        {scene_section}

        Your job: analyze the user's instruction and determine the planning data needed.

        Extract:
        - objects: the rooms, balls, and grippers involved
        - init: the CURRENT state predicates
        - goals: the desired FINAL state predicates

        Important guidelines:
        - Object names CANNOT contain spaces, use underscores
        - Identify all necessary rooms, balls, and grippers from the instruction

        Available predicates (DO NOT create new ones):
        - room: args [room_name] — type declaration for rooms
        - ball: args [ball_name] — type declaration for balls
        - gripper: args [gripper_name] — type declaration for grippers
        - at-robby: args [room_name] — the robot is in the specified room
        - at: args [ball_name, room_name] — the ball is in the specified room
        - free: args [gripper_name] — the gripper is not holding anything
        - carry: args [object_name, gripper_name] — the gripper is holding the object
        """

        return system_message

    def _create_pddl_agent_executor(self, scene_desc: Optional[str] = None) -> AgentExecutor:
        """Create an agent that generates planning data using structured output."""
        if scene_desc is None:
            with self._init_lock:
                scene_desc = self.scene_description

        system_message = self._build_system_message(scene_desc)

        # Select the appropriate output schema based on domain mode
        if self.domain_mode == "blocksworld":
            output_schema = BlocksworldDomainData
        elif self.domain_mode == "gripper":
            output_schema = GripperDomainData
        else:
            output_schema = DefaultDomainData

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Use response_format parameter for structured output
        agent = create_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
            response_format=output_schema,
        )

        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=12)

    # -----------------------
    # Action server callbacks
    # -----------------------
    def goal_callback(self, goal_request) -> GoalResponse:
        if not self.initialized:
            self.get_logger().warn("[high-level action] Goal received but node not initialized yet.")
            return GoalResponse.REJECT
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

        self.get_logger().info(f"Current scene description: {self.scene_description}")

        prompt_text = goal_handle.request.prompt
        self.get_logger().info(f"[high-level action] Executing prompt: {prompt_text}")

        feedback_msg = self.high_level_action_type.Feedback()
        result_container: Dict[str, Any] = {"success": False, "final_response": ""}

        def run_pipeline():
            try:
                goal_text = goal_handle.request.prompt.strip()
                if not goal_text:
                    result_container["success"] = False
                    result_container["final_response"] = "Empty prompt"
                    return

                self.get_logger().info(f"High-level Prompt action received: {goal_text}")

                request_scene_desc = getattr(goal_handle.request, "scene_desc", None)
                self.get_logger().info(f"Scene description attached to request: {request_scene_desc}")
                steps = self._generate_plan(
                    goal_text,
                    start_time=self.start_time,
                    request_scene_desc=request_scene_desc,
                )
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

        result_msg = self.high_level_action_type.Result()
        result_msg.success = bool(result_container.get("success", False))
        result_msg.final_response = str(result_container.get("final_response", ""))

        goal_handle.succeed()
        self.get_logger().info(f"[high-level action] Goal finished. success={result_msg.success}")
        return result_msg

    # -----------------------
    # Medium-level communication
    # -----------------------
    def send_step_to_medium_async(self, step_text: str, timeout: float = 120.0) -> Optional[Prompt.Result]:
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