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
from langchain.agents import create_agent, AgentExecutor
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


from dotenv import load_dotenv

ENV_PATH = '/home/group11/final_project_ws/src/high_level_pddl_planner/.env'
load_dotenv(dotenv_path=ENV_PATH)

FAST_DOWNWARD_PY = os.getenv("FAST_DOWNWARD_PY", "./fastdownward/fast-downward.py")
SAS_PATH_PLAN = "/home/group11/final_project_ws/src/high_level_pddl_planner/sas_plan"
TRANSFORM_PY = os.getenv("TRANSFORM_PY", "./transform.py")
TEMPLATE_PROBLEM = os.getenv("TEMPLATE_PROBLEM", "./template_problem.pddl")
DOMAIN_PDDL = os.getenv("DOMAIN_PDDL", "./domain.pddl")

class PlanningData(BaseModel):
    """Structured planning data for robot manipulation tasks."""
    objects: List[str] = Field(description="List of object names involved in the task")
    goals: List[Dict[str, Any]] = Field(description="List of goal predicates with format {predicate: str, args: List[str]}")


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

        self.get_logger().info("Ros2 High-Level Agent Node initialized. Fetching scene description...")

        init_thread = threading.Thread(target=self._initialize_scene_description, daemon=False)
        init_thread.start()

        self.get_logger().info("Ros2 High-Level Agent Node ready (waiting for scene description before accepting requests).")

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
            if not self.vision_vqa_client.wait_for_server(timeout_sec=30.0):
                self.get_logger().error("/vqa action server unavailable after 30 seconds. Node will not initialize.")
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
        state['gripper_closed'] = not state['gripper_open']
        
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
        if current_state.get('gripper_closed'):
            init_predicates.append({"predicate": "gripper-closed", "args": []})
        
        return init_predicates

    # -----------------------
    # Transcript handling
    # -----------------------
    def transcript_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        if not self.initialized:
            self.get_logger().warn("Node not fullly initialized yet. Ignoring transcript.")
            self.response_pub.publish(String(data="Still initializing, please wait a moment..."))

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
            
            # Convert current state to init predicates for PDDL template
            init_predicates = self._state_to_init_predicates(current_state)

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

            final_text = agent_resp.get("output") if isinstance(agent_resp, dict) else str(agent_resp)
            self.get_logger().info(f"Agent output (raw):\n{final_text}")

            # If the agent is asking a clarifying question (prefixed with NORMAL), just relay it
            final_text_stripped = final_text.strip()
            if final_text_stripped.upper().startswith("NORMAL"):
                clarification = final_text_stripped[len("NORMAL"):].lstrip(" :")
                clarification = clarification or final_text_stripped
                self.get_logger().info(f"Agent requests clarification: {clarification}")
                self.chat_history.append({"role": "assistant", "content": clarification})
                self.response_pub.publish(String(data=clarification))
                return []

            self.chat_history.append({"role": "assistant", "content": final_text})

            self.response_pub.publish(String(data="Generating planning data and PDDL problem file"))

            # The agent output should contain structured_response from the agent executor
            if not isinstance(agent_resp, dict) or "structured_response" not in agent_resp:
                msg = "Hmm... I couldn't generate valid planning data. Could you try rephrasing that?"
                self.get_logger().error("Failed to get structured response from agent.")
                self.response_pub.publish(String(data=msg))
                return []

            structured_data = agent_resp["structured_response"]
            
            # structured_data is a PlanningData Pydantic model instance
            if isinstance(structured_data, PlanningData):
                json_gen_data = {
                    "objects": structured_data.objects,
                    "goals": structured_data.goals,
                    "init": init_predicates
                }
            else:
                # Fallback if it's already a dict
                json_gen_data = structured_data if isinstance(structured_data, dict) else structured_data.model_dump()
                json_gen_data["init"] = init_predicates

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

            # Call transform.py to generate problem PDDL
            self.get_logger().info(f"Calling transform.py to generate problem file...")
            transform_result = self._run_transform(TEMPLATE_PROBLEM, str(json_path), str(problem_path))
            
            if not transform_result or not problem_path.exists():
                msg = "Failed to generate PDDL problem file. Transform failed."
                self.get_logger().error("Transform failed or problem file not created.")
                self.response_pub.publish(String(data=msg))
                return []

            self.get_logger().debug(f"Problem:\n{problem_path.read_text()}")

            # Run Fast Downward with domain.pddl and generated problem.pddl
            self.response_pub.publish(String(data="Solving the PDDL problem using Fast Downward"))
            plan_result = self._run_fast_downward(DOMAIN_PDDL, str(problem_path))

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
        if not self.initialized:
            response.success = False
            response.message = "Node not yet initialized. Please wait for scene analysis to complete."
            self.response_pub.publish(String(data=response.message))
            return response

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
    # JSON and Transform helpers
    # -----------------------
    def _run_transform(self, template_file: str, data_json_file: str, output_file: str) -> bool:
        """Call transform.py to generate problem PDDL from template and JSON data."""
        cmd = ["python3", TRANSFORM_PY, template_file, data_json_file, output_file]
        self.get_logger().info(f"Calling transform.py: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
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

        tools.append(vqa)

        return tools

    # -----------------------
    # Create Planning Agent
    # -----------------------
    def _create_pddl_agent_executor(self, scene_desc: Optional[str] = None) -> AgentExecutor:
        """Create an agent that generates planning data (objects and goals) in JSON format using structured output."""
        if scene_desc is None:
            with self._init_lock:
                scene_desc = self.scene_description if self.scene_description else "Scene not yet analyzed"

        system_message = f"""You are a planning assistant for a robot manipulation system.

        The current robot state will be provided in the user message.

        Current scene description: {scene_desc}

        Your task is to analyze the user's instruction and generate planning data that defines:
        1. Objects involved in the task (list the specific object names)
        2. Goal predicates that should be achieved (list the desired states)

        The planning data format:
        - objects: List of object names (e.g., ["gear", "bolt"])
        - goals: List of goal predicates, each with predicate name and arguments
          Examples:
          - {{"predicate": "gripper-closed", "args": []}}
          - {{"predicate": "robot-at-location", "args": ["handover"]}}
          - {{"predicate": "object-at-location", "args": ["gear", "handover"]}}
          - {{"predicate": "robot-have", "args": ["gear"]}}

        Common predicates:
        - gripper-open: Empty args list
        - gripper-closed: Empty args list
        - robot-at-location: args = [location_name]
        - object-at-location: args = [object_name, location_name]
        - robot-have: args = [object_name]

        Available locations: home, ready, handover

        IMPORTANT GUIDELINES:
        - If the instruction is unclear, respond with a clarifying question only. Prefix with NORMAL and ask the question.
        - Use vision tools (vqa) to inspect the scene and identify objects if needed.
        - When asked to "handover" an object, goals should include gripper-closed at handover location.
        - List all relevant objects in the objects array.
        - Include all necessary goal predicates for the desired end state.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent with structured output
        agent = create_agent(
            self.llm,
            self.tools,
            prompt,
            response_format=ToolStrategy(PlanningData, handle_errors=True)
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