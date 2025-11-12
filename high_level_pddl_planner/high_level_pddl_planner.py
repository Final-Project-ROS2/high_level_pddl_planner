"""
ROS2 High-Level Agent Node (PDDL-based) - Enhanced Version

Replaces the previous "LLM -> direct steps" workflow with:
LLM (with vision tools) -> generate PDDL domain & problem -> run Fast Downward -> parse plan -> dispatch steps to /medium_level

Features:
- Chat history management
- /confirm service for plan approval
- Plan logging to /response topic
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

# Action used for inter-level communication (Prompt action)
from custom_interfaces.action import Prompt

# Vision service types (assumes these exist in your workspace)
from custom_interfaces.srv import (
    DetectObjects,
    ClassifyBBox,
    DetectGrasps,
    DetectGraspBBox,
    UnderstandScene,
)

# LangChain LLM/agent imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

# Load env
ENV_PATH = '/home/group11/final_project_ws/src/high_level_pddl_planner/.env'
load_dotenv(dotenv_path=ENV_PATH)

# Fast Downward path: set FAST_DOWNWARD_PY env or default
FAST_DOWNWARD_PY = os.getenv("FAST_DOWNWARD_PY", "./fastdownward/fast-downward.py")
SAS_PATH_PLAN = "/home/group11/final_project_ws/src/high_level_pddl_planner/sas_plan"


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
        self.get_logger().info("Initializing Ros2 High-Level Agent Node (PDDL mode)...")

        self.declare_parameter("real_hardware", False)
        self.real_hardware: bool = self.get_parameter("real_hardware").get_parameter_value().bool_value

        # LLM init (Gemini or configured model)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.get_logger().warn("No LLM API key found in environment variable GEMINI_API_KEY.")

        # Configure LLM; temperature 0 to favor deterministic outputs for PDDL
        self.llm = ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
                                          google_api_key=api_key, temperature=0.0)

        # Transcript subscription
        self.transcript_sub = self.create_subscription(String, "/transcript", self.transcript_callback, 10)
        self._last_transcript_lock = threading.Lock()
        self._last_transcript: Optional[str] = None

        # Medium-level action client
        self.medium_level_client = ActionClient(self, Prompt, "/medium_level")

        # Vision service clients - real types from your specification
        self.vision_detect_objects_client = self.create_client(DetectObjects, "/vision/detect_objects")
        self.vision_classify_all_client = self.create_client(Trigger, "/vision/classify_all")
        self.vision_classify_bb_client = self.create_client(ClassifyBBox, "/vision/classify_bb")
        self.vision_detect_grasp_client = self.create_client(DetectGrasps, "/vision/detect_grasp")
        self.vision_detect_grasp_bb_client = self.create_client(DetectGraspBBox, "/vision/detect_grasp_bb")
        self.vision_understand_scene_client = self.create_client(UnderstandScene, "/vision/understand_scene")

        # Track tools called
        self._tools_called: List[str] = []
        self._tools_called_lock = threading.Lock()

        # Initialize LangChain tools and agent
        self.tools = self._initialize_tools()
        self.agent_executor = self._create_pddl_agent_executor()

        # Chat history
        self.chat_history: List[Dict[str, str]] = []  # [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
        self.latest_plan: Optional[List[str]] = None

        # Store PDDL generation results for reference
        self.latest_pddl: Optional[PDDLGenerationResult] = None

        # Create confirmation service
        self.confirm_srv = self.create_service(Trigger, "/confirm", self.confirm_service_callback)

        # High-level action server (Prompt action)
        self._action_server = ActionServer(
            self,
            Prompt,
            "prompt_high_level",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        # Response publisher
        self.response_pub = self.create_publisher(String, "/response", 10)
        self.benchmark_pub = self.create_publisher(String, "/benchmark_logs", 10)

        self.get_logger().info("Ros2 High-Level Agent Node (PDDL) ready.")

    # -----------------------
    # Transcript handling
    # -----------------------
    def transcript_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        with self._last_transcript_lock:
            self._last_transcript = text
        self.get_logger().info(f"Received transcript: {text}")
        plan_thread = threading.Thread(target=self._generate_plan, args=(text,), daemon=True)
        plan_thread.start()

    def _generate_plan(self, instruction_text: str) -> List[str]:
        """
        Generate a plan (PDDL domain/problem -> Fast Downward -> parsed steps) but do NOT execute.
        The plan is stored internally for later confirmation.
        """
        # Add user message to chat history
        self.chat_history.append({"role": "user", "content": instruction_text})

        with self._tools_called_lock:
            self._tools_called = []

        try:
            self.get_logger().info("High-level agent (PDDL): thinking and generating plan...")
            self.response_pub.publish(String(data="Got it! Let me think through that..."))
            self.response_pub.publish(String(data="Analyzing scene and generating PDDL files"))

            # Build LangChain chat history
            langchain_history = []
            for msg in self.chat_history[:-1]:  # Exclude the current message we just added
                if msg["role"] == "user":
                    langchain_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_history.append(AIMessage(content=msg["content"]))

            # Invoke agent with chat history
            agent_resp = self.agent_executor.invoke({
                "input": instruction_text,
                "chat_history": langchain_history
            })

            final_text = agent_resp.get("output") if isinstance(agent_resp, dict) else str(agent_resp)
            self.get_logger().info(f"Agent output (raw):\n{final_text}")

            # Add AI response to chat history
            self.chat_history.append({"role": "assistant", "content": final_text})

            self.response_pub.publish(String(data="I'm generating the PDDL files now"))

            # Extract domain and problem from the LLM output
            pddl_gen = self._parse_pddl_from_text(final_text)
            if pddl_gen is None:
                msg = "Hmm... I couldn't generate valid PDDL files. Could you try rephrasing that?"
                self.get_logger().error("Failed to parse PDDL domain/problem from LLM output.")
                self.response_pub.publish(String(data=msg))
                return []

            # Store PDDL for reference
            self.latest_pddl = pddl_gen

            # Save PDDL to temporary directory
            tmpdir = tempfile.mkdtemp(prefix="pddl_")
            domain_path = Path(tmpdir) / "domain.pddl"
            problem_path = Path(tmpdir) / "problem.pddl"
            domain_path.write_text(pddl_gen.domain_pddl)
            problem_path.write_text(pddl_gen.problem_pddl)
            self.get_logger().info(f"PDDL files saved to {tmpdir}")
            self.get_logger().debug(f"Domain:\n{pddl_gen.domain_pddl}")
            self.get_logger().debug(f"Problem:\n{pddl_gen.problem_pddl}")

            # Run Fast Downward
            self.response_pub.publish(String(data="I'm solving the PDDL problem using Fast Downward"))
            plan_result = self._run_fast_downward(str(domain_path), str(problem_path))
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

            # Store the plan
            self.latest_plan = plan_lines

            # Present plan to user for confirmation
            readable_plan = "\n".join([f"{i+1}. {s}" for i, s in enumerate(plan_lines)])
            self.response_pub.publish(String(
                data=f"Here's what I plan to do:\n{readable_plan}\n\nPlease review and confirm if this looks good!"
            ))
            self.get_logger().info(f"Generated plan with {len(plan_lines)} steps, waiting for /confirm.")
            return plan_lines

        except Exception as e:
            self.get_logger().error(f"Error generating plan: {e}")
            self.response_pub.publish(String(data="Sorry, something went wrong while planning."))
            return []

    def confirm_service_callback(self, request, response):
        """
        When the user confirms, execute the latest plan step-by-step.
        Clears chat history and latest plan after execution.
        """
        if not self.latest_plan:
            response.success = False
            response.message = "No plan to confirm. Please give a new instruction first."
            self.response_pub.publish(String(data=response.message))
            return response

        # Execute the plan in a separate thread to avoid blocking the service callback
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

        # Start execution in background thread
        execution_thread = threading.Thread(target=execute_plan, daemon=True)
        execution_thread.start()

        # Return immediately from service call
        response.success = True
        response.message = "Plan execution started."
        return response

    # -----------------------
    # PDDL parsing & Fast Downward helpers
    # -----------------------
    def _parse_pddl_from_text(self, text: str) -> Optional[PDDLGenerationResult]:
        """
        Attempt to extract DOMAIN and PROBLEM code blocks from LLM output.
        """
        try:
            # Try to find code fences first (```pddl ... ```)
            domain_match = re.search(r"DOMAIN:\s*```pddl\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            problem_match = re.search(r"PROBLEM:\s*```pddl\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if domain_match and problem_match:
                domain = domain_match.group(1).strip()
                problem = problem_match.group(1).strip()
                reasoning_match = re.search(r"REASONING:\s*(.*?)(?=DOMAIN:|PROBLEM:|$)", text, re.DOTALL | re.IGNORECASE)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                return PDDLGenerationResult(domain_pddl=domain, problem_pddl=problem, reasoning=reasoning)

            # Fallback: use 'DOMAIN:' and 'PROBLEM:' split without code fences
            if "DOMAIN:" in text and "PROBLEM:" in text:
                domain_part = text.split("DOMAIN:")[1].split("PROBLEM:")[0].strip()
                problem_part = text.split("PROBLEM:")[1].strip()
                # Remove surrounding backticks if present
                domain = domain_part.strip("` \n")
                problem = problem_part.strip("` \n")
                reasoning_match = re.search(r"REASONING:\s*(.*?)(?=DOMAIN:|PROBLEM:|$)", text, re.DOTALL | re.IGNORECASE)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                return PDDLGenerationResult(domain_pddl=domain, problem_pddl=problem, reasoning=reasoning)

            # Last resort: try finding parenthesis blocks typical of PDDL (heuristic)
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
        """
        Call Fast Downward to produce a plan.
        """
        workdir = str(Path(domain_file).parent)
        cmd = ["python3", FAST_DOWNWARD_PY, domain_file, problem_file, "--search", "astar(lmcut())"]
        self.get_logger().info(f"Calling Fast Downward: {' '.join(cmd)} (workdir={workdir})")
        try:
            result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)
            plan_text = ""
            sas_plan_path = Path(workdir) / "sas_plan"
            self.get_logger().info(f"Fast Downward stdout:\n{result.stdout}")
            self.get_logger().info(f"Plan file path: {sas_plan_path}")
            if sas_plan_path.exists():
                self.get_logger().info("Plan file found, reading...")
                plan_text = sas_plan_path.read_text()
            status = "success" if result.returncode == 0 else "failed"
            return PlanningResult(status=status, return_code=result.returncode, stdout=result.stdout, stderr=result.stderr, plan=plan_text)
        except subprocess.TimeoutExpired as e:
            return PlanningResult(status="timeout", return_code=-1, stdout=str(e.stdout or ""), stderr=str(e.stderr or ""))
        except Exception as e:
            return PlanningResult(status="error", return_code=-1, stdout="", stderr=str(e))

    def _parse_plan_text(self, plan_text: str) -> List[str]:
        """
        Parse plan text (sas_plan) into a list of textual steps.
        """
        lines = []
        for ln in plan_text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(";"):  # comment lines
                continue
            # Remove cost annotations in brackets at line end
            ln = re.sub(r"\s*\[.*?\]\s*$", "", ln)
            ln = re.sub(r"\s*\(cost.*?\)\s*$", "", ln, flags=re.IGNORECASE)
            # Remove surrounding parentheses if present
            if ln.startswith("(") and ln.endswith(")"):
                ln = ln[1:-1].strip()
            # Normalize whitespace and return
            ln = " ".join(ln.split())
            if ln:
                lines.append(ln)
        return lines

    # -----------------------
    # Tools (LangChain wrappers)
    # -----------------------
    def _initialize_tools(self) -> List[BaseTool]:
        tools: List[BaseTool] = []

        @tool
        def detect_objects(image_hint: Optional[str] = "") -> str:
            """
            Call /vision/detect_objects (DetectObjects.srv) which returns bounding boxes and meta info.
            Returns a short textual summary with counts and first few bboxes.
            """
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
            """
            Trigger /vision/classify_all (std_srvs/Trigger) to classify entire frame or all detections.
            """
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
        def classify_bb(x1: int, y1: int, x2: int, y2: int) -> str:
            """
            Call /vision/classify_bb with bounding box coordinates.
            Returns the top label + confidence and the raw 'all_predictions' JSON string (truncated).
            """
            tool_name = "classify_bb"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_classify_bb_client.wait_for_service(timeout_sec=5.0):
                    return "Service /vision/classify_bb unavailable"
                req = ClassifyBBox.Request()
                req.x1 = int(x1)
                req.y1 = int(y1)
                req.x2 = int(x2)
                req.y2 = int(y2)
                future = self.vision_classify_bb_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                resp = future.result()
                if resp is None:
                    return "No response from /vision/classify_bb"
                if not resp.success:
                    return f"classify_bb failed: {resp.all_predictions or 'error'}"
                allpred = resp.all_predictions or ""
                if len(allpred) > 400:
                    allpred_trunc = allpred[:400] + "...(truncated)"
                else:
                    allpred_trunc = allpred
                return f"classify_bb: label='{resp.label}', confidence={resp.confidence:.3f}, all_predictions={allpred_trunc}"
            except Exception as e:
                return f"ERROR in classify_bb: {e}"

        tools.append(classify_bb)

        @tool
        def detect_grasp() -> str:
            """
            Call /vision/detect_grasp to compute grasps for all detected objects.
            Returns a short summary describing how many grasps were found and top qualities.
            """
            tool_name = "detect_grasp"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_detect_grasp_client.wait_for_service(timeout_sec=5.0):
                    return "Service /vision/detect_grasp unavailable"
                req = DetectGrasps.Request()
                future = self.vision_detect_grasp_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                resp = future.result()
                if resp is None:
                    return "No response from /vision/detect_grasp"
                if not resp.success:
                    return f"detect_grasp failed: {resp.error_message or 'unknown'}"
                total = int(resp.total_grasps)
                qualities = []
                try:
                    for i in range(min(3, len(resp.grasp_poses))):
                        qualities.append(f"{resp.grasp_poses[i].quality_score:.3f}")
                except Exception:
                    pass
                qual_summary = ", ".join(qualities) if qualities else "no quality info"
                return f"detect_grasp: total_grasps={total}, sample_qualities=[{qual_summary}]"
            except Exception as e:
                return f"ERROR in detect_grasp: {e}"

        tools.append(detect_grasp)

        @tool
        def detect_grasp_bb(x1: int, y1: int, x2: int, y2: int) -> str:
            """
            Call /vision/detect_grasp_bb to compute a single grasp pose for the specified bounding box.
            Returns a compact textual description of the returned GraspPose.
            """
            tool_name = "detect_grasp_bb"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_detect_grasp_bb_client.wait_for_service(timeout_sec=5.0):
                    return "Service /vision/detect_grasp_bb unavailable"
                req = DetectGraspBBox.Request()
                req.x1 = int(x1)
                req.y1 = int(y1)
                req.x2 = int(x2)
                req.y2 = int(y2)
                future = self.vision_detect_grasp_bb_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                resp = future.result()
                if resp is None:
                    return "No response from /vision/detect_grasp_bb"
                if not resp.success:
                    return f"detect_grasp_bb failed: {resp.error_message or 'unknown'}"
                gp = resp.grasp_pose
                pos = gp.position
                ori = gp.orientation
                return (f"grasp_bb: object_id={gp.object_id}, bbox={list(gp.bbox)}, "
                        f"pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f}), "
                        f"ori=({ori.x:.3f},{ori.y:.3f},{ori.z:.3f},{ori.w:.3f}), "
                        f"quality={gp.quality_score:.3f}, width={gp.width:.3f}, approach={gp.approach_direction}")
            except Exception as e:
                return f"ERROR in detect_grasp_bb: {e}"

        tools.append(detect_grasp_bb)

        @tool
        def understand_scene() -> str:
            """
            Call /vision/understand_scene which returns a SceneUnderstanding message.
            We extract a short natural-language summary and a few stats for the LLM.
            """
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
    # Create PDDL-generating agent
    # -----------------------
    def _create_pddl_agent_executor(self) -> AgentExecutor:
        """
        Create an agent whose job is to generate valid PDDL domain/problem given
        a natural language instruction and optionally using the vision tools.
        """
        system_message = (
            "You are a PDDL domain and problem generator for a robot planning system.\n"
            "You have access to tools that query vision"
            "capabilities (detect_objects, classify_all, classify_bb, detect_grasp, detect_grasp_bb, understand_scene).\n"
            "Your only task is to produce two valid PDDL files, one for DOMAIN and one for PROBLEM, "
            "that together describe how the robot should solve the given task.\n\n"
            "Assume a simple robot with any capabilities necessary. "
            "Always assume that the robot initial state is not the same as the target state.\n"
            "The PDDL file should be as simple as possible. \n"
            "Follow this format exactly:\n"
            "REASONING:\n[explain assumptions]\n\n"
            "DOMAIN:\n```pddl\n[domain content]\n```\n\n"
            "PROBLEM:\n```pddl\n[problem content]\n```\n"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=12)

    # -----------------------
    # Action server callbacks (high-level)
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
        """
        Execute incoming high-level Prompt action by running the PDDL pipeline.
        Publishes periodic feedback with tools called.
        """
        start_time = time.perf_counter()
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

                # Generate plan but do not execute
                steps = self._generate_plan(goal_text)
                if not steps:
                    result_container["success"] = False
                    result_container["final_response"] = "Failed to generate plan"
                    return

                # Wait for confirmation
                msg = f"Generated {len(steps)} step(s). Please review and confirm via /confirm to execute."
                result_container["success"] = True
                result_container["final_response"] = msg
            except Exception as e:
                self.get_logger().error(f"Exception in action pipeline: {e}")
                result_container["success"] = False
                result_container["final_response"] = f"Error: {e}"

        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()

        # Publish periodic feedback
        while thread.is_alive():
            with self._tools_called_lock:
                tools_snapshot = list(self._tools_called)
            feedback_msg.tools_called = tools_snapshot
            try:
                goal_handle.publish_feedback(feedback_msg)
            except Exception:
                pass
            time.sleep(0.5)

        # final feedback
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
        end_time = time.perf_counter()
        benchmark_info = f"High-level action completed in {end_time - start_time:.2f} seconds.\n Number of tools called: {len(tools_snapshot)}"
        self.benchmark_pub.publish(String(data=benchmark_info))
        return result_msg

    # -----------------------
    # Helpers: send step and return result object
    # -----------------------
    def send_step_to_medium(self, step_text: str, timeout: float = 60.0) -> Optional[Prompt.Result]:
        """
        Synchronous helper: sends a step to the /medium_level Prompt action server and waits for the result.
        """
        try:
            if not self.medium_level_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("/medium_level action server unavailable")
                return None
            goal = Prompt.Goal()
            goal.prompt = step_text
            send_future = self.medium_level_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Medium-level goal rejected")
                return None
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
            result = result_future.result().result
            return result
        except Exception as e:
            self.get_logger().error(f"Exception when sending to medium: {e}")
            return None

    def send_step_to_medium_async(self, step_text: str, timeout: float = 60.0) -> Optional[Prompt.Result]:
        """
        Thread-safe version that uses threading.Event instead of spin_until_future_complete.
        """
        try:
            if not self.medium_level_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("/medium_level action server unavailable")
                return None
            
            goal = Prompt.Goal()
            goal.prompt = step_text
            
            # Use events to wait for async operations
            goal_event = threading.Event()
            result_event = threading.Event()
            goal_handle_container = [None]
            result_container = [None]
            
            # Callback for goal response
            def goal_response_callback(future):
                goal_handle_container[0] = future.result()
                goal_event.set()
            
            # Callback for result
            def result_callback(future):
                result_container[0] = future.result()
                result_event.set()
            
            # Send goal
            send_future = self.medium_level_client.send_goal_async(goal)
            send_future.add_done_callback(goal_response_callback)
            
            # Wait for goal acceptance
            if not goal_event.wait(timeout=5.0):
                self.get_logger().error("Timeout waiting for goal acceptance")
                return None
            
            goal_handle = goal_handle_container[0]
            if not goal_handle.accepted:
                self.get_logger().error("Medium-level goal rejected")
                return None
            
            # Get result
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(result_callback)
            
            # Wait for result
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
        node.get_logger().info("Shutting down Ros2 High-Level Agent Node (PDDL).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()