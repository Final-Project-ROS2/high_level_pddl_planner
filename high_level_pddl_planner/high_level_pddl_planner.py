"""
ROS2 High-Level Agent Node (PDDL-based)

Replaces the previous "LLM -> direct steps" workflow with:
LLM (with vision tools) -> generate PDDL domain & problem -> run Fast Downward -> parse plan -> dispatch steps to /medium_level

Keep your vision services & medium-level action server available.
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

from std_srvs.srv import SetBool
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
        self.llm = ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
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
        plan_thread = threading.Thread(target=self._plan_and_dispatch_from_transcript, args=(text,), daemon=True)
        plan_thread.start()

    def _plan_and_dispatch_from_transcript(self, instruction_text: str):
        """
        Full pipeline:
         - Reset tools_called
         - Ask agent (LLM) to generate PDDL domain & problem (agent may call vision tools)
         - Save files to temp dir
         - Run Fast Downward
         - Parse plan -> list of textual steps
         - Dispatch steps to /medium_level synchronously
        """
        with self._tools_called_lock:
            self._tools_called = []

        try:
            self.get_logger().info("PDDL pipeline: requesting PDDL generation from agent...")
            self.response_pub.publish(String(data="Hey there! I'm thinking about how to handle your request"))
            self.response_pub.publish(String(data="Analyzing scene and generating PDDL files"))

            # Prepare input: embed instruction plus minimal context placeholder
            # (If you want, you can first call detect_objects or other vision tools here and
            # include their outputs in the input string. Agent can also call tools itself.)
            agent_resp = self.agent_executor.invoke({"input": instruction_text})
            final_text = agent_resp.get("output") if isinstance(agent_resp, dict) else str(agent_resp)
            self.get_logger().info(f"Agent output (raw):\n{final_text}")
            self.response_pub.publish(String(data="I'm generating the PDDL files now"))

            # Extract domain and problem from the LLM output
            pddl_gen = self._parse_pddl_from_text(final_text)
            if pddl_gen is None:
                self.get_logger().error("Failed to parse PDDL domain/problem from LLM output.")
                self.response_pub.publish(String(data="I failed to extract PDDL. Try again or adjust prompt."))
                return

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
                self.get_logger().warn(f"Planner returned no plan. status={plan_result.status} rc={plan_result.return_code}")
                self.response_pub.publish(String(data=f"I couldn't find a plan."))
                return

            self.get_logger().info("Plan obtained from Fast Downward. Parsing to steps...")
            plan_lines = self._parse_plan_text(plan_result.plan)
            if not plan_lines:
                self.get_logger().warn("No actionable plan lines parsed.")
                self.response_pub.publish(String(data="No actionable steps parsed from plan."))
                return

            self.response_pub.publish(String(data=f"Plan found with {len(plan_lines)} steps. Executing"))
            # Dispatch steps to medium_level
            for i, step in enumerate(plan_lines, start=1):
                start_msg = f"Executing plan step {i}/{len(plan_lines)}: {step}"
                self.response_pub.publish(String(data=start_msg))
                self.get_logger().info(start_msg)
                result = self.send_step_to_medium(step)
                if result is None:
                    err_msg = f"Failed to send step {i}. Aborting."
                    self.response_pub.publish(String(data=err_msg))
                    self.get_logger().error(err_msg)
                    break
                else:
                    if result.success:
                        ok_msg = f"Step {i} finished successfully: {result.final_response}"
                    else:
                        ok_msg = f"Step {i} executed but reported failure: {result.final_response}"
                    self.response_pub.publish(String(data=ok_msg))
                    self.get_logger().info(ok_msg)

            self.response_pub.publish(String(data="Plan execution finished."))

        except Exception as e:
            self.get_logger().error(f"Exception in PDDL planning pipeline: {e}")
            self.response_pub.publish(String(data=f"Planning error: {e}"))

    # -----------------------
    # PDDL parsing & Fast Downward helpers
    # -----------------------
    def _parse_pddl_from_text(self, text: str) -> Optional[PDDLGenerationResult]:
        """
        Attempt to extract DOMAIN and PROBLEM code blocks from LLM output.
        Accepts variants like:
           DOMAIN:
           ```pddl
           ...
           ```
           PROBLEM:
           ```pddl
           ...
           ```
        or plain blocks with 'DOMAIN:' and 'PROBLEM:' markers.
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
        Call Fast Downward to produce a plan. Use FAST_DOWNWARD_PY if provided.
        The plan is typically saved to 'sas_plan' in the working directory (or Fast Downward's dir).
        We use the temp directory (same as domain/problem) for predictable output.
        """
        workdir = str(Path(domain_file).parent)
        # If FAST_DOWNWARD_PY is an absolute/relative path; otherwise try to call installed fast-downward
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
            # Some Fast Downward variants print a plan to stdout if configured; prefer sas_plan
            status = "success" if result.returncode == 0 else "failed"
            return PlanningResult(status=status, return_code=result.returncode, stdout=result.stdout, stderr=result.stderr, plan=plan_text)
        except subprocess.TimeoutExpired as e:
            return PlanningResult(status="timeout", return_code=-1, stdout=str(e.stdout or ""), stderr=str(e.stderr or ""))
        except Exception as e:
            return PlanningResult(status="error", return_code=-1, stdout="", stderr=str(e))

    def _parse_plan_text(self, plan_text: str) -> List[str]:
        """
        Parse plan text (sas_plan) into a list of textual steps.
        Typical sas_plan lines look like:
            (move robot a b) [1]
        We'll strip parentheses, trailing costs, and return readable strings.
        """
        lines = []
        for ln in plan_text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(";"):  # comment lines
                continue
            # Remove cost annotations in brackets at line end, e.g. " [1]" or " (cost 1)"
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

        # Tool to dispatch a step to medium-level planner, waits for completion
        @tool
        def send_to_medium_level(step_text: str, wait_for_result: bool = True) -> str:
            """
            Send a single textual step to the medium-level planner (/medium_level action server, Prompt action).
            If wait_for_result is True, we wait for the medium-level response and return a short summary.
            """
            name = "send_to_medium_level"
            with self._tools_called_lock:
                self._tools_called.append(name)
            try:
                if not self.medium_level_client.wait_for_server(timeout_sec=5.0):
                    return "Medium-level action server /medium_level unavailable"
                goal = Prompt.Goal()
                goal.prompt = step_text
                send_future = self.medium_level_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, send_future)
                goal_handle = send_future.result()
                if not goal_handle.accepted:
                    return "Medium-level goal rejected"
                if wait_for_result:
                    result_future = goal_handle.get_result_async()
                    rclpy.spin_until_future_complete(self, result_future)
                    res = result_future.result().result
                    return f"medium_level result: success={res.success}, response={res.final_response}"
                else:
                    return "Sent step to medium_level (not waiting for result)."
            except Exception as e:
                return f"ERROR in send_to_medium_level: {e}"

        # tools.append(send_to_medium_level)

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
            "You are strictly prohibited from calling any execution or control tools "
            "like send_to_medium_level.\n\n"
            "Your only task is to produce two valid PDDL files, one for DOMAIN and one for PROBLEM, "
            "that together describe how the robot should solve the given task.\n\n"
            "Assume a simple robot with any capabilities necessary. "
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
        return AgentExecutor(agent=create_tool_calling_agent(self.llm, self.tools, prompt),
                            tools=self.tools,
                            verbose=True)

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
        prompt_text = goal_handle.request.prompt
        self.get_logger().info(f"[high-level action] Executing prompt: {prompt_text}")

        feedback_msg = Prompt.Feedback()
        result_container: Dict[str, Any] = {"success": False, "final_response": ""}

        def run_pipeline():
            try:
                # Reuse the transcript pipeline for core logic
                self._plan_and_dispatch_from_transcript(prompt_text)
                result_container["success"] = True
                result_container["final_response"] = "PDDL planning + dispatch finished (or aborted)."
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
