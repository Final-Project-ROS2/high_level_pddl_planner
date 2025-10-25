import subprocess
import re
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# --- New imports for PDDL pipeline ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool

class PDDLGenerationResult(BaseModel):
    domain_pddl: str = Field(description="Generated PDDL domain file")
    problem_pddl: str = Field(description="Generated PDDL problem file")
    reasoning: str = Field(description="Reasoning for the PDDL design")

class PlanningResult(BaseModel):
    status: str
    return_code: int
    stdout: str
    stderr: str
    plan: str = ""
    plan_length: int = 0

class PDDLPlanner:
    """Handles PDDL generation and planning using an LLM + Fast Downward."""

    def __init__(self, api_key: str, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0
        )
        self.agent = self._create_pddl_agent()

    def _create_pddl_agent(self) -> AgentExecutor:
        """Create an LLM agent for generating PDDL domain & problem."""
        system_prompt = """You are a PDDL expert. 
        Given a high-level natural language instruction, generate:
        1. A valid PDDL domain.
        2. A valid PDDL problem.
        Use standard PDDL structure and ensure syntax correctness.
        Provide both inside ```pddl``` code blocks."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        return create_tool_calling_agent(self.llm, [], prompt)

    def generate_pddl(self, instruction: str) -> PDDLGenerationResult:
        """Use the LLM to produce domain/problem PDDL from natural language."""
        query = f"""
        Generate PDDL domain and problem files for this task:
        {instruction}
        Format response as:
        REASONING:
        [reasoning]
        DOMAIN:
        ```pddl
        [domain content]
        ```
        PROBLEM:
        ```pddl
        [problem content]
        ```
        """
        response = self.agent.invoke({"input": query})
        text = response["output"]

        reasoning = re.search(r"REASONING:\s*(.*?)(?=DOMAIN:|$)", text, re.DOTALL)
        domain = re.search(r"DOMAIN:\s*```pddl\s*(.*?)```", text, re.DOTALL)
        problem = re.search(r"PROBLEM:\s*```pddl\s*(.*?)```", text, re.DOTALL)

        if not (domain and problem):
            raise ValueError("Failed to parse PDDL output from LLM.")

        return PDDLGenerationResult(
            domain_pddl=domain.group(1).strip(),
            problem_pddl=problem.group(1).strip(),
            reasoning=reasoning.group(1).strip() if reasoning else ""
        )

    def save_pddl(self, pddl: PDDLGenerationResult, output_dir="pddl") -> Dict[str, str]:
        """Save domain and problem files."""
        path = Path(output_dir)
        path.mkdir(exist_ok=True)
        domain_path = path / "domain.pddl"
        problem_path = path / "problem.pddl"
        with open(domain_path, "w") as f: f.write(pddl.domain_pddl)
        with open(problem_path, "w") as f: f.write(pddl.problem_pddl)
        return {"domain": str(domain_path), "problem": str(problem_path)}

    def run_fast_downward(self, domain_file: str, problem_file: str) -> PlanningResult:
        """Call Fast Downward to generate a plan."""
        cmd = [
            "uv", "run", "./fast-downward/fast-downward.py",
            domain_file,
            problem_file,
            "--search", "astar(lmcut())"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            plan_text = ""
            plan_path = Path("sas_plan")
            if plan_path.exists():
                plan_text = plan_path.read_text().strip()
            return PlanningResult(
                status="success" if result.returncode == 0 else "failed",
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                plan=plan_text,
                plan_length=len([
                    line for line in plan_text.split("\n")
                    if line.strip() and not line.startswith(";")
                ])
            )
        except Exception as e:
            return PlanningResult(status="error", return_code=-1, stdout="", stderr=str(e))

# --- Integration with existing agent node ---
class AIAgentNode(Node):
    def __init__(self):
        super().__init__('ai_agent_node')
        self.planner = PDDLPlanner(api_key=os.getenv("GEMINI_API_KEY"))
        self.action_client = ActionClient(self, MediumLevelAction, '/medium_level')
        self.task_sub = self.create_subscription(String, '/task_instruction', self.handle_instruction, 10)

    def handle_instruction(self, msg):
        instruction = msg.data
        self.get_logger().info(f"Received instruction: {instruction}")

        # Step 1: Generate PDDL
        pddl = self.planner.generate_pddl(instruction)
        files = self.planner.save_pddl(pddl)

        # Step 2: Run planner
        result = self.planner.run_fast_downward(files["domain"], files["problem"])
        if result.status != "success":
            self.get_logger().error(f"Planning failed: {result.stderr}")
            return

        # Step 3: Execute steps via /medium_level
        plan_steps = [
            line.strip().lower()
            for line in result.plan.split("\n")
            if line and not line.startswith(";")
        ]

        for i, step in enumerate(plan_steps):
            self.get_logger().info(f"Executing plan step {i+1}: {step}")
            goal_msg = MediumLevel.Goal()
            goal_msg.command = step
            future = self.action_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future)
            result_future = future.result().get_result_async()
            rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info("Plan execution complete.")
