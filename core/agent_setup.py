from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from core.prompt_templates import get_reasoning_prompt
import sympy as sp
import re


def get_llm(groq_api_key: str):
    return ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)


def get_math_agent(groq_api_key: str):
    llm = get_llm(groq_api_key)
    math_chain = LLMMathChain.from_llm(llm=llm)

    def safe_math_router(question: str) -> str:
        """Route to calculator or symbolic tool based on input."""
        question = question.lower().strip()
        symbolic_keywords = ["differentiate", "derive", "integrate", "simplify", "solve", "expand", "∫", "x", "sin", "cos", "tan"]

        # If any symbolic keyword is present → use SymPy
        symbolic_patterns = [
            r"(differentiate|derive|derivative)",
            r"(integrate|∫|integral)",
            r"(simplify|solve|expand)",
            r"[a-zA-Z]+\s*\*\*\s*\d",  # x**2 or similar
            r"\b(dx|dy)\b"
        ]

        if any(re.search(pattern, question) for pattern in symbolic_patterns):
            return handle_symbolic_math(question)

        # Otherwise, try numeric evaluation
        try:
            return math_chain.run(question)
        except Exception as e:
            return f"❌ Unable to compute. Try rephrasing or use valid numeric math.\\nError: {str(e)}"

    calculator = Tool(
        name="SmartMathSolver",
        func=safe_math_router,
        description=(
            "Smartly routes math problems:\n"
            "- 🔢 Pure numeric → calculator (e.g. '12 + 3 * 2')\n"
            "- ∫ Symbolic → SymPy (e.g. 'differentiate x**2 * sin(x)')"
        )
    )

    agent = initialize_agent(
        tools=[calculator],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        max_iterations = 20,
        handle_parsing_errors=True
    )
    return agent


def get_wikipedia_agent():
    wikipedia = WikipediaAPIWrapper()
    tool = Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Search for general knowledge or definitions using Wikipedia."
    )
    agent = initialize_agent(
        tools=[tool],
        llm=None,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True
    )
    return agent


def get_reasoning_agent(groq_api_key: str):
    llm = get_llm(groq_api_key)
    prompt = get_reasoning_prompt()
    chain = LLMChain(llm=llm, prompt=prompt)
    reasoning_tool = Tool(
        name="ReasoningTool",
        func=chain.run,
        description="Answer logic-based questions with multi-step reasoning."
    )
    agent = initialize_agent(
        tools=[reasoning_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True
    )
    return agent


def handle_symbolic_math(query: str) -> str:
    try:
        query = query.lower().strip()
        x = sp.Symbol('x')

        match = re.search(r'(differentiate|derive) (.+)', query)
        if match:
            expr = sp.sympify(match.group(2))
            return f"$$\\frac{{d}}{{dx}}({sp.latex(expr)}) = {sp.latex(sp.diff(expr, x))}$$"

        match = re.search(r'(integrate|∫) (.+)', query)
        if match:
            expr = sp.sympify(match.group(2))
            return f"$$\\int {sp.latex(expr)} \\, dx = {sp.latex(sp.integrate(expr, x))}$$"

        match = re.search(r'simplify (.+)', query)
        if match:
            expr = sp.sympify(match.group(1))
            return f"$$\\text{{Simplified: }} {sp.latex(sp.simplify(expr))}$$"

        match = re.search(r'solve (.+)', query)
        if match:
            expr = sp.sympify(match.group(1))
            solutions = sp.solve(expr, x)
            sol_latex = ",\\; ".join([f"x = {sp.latex(sol)}" for sol in solutions])
            return f"$$\\text{{Solutions: }} {sol_latex}$$"

        match = re.search(r'expand (.+)', query)
        if match:
            expr = sp.sympify(match.group(1))
            return f"$$\\text{{Expanded: }} {sp.latex(sp.expand(expr))}$$"

        match = re.search(r'limit (.+) as (.+) approaches (.+)', query)
        if match:
            expr_str = match.group(1).strip()
            var_str = match.group(2).strip()
            limit_point = match.group(3).strip()

            var = sp.Symbol(var_str)
            expr = sp.sympify(expr_str)
            limit_value = sp.sympify(limit_point)

            limit_result = sp.limit(expr, var, limit_value)
            return f"$$\\lim_{{{var_str} \\to {limit_point}}} {sp.latex(expr)} = {sp.latex(limit_result)}$$"

        return "❌ Unsupported symbolic operation. Try: differentiate, integrate, simplify, solve, or expand"
    except Exception as e:
        return f"❌ Error in symbolic math: {str(e)}"

