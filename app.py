from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from deepagents import create_deep_agent
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ---- LLMs ----
# Sonnet 4.5 – main workhorse
claude = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0.1,
    max_tokens=1024,
)

# Small, fast model – great for quick explainer / breadth scans
haiku_model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",  # <-- corrected to match console
    temperature=0.3,
    max_tokens=1024,
)

# Heavy research / deep reasoning model – “research” role
opus_model = ChatAnthropic(
    model="claude-opus-4-5-20251101",
    temperature=0.2,
    max_tokens=2048,
)

print(f"[BOOT] Using Anthropic model: {claude.model!r}")


# ---- Tools ----


@tool
def web_search(query: str) -> str:
    """Search the web for info on a query (mocked)."""
    return f"Mock results for '{query}': key facts from recent sources."


# ---- Deep Agent (single generalist) ----

main_agent = create_deep_agent(
    model=claude,
    tools=[web_search],
    system_prompt=(
        "You are a fast research agent. Do NOT overthink. "
        "Use at most 3 TODO steps and respond with a short markdown report."
    ),
)


def sanity_test():
    print("[TEST] Running direct Claude sanity test...")
    resp = claude.invoke(
        "Say 'hello Raj, Claude connection works.' in one short sentence."
    )
    # Only print the text, not the whole metadata blob
    print("[TEST] Claude sanity response:", resp.content)


# ---- Helper: safely extract text from DeepAgents result ----


def extract_output(result) -> str:
    """Try to pull a useful text answer out of a DeepAgents / LangGraph result."""
    output_text = None

    if isinstance(result, dict):
        # Case 1: DeepAgents returns a top-level 'output' string
        if isinstance(result.get("output"), str):
            output_text = result["output"]
        else:
            msgs = result.get("messages")
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]

                # Case 2: LangChain chat message (AIMessage, etc.)
                if hasattr(last, "content"):
                    output_text = last.content
                # Case 3: plain dict with "content"
                elif isinstance(last, dict):
                    output_text = last.get("content")

    if not output_text:
        # Fallback: stringify the entire result
        output_text = str(result)

    return output_text


# ---- Single-agent fast research (your current working flow) ----


def run_research(topic: str):
    task = (
        f"Research: {topic}. "
        f"Give a short, clear markdown report. Do not create TODO lists."
    )

    print("[AGENT] Running fast mode...")

    result = main_agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config={
            "recursion_limit": 12,  # keep planning shallow but not too strict
            # "max_steps": 4,       # you can re-enable if needed
        },
    )

    output_text = extract_output(result)

    # write report
    workspace = Path("agent_workspace")
    workspace.mkdir(exist_ok=True)
    report_path = workspace / "report.md"
    report_path.write_text(f"# Report: {topic}\n\n{output_text}", encoding="utf-8")

    print(f"[AGENT] DONE → {report_path}")


# ---- Multi-agent research: group of agents + supervisor merge ----


def create_specialist_agent(model, role_description: str):
    """
    Create a Deep Agent with a specific perspective / role
    and a specific underlying model (Haiku/Sonnet/Opus).
    """
    return create_deep_agent(
        model=model,
        tools=[web_search],
        system_prompt=(
            "You are a deep research specialist agent. "
            "Collaborate with other agents, but you only see the user query, "
            "not their outputs. "
            f"Your role: {role_description}"
        ),
    )


def run_multi_agent_research(topic: str):
    """
    Orchestrates multiple specialist agents and a supervisor model that merges their reports.
    """
    # 1) Define specialist roles
    # roles = [
    #     ("Explainer", "Focus on fundamentals, definitions, and core concepts."),
    #     ("Skeptic", "Focus on risks, limitations, and open problems."),
    #     ("Pragmatist", "Focus on real-world applications, tools, and examples."),
    # ]
    # 1) Define specialist roles + which model they use
    roles = [
        # Fast breadth scan
        (
            "Explainer",
            "Focus on fundamentals, definitions, and core concepts.",
            haiku_model,
        ),
        # Strong reasoning / analysis
        ("Skeptic", "Focus on risks, limitations, and open problems.", claude),
        # Deep “researchy” applications & trade-offs
        (
            "Pragmatist",
            "Focus on real-world applications, tools, and examples.",
            opus_model,
        ),
    ]

    specialists = []
    for name, desc, model in roles:
        agent = create_specialist_agent(model, desc)
        specialists.append((name, desc, agent))

    # 2) Ask each specialist to produce a sub-report
    subreports = []
    for name, desc, agent in specialists:
        print(f"[AGENT:{name}] Running specialist research...")

        task = (
            f"Topic: {topic}\n"
            f"Perspective: {name} — {desc}\n\n"
            f"Write a markdown report from this perspective. "
            f"Be specific, include bullet points and examples where relevant."
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": task}]},
            config={
                "recursion_limit": 12,  # keep planning shallow but not too strict
                # "max_steps": 4,       # you can re-enable if needed
            },
            # {"messages": [{"role": "user", "content": task}]},
            # You can also pass config here if you want per-agent limits
        )
        text = extract_output(result)
        subreports.append((name, text))

    # 3) Supervisor model merges sub-reports
    print("[SUPERVISOR] Merging specialist reports...")

    merge_prompt = [
        {
            "role": "user",
            "content": (
                f"You are a supervisor model overseeing a team of specialist agents.\n"
                f"The topic is: {topic}\n\n"
                f"You received the following partial reports:\n\n"
                + "\n\n".join(
                    f"=== Report from {name} ===\n{subtext}"
                    for name, subtext in subreports
                )
                + "\n\nUsing ALL of this, write a single, well-structured markdown "
                "report with these sections:\n"
                "- Overview\n"
                "- Key Insights\n"
                "- Trade-offs & Risks\n"
                "- Open Questions\n"
                "- Summary\n\n"
                "Deduplicate overlapping content, resolve contradictions, and keep it crisp."
            ),
        }
    ]

    supervisor_resp = claude.invoke(merge_prompt)
    final_text = (
        supervisor_resp.content
        if hasattr(supervisor_resp, "content")
        else str(supervisor_resp)
    )

    # 4) Write outputs: individual sub-reports + final merged report
    workspace = Path("agent_workspace")
    workspace.mkdir(exist_ok=True)

    # Write individual specialist reports
    for name, text in subreports:
        path = workspace / f"report_{name.lower()}.md"
        path.write_text(f"# {name} Report: {topic}\n\n{text}", encoding="utf-8")
        print(f"[AGENT:{name}] wrote → {path}")

    # Write final merged report
    final_path = workspace / "final_report.md"
    final_path.write_text(f"# Final Report: {topic}\n\n{final_text}", encoding="utf-8")
    print(f"[SUPERVISOR] DONE → {final_path}")


def run_multi_agent_research_parallel(topic: str):
    """
    Orchestrates multiple specialist agents in parallel
    and a supervisor model that merges their reports.
    """
    # 1) Define specialist roles
    # roles = [
    #     ("Explainer", "Focus on fundamentals, definitions, and core concepts."),
    #     ("Skeptic", "Focus on risks, limitations, and open problems."),
    #     ("Pragmatist", "Focus on real-world applications, tools, and examples."),
    # ]
    roles = [
        # Fast breadth scan
        (
            "Explainer",
            "Focus on fundamentals, definitions, and core concepts.",
            haiku_model,
        ),
        # Strong reasoning / analysis
        ("Skeptic", "Focus on risks, limitations, and open problems.", claude),
        # Deep “researchy” applications & trade-offs
        (
            "Pragmatist",
            "Focus on real-world applications, tools, and examples.",
            opus_model,
        ),
    ]

    specialists = []
    for name, desc, model in roles:
        agent = create_specialist_agent(model, desc)
        specialists.append((name, desc, agent))

    # Create agents once (cheap but nice to separate concerns)
    # specialists = []
    # for name, desc in roles:
    #     agent = create_specialist_agent(desc)
    #     specialists.append((name, desc, agent))

    # 2) Run all specialists in parallel
    subreports = []

    def run_one_specialist(name: str, desc: str, agent):
        print(f"[AGENT:{name}] Running specialist research...")
        task = (
            f"Topic: {topic}\n"
            f"Perspective: {name} — {desc}\n\n"
            f"Write a markdown report from this perspective. "
            f"Be specific, include bullet points and examples where relevant."
        )
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})
        text = extract_output(result)
        return name, text

    # Use a small thread pool; 3–5 is plenty for external API calls
    with ThreadPoolExecutor(max_workers=len(specialists)) as executor:
        future_to_name = {
            executor.submit(run_one_specialist, name, desc, agent): name
            for name, desc, agent in specialists
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                name, text = future.result()
                subreports.append((name, text))
                print(f"[AGENT:{name}] Done.")
            except Exception as e:
                print(f"[AGENT:{name}] FAILED with error: {e}")
                subreports.append((name, f"(Error running {name} agent: {e})"))

    # after the as_completed loop
    subreports.sort(key=lambda x: ["Explainer", "Skeptic", "Pragmatist"].index(x[0]))

    # 3) Supervisor model merges sub-reports
    print("[SUPERVISOR] Merging specialist reports...")

    merge_prompt = [
        {
            "role": "user",
            "content": (
                f"You are a supervisor model overseeing a team of specialist agents.\n"
                f"The topic is: {topic}\n\n"
                f"You received the following partial reports:\n\n"
                + "\n\n".join(
                    f"=== Report from {name} ===\n{subtext}"
                    for name, subtext in subreports
                )
                + "\n\nUsing ALL of this, write a single, well-structured markdown "
                "report with these sections:\n"
                "- Overview\n"
                "- Key Insights\n"
                "- Trade-offs & Risks\n"
                "- Open Questions\n"
                "- Summary\n\n"
                "Deduplicate overlapping content, resolve contradictions, and keep it crisp."
            ),
        }
    ]

    # supervisor_resp = claude.invoke(merge_prompt)
    supervisor_resp = opus_model.invoke(merge_prompt)

    final_text = (
        supervisor_resp.content
        if hasattr(supervisor_resp, "content")
        else str(supervisor_resp)
    )

    # 4) Write outputs: individual sub-reports + final merged report
    workspace = Path("agent_workspace")
    workspace.mkdir(exist_ok=True)

    # Write individual specialist reports
    for name, text in subreports:
        path = workspace / f"report_{name.lower()}.md"
        path.write_text(f"# {name} Report: {topic}\n\n{text}", encoding="utf-8")
        print(f"[AGENT:{name}] wrote → {path}")

    # Write final merged report
    final_path = workspace / "final_report.md"
    final_path.write_text(f"# Final Report: {topic}\n\n{final_text}", encoding="utf-8")
    print(f"[SUPERVISOR] DONE → {final_path}")


if __name__ == "__main__":
    sanity_test()  # prove Claude works before we touch DeepAgents

    topic = input("Enter research topic: ")

    # 👉 OPTION 1: keep using single-agent fast mode (current behavior)
    # run_research(topic)

    # 👉 OPTION 2: use multi-agent mode instead:
    # run_multi_agent_research(topic)
    # option 3: use multi-agent mode with parallel execution
    run_multi_agent_research_parallel(topic)
