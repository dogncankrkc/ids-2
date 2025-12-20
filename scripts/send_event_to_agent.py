"""
Cloud Analyst Agent Caller (Google ADK)
--------------------------------------
Sends a single IDS incident event (JSON/dict) to a cloud-side
Cybersecurity Analyst AI agent and returns the analysis report.

Required:
- GOOGLE_API_KEY (env)
- google-adk installed

Author: Dogancan Karakoc
"""

from __future__ import annotations

import json
from typing import Dict

from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools import url_context


# =====================================================
# AGENT DEFINITION (Cloud-side Analyst)
# =====================================================

_search_agent = LlmAgent(
    name="IDS_Search_Agent",
    model="gemini-2.5-flash",
    description="Performs optional web searches for threat context.",
    instruction="Use GoogleSearchTool when background info is required.",
    tools=[GoogleSearchTool()],
)

_url_agent = LlmAgent(
    name="IDS_Url_Context_Agent",
    model="gemini-2.5-flash",
    description="Fetches and summarizes content from URLs if provided.",
    instruction="Use UrlContextTool only if URLs are given.",
    tools=[url_context],
)

root_agent = LlmAgent(
    name="IDS_Security_Analyst_Agent",
    model="gemini-2.5-flash",
    description=(
        "Cloud-side AI analyst that interprets IDS outputs generated "
        "by edge-based intrusion detection systems."
    ),
    instruction=(
        "You are a cybersecurity analyst AI.\n\n"
        "Rules:\n"
        "- You do NOT make detection decisions.\n"
        "- You do NOT modify or override classifications.\n"
        "- You do NOT perform intrusion detection.\n\n"
        "You ONLY:\n"
        "- interpret the given IDS event,\n"
        "- explain the attack type technically,\n"
        "- assess severity and potential impact,\n"
        "- propose mitigation and response actions,\n"
        "- generate a concise incident analysis report.\n\n"
        "Input will be a JSON object produced by an edge IDS.\n"
        "Output must be professional, technical, and suitable for "
        "an IEEE paper appendix or SOC report."
    ),
    tools=[
        agent_tool.AgentTool(agent=_search_agent),
        agent_tool.AgentTool(agent=_url_agent),
    ],
)


# =====================================================
# PUBLIC API
# =====================================================

def send_event_to_agent(incident_event: dict) -> str:
    prompt = (
        "Analyze the following EDGE IDS incident event and produce "
        "a structured incident analysis report.\n\n"
        "EDGE EVENT JSON:\n"
        f"{json.dumps(incident_event, indent=2, ensure_ascii=False)}"
    )

    response = root_agent.run(prompt)

    # ADK response object -> text
    if isinstance(response, str):
        return response

    if hasattr(response, "content"):
        return response.content

    return str(response)
