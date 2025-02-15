""" This file contains unique actions for the planner agent. """

import os
import datetime
import shutil
import difflib
from ..low_level_actions import read_file, write_file, append_file
from ..schema import ActionInfo, EnvException
from ..LLM import complete_text_fast, complete_text

from ..retriever import consult_archive_agent_for_tool



def reflect_domain_knowledge(research_problem = "", curr_idea = "",  work_dir = ".", **kwargs):

    try:
        content = read_file("train.py", work_dir=work_dir, **kwargs)
    except:
        content = ""

    analyze_prompt = f"""
    You have been assigned the task of analyzing the following coding problem to identify substeps that require domain-specific knowledge for an LLM Coder.
    If starter code is provided, only include substeps that involve modifications to the starter code to implement the idea.

    Domain knowledge refers to tasks that go beyond the general capabilities of a large language model (LLM). Examples of domain knowledge include:
    1. Downloading raw biological data.
    2. Processing raw biological data (e.g., SMILES strings, amino acid sequences).
    3. Using APIs from domain-specific libraries.

    General computer science knowledge and typical coding tasks do not count as domain knowledge. Examples include:
    1. Implementing and training general machine learning algorithms.
    2. Evaluating standard metrics (e.g., MAE, MSE).
    3. Extracting features from a dataframe.
    4. Analyzing a dataset.

    For each step, label it as either (requires domain knowledge - reason) or (does not require domain knowledge).
    Avoid unnecessary steps and aim to keep the plan clear and concise.

    Task Description: {research_problem}

    Idea Description (High Level): {curr_idea}

    Starter Code: {content}
    """

    response =  complete_text_fast(analyze_prompt, log_file=kwargs["log_file"])
    return response


INSTRUCTOR_ACTIONS = [
    ActionInfo(
        name="Reflect Domain Knowledge",
        description="Consider the steps involved in a research idea that may require specialized domain knowledge.",
        usage={},
        return_value="The outcome will be a reflection on the potential steps.",
        function=reflect_domain_knowledge
    ),
    ActionInfo(
        name="Draft Answer",
        description="Use this action to provide a draft answer to the current task.",
        usage={
            "answer_file": "The name of the code file you wish to submit (e.g., train.py). Use 'N/A' if no runnable implementation is available after code execution.",
            "metric": "The metric from the validation set used to assess the quality of the draft. Include the metric name and value. This should be confirmed via code execution; otherwise, use 'N/A'.",
            "report": "A summary of the steps you have taken to investigate the idea and their corresponding results. You may copy this from the research plan or status field."
        },
        return_value="The outcome will be empty.",
        function=(lambda **kwargs: "")
    ),
    ActionInfo(
        name="Consult Archive Agent",
        description="Ask a coding question to the archive agent, which creates a tool for you. The archive agent has access to specific domain knowledge, which it will use to formulate a response. The knowledge base includes very niche content, such as detailed technical or API documentation, which is typically outside of LLM's expertise.",
        usage={
            "domain_description": "A description of the domain of knowledge. The expert uses this description to perform a similarity check against the available domain descriptions.",
            "name": "The name of the tool for future reference.",
            "question": "A detailed description of the tool's functionality."
        },
        return_value="The outcome will be a report on the tool construction. If successful, you can use the tool's name to access documentation for its use.",
        function=consult_archive_agent_for_tool
    ),
    ActionInfo(
        name="Report Failure",
        description="Use this action to report failure due to a lack of domain knowledge, inability to debug, or insufficient time.",
        usage={
            "failure_description": "A detailed account of the steps you have taken for this idea and the reasons for the failure."
        },
        return_value="The outcome will be empty.",
        function=(lambda **kwargs: ""),
    )
]

