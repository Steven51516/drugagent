import json
import os
import re
# from scrapegraphai.graphs import SmartScraperGraph

from drugagent.LLM import complete_text_fast
from drugagent.utils import extract_jsons

current_script_dir = os.path.dirname(os.path.abspath(__file__))
doc_dir = os.path.join(current_script_dir, 'doc')



# IMPORTANT: As the documentations collected for case study is relatively short, we didn't use RAG based method to save resources. We suggest using the SmartScraperGraph library for RAG if needed.

# doc search function modified from https://github.com/SageMindAI/autogen-agi/blob/master/agents/agent_functions.py

def update_tools_json(tool_data, workdir):
    tools_file = os.path.join(workdir, "tools.json")

    # Check if tools.json exists
    if os.path.exists(tools_file):
        with open(tools_file, 'r') as file:
            try:
                tools = json.load(file)
                if not isinstance(tools, list):
                    tools = []  # Reset to empty list if data is invalid
            except json.JSONDecodeError:
                tools = []  # Reset to empty list if file is not a valid JSON
    else:
        tools = []  # If file does not exist, initialize as an empty list

    tools.append(tool_data)
    with open(tools_file, 'w') as file:
        json.dump(tools, file, indent=4)



ARCHIVE_AGENT_MATCH_DOMAIN_PROMPT = """You are an expert at finding a matching domain based on a DOMAIN_DESCRIPTION. Given the following DOMAIN_DESCRIPTION and list of AVAILABLE_DOMAINS, please respond with a JSON array of the form:

[
    {{
        "domain": <the name of the matching domain file or "None">,
        "domain_description": <the description of the matching domain or "None">
        "analysis": <your analysis of how closesly the domain matches the given DOMAIN_DESCRIPTION>,
        "rating": <the rating of the similarity of the domain from 1 to 10>
    }},
    {{
        "domain": <the name of the matching domain file or "None">,
        "domain_description": <the description of the matching domain or "None">
        "analysis": <your analysis of how closesly the domain matches the given DOMAIN_DESCRIPTION>,
        "rating": <the rating of the similarity of the domain from 1 to 10>
    }},
    ...
]

IMPORTANT: Be very critical about your analysis and ratings. If an important keyword is missing in the domain description, it should be rated low. If the domain description is not very similar to the domain, it should be rated low. If the domain description is not similar to the domain at all, it should be rated very low.

DOMAIN_DESCRIPTION:
---------------
{domain_description}
---------------

QUESTION:
---------------
{question}
---------------

AVAILABLE_DOMAINS:
---------------
{available_domains}
---------------

JSON_ARRAY_RESPONSE:
"""



GENERATE_ANSWER_TEMPLATE = """You are an expert with access to the following document. It may or may not be relevant to answering the question.

Question: {question}

Idea Context: {idea}

Document: {doc}

Your goal is to answer the question based on your knowledge and the information provided in the document. Please format your response in JSON as follows:

{{
    "Import": "How to import this tool?",
    "Parameter": "What are the parameters of the tool if it is a function? Please describe them in plain text, not additional JSON.",
    "Usage": "How to use it? Include input types, output types, and provide an example if applicable. Describe them in plain text, not additional JSON."
}}

*Note*: You are allowed to return only one tool. Choose the most relevant tool based on the research idea instead of using the entire document.
"""

example_json = """
{{
    "Import": "How to import this tool?",
    "Parameter": "What are the parameters of the tool? Describe them in plain text, not additional JSON.",
    "Usage": "How to use it? Include input types, output types, and provide an example if applicable. Describe them in plain text, not additional JSON."
}}
"""


def consult_archive_agent_for_tool(domain_description, name, question, research_problem="", curr_idea="", 
                                    doc_dir=doc_dir, 
                                    work_dir=".", **kwargs):
    
    failure_prompt = f"Domain not found for domain description: {domain_description} and question: {question}. The Archive Agent cannot help on your question."
    # Load the JSON metadata file
    with open(os.path.join(doc_dir, "doc_info.json"), 'r') as f:
        doc_info = json.load(f)

    # Prepare the domain descriptions by reading the actual content from each file
    domain_descriptions = [
        {
            "domain_name": doc.get("name"),
            "domain_description": doc.get("description")
        }
        for doc in doc_info
    ]

    # Build the string of all domain descriptions
    str_desc = "\n".join(
        [f"Domain: {desc['domain_name']}\n\nDescription:\n{'*' * 50}\n{desc['domain_description']}\n{'*' * 50}\n"
         for desc in domain_descriptions]
    )
    
    # Format the prompt for ARCHIVE_AGENT_MATCH_DOMAIN_PROMPT
    find_domain_query = ARCHIVE_AGENT_MATCH_DOMAIN_PROMPT.format(
        domain_description=domain_description,
        available_domains=str_desc,
        question=question
    )

    domain_response = complete_text_fast(find_domain_query, log_file=None)

    try:
        domain_response = extract_jsons(domain_response)
        domain_response = sorted(domain_response, key=lambda x: int(x["rating"]), reverse=True)
        top_domain = domain_response[0]
    except:
        return failure_prompt

    DOMAIN_RESPONSE_THRESHOLD = 5

    if top_domain["rating"] < DOMAIN_RESPONSE_THRESHOLD:
        return failure_prompt
    
    # Retrieve domain content
    domain_name = top_domain["domain"]
    file_path = os.path.join(doc_dir, domain_name)
    
    with open(file_path, 'r') as f:
        file_content = f.read()

    prompt = GENERATE_ANSWER_TEMPLATE.format(question=question, doc=file_content, idea=curr_idea)
    result = complete_text_fast(prompt, log_file=None)

    # Hallucination check
    hallucination_prompt = f"""
    You are a grader assessing whether 1. the retrieved doc is highly relevant to the user question and 2. The answer is grounded in / supported by retrieved facts.

    Question: {question}

    Retrieved Facts:
    {file_content}

    Generated Result:
    {result}

    Give a binary score 'yes' or 'no' without explanation. 'Yes' means that 1. the retrieved doc is highly relevant to the user question and 2. The answer is grounded in / supported by retrieved facts.
    """

    evaluation = complete_text_fast(hallucination_prompt, log_file=None)

    if re.search(r'\bno\b', evaluation):
        return failure_prompt

    for attempt in range(5):
        try:
            tool_data = extract_jsons(result)[0]
            tool_data['name'] = name
            break
        except Exception as e:
            if attempt == 4:
                print("Failed to parse JSON after 5 attempts.")
                return "Failed to create tool due to format error. Please try again."
            
            print("Using LLM to reformat the response...")
            result = complete_text_fast(
                prompt=f"The following response could not be parsed as JSON:\n{result}\n"
                       f"Please format this response as a valid JSON according to this example:\n{example_json}\n"
                       f"Provide only the corrected JSON output.",
                log_file=None
            )

    # Update tools json file with the correct tool data
    update_tools_json(tool_data, work_dir)

    return json.dumps(tool_data, indent=4)




