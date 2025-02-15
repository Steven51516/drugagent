""" This file contains unique actions for the planner agent. """

from ..schema import ActionInfo, EnvException
from ..LLM import complete_text_fast, complete_text

PROMPT_ROUND1 = """
You are a computer science expert tasked with generating a variety of innovative ideas for a machine learning experiment using different computer science methods.

Experiment Description: {query}

Guidelines:
- Present each idea as a clear and concise sentence, emphasizing specific computer science methods or concepts.
- Avoid vague terms and focus on actionable, distinct concepts.
- Each idea should be centered around one concept(e.g., do not merge ideas like ideaA+ideaB; focus on ideaA).
- Limit the number of ideas to {idea_num}.
- Ensure at least one idea is simple to implement.

Additional Instruction: {idea_direction}

Example:
For an experiment on image classification, your ideas might include:
1. "Use CNN."
2. "Fine-tune a pre-trained ResNet model."
"""

PROMPT_ROUND2 = """
You are an expert in computational biology and chemistry. A computer scientist has proposed several computer science concepts that could be applied to a machine learning experiment: "{query}" with the following ideas: {ideas}. Your task is to refine these ideas by incorporating relevant computational biology concepts while ensuring practical feasibility.

Guidelines:
- Merge the computer science concept with the appropriate computational biology concept into a single, cohesive idea.
- Focus on practical connections between raw biological data and the proposed machine learning techniques.
- Keep the phrasing concise and high-level (~10 words), preserving the original format of the idea.
- Provide essential refinements to ensure the idea's feasibility, avoiding excessive detail.
- If an idea is not feasible, reject it and explain why.

Scoring:
- Rate each idea based on:
  - "performance": A score from 1-10, with an explanation of the potential performance.
  - "feasibility": A score from 1-10, indicating the likelihood of successful implementation, assuming the coder may be inexperienced.

Response Format:
For each idea, return the following:
- "idea": The refined idea, combining computer science and computational biology concepts.
- "performance": A score (1-10) with an explanation.
- "feasibility": A score (1-10) with an explanation.

- Use consistent phrasing for ideas requiring similar domain knowledge to avoid confusion.
"""


def generate_idea(number_of_ideas, additional_info, research_problem = "",  **kwargs):
    get_idea_prompt = PROMPT_ROUND1.format(query=research_problem, idea_num=number_of_ideas, idea_direction=additional_info)
    ideas = complete_text_fast(get_idea_prompt, log_file=kwargs["log_file"])
    refine_idea_prompt = PROMPT_ROUND2.format(query=research_problem, ideas=ideas)
    refined_ideas = complete_text_fast(refine_idea_prompt, log_file=kwargs["log_file"])
    return refined_ideas


PLANNER_ACTIONS = [
    ActionInfo(
        name="Generate Idea",
        description="Use this action to generate additional high-level research ideas for a specific problem.",
        usage={
            "number_of_ideas": "The number of ideas to generate.",
            "additional_info": "Additional instructions for idea generation as a single string. This may include: 1. Preferences for the direction of the ideas. 2. Other information that may inform the idea generation process."
        },
        return_value="The outcome will be a description of all generated ideas.",
        function=generate_idea
    ),
    ActionInfo(
        name="Investigate Idea",
        description="Use this action to assign an idea to the instructor. The instructor will then attempt to implement the idea, which may result in success or failure.",
        usage={
            "idea_id": "The ID of the idea for future reference.",
            "idea": "The description of the idea.",
            "initial_context": "Context that may assist the Instructor, such as details from the starter file or the dataset you have examined."
        },
        return_value="The outcome will be a description of the result of the idea investigation.",
        function=(lambda **kwargs: ""),
        is_primitive=True
    ),
    ActionInfo(
        name="Final Answer",
        description="Use this action to submit the final idea that works best among all investigated ideas.",
        usage={
            "idea_id": "The ID of the idea you wish to submit as the final answer."
        },
        return_value="The outcome will be nothing.",
        function=(lambda **kwargs: ""),
    ),
    ActionInfo(
        name="Report Failure",
        description="Use this action to report failure if no valid idea is found when the termination criteria are met.",
        usage={
            "failure_description": "A detailed report of your plan and the reasons for the failure."
        },
        return_value="The outcome will be nothing.",
        function=(lambda **kwargs: ""),
    )
]

