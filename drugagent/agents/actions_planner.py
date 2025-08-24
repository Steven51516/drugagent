""" This file contains unique actions for the planner agent. """

from ..schema import ActionInfo, EnvException
from ..LLM import complete_text_fast, complete_text


PLANNER_ACTIONS = [
    # ActionInfo(
    #     name="Generate Idea",
    #     description="Use this action to generate additional high-level research ideas for a specific problem.",
    #     usage={
    #         "number_of_ideas": "The number of ideas to generate.",
    #         "additional_info": "Additional instructions for idea generation as a single string. This may include: 1. Preferences for the direction of the ideas. 2. Other information that may inform the idea generation process."
    #     },
    #     return_value="The outcome will be a description of all generated ideas.",
    #     function=generate_idea
    # ),
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

