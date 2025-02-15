"""
This file contains the Environment class, which prepares the environment for the research agent to run in.
"""

import json
import os
import sys
import subprocess
import shutil
import copy
import time
import fnmatch
import signal
from traceback import format_exception
from multiprocessing import active_children
import readline # to make sure input() works properly
from dacite import from_dict

from .low_level_actions import LOW_LEVEL_ACTIONS
from .high_level_actions import HIGH_LEVEL_ACTIONS
from .schema import Step, Trace, EnvException, TooLongPromptError, LLMError, EnhancedJSONEncoder, Action
# from .LLM import complete_text_claude
from .prepare_task import prepare_task, get_task_info
from .agents import PlannerAgent, InstructorAgent, ReasoningActionAgent
from .agents.actions_planner import PLANNER_ACTIONS
from .agents.actions_instructor import INSTRUCTOR_ACTIONS





class TimeoutException(Exception): pass


class Environment:
    def __init__(self, args):

        self._args = args
        self._log_dir = os.path.join(args.log_dir, "env_log")
        self._setup_log_dir()

        if not args.interactive:
            self._benchmark_folder_name, self._research_problem = get_task_info(args.task)
            self._work_dir = os.path.join(args.work_dir, self.benchmark_folder_name)
            self._read_only_files = []
            self._initialize_task_env() # set up work dir and log dir

        self._action_infos = {t.name: t for t in LOW_LEVEL_ACTIONS + HIGH_LEVEL_ACTIONS + PLANNER_ACTIONS + INSTRUCTOR_ACTIONS}
        self._action_infos_dict = {}

    

        self._static_kwargs_for_tools = {
            "device": args.device,
            "python": args.python,
            "work_dir": self.work_dir,
            "args": args,
            "read_only_files": self.read_only_files,
            "research_problem": self.research_problem,
        }

        self._pause = False

        self._trace = self._initialize_trace()
        if not self.args.resume:
            self.save(0)

        self._start_time = time.time()

        self._idea_map = {}

    ############################## getters ########################################

    @property
    def research_problem(self):
        return self._research_problem

    @property
    def benchmark_folder_name(self):
        return self._benchmark_folder_name

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def work_dir(self):
        return self._work_dir
    
    @property
    def read_only_files(self):
        return self._read_only_files

    @property
    def action_infos(self):
        return self._action_infos
    
    @property
    def action_infos_dict(self):
        return self._action_infos_dict
    
    @property
    def args(self):
        return self._args

    @property
    def static_kwargs_for_tools(self):
        return self._static_kwargs_for_tools
    
    @property
    def trace(self):
        return copy.deepcopy(self._trace)

    @property
    def start_time(self):
        return self._start_time
    
    ############################## internal functions ########################################
    
    def _setup_log_dir(self):
        # set up log dir
        if os.path.exists(self.args.log_dir):
            print("log_dir {} already exists".format(self.log_dir))
        else:
            os.makedirs(self.log_dir)

        if os.path.exists(os.path.join(self.log_dir, "tool_logs")):
            print("tools_log_dir {} already exists".format(os.path.join(self.log_dir, "tool_logs")))
            # raise ValueError("log_dir {} already exists".format(self.log_dir))
        else:
            os.makedirs(os.path.join(self.log_dir, "tool_logs"))

        if os.path.exists(os.path.join(self.log_dir, "traces")):
            print("tools_log_dir {} already exists".format(os.path.join(self.log_dir, "traces")))
            # raise ValueError("log_dir {} already exists".format(self.log_dir))
        else:
            os.makedirs(os.path.join(self.log_dir, "traces"))

    def _initialize_task_env(self):

        work_dir = self.work_dir

        # remove the workspace folder if it exists
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

        benchmark_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "benchmarks", self.benchmark_folder_name)


        tool_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "agent_tools")
        workdir_parent = os.path.dirname(self.work_dir)  # Get parent directory of work_dir
        destination_dir = os.path.join(workdir_parent, "agent_tools")

        try:
            # Copy the directory
            shutil.copytree(tool_dir, destination_dir, dirs_exist_ok=True)
            print(f"Successfully copied '{tool_dir}' to '{destination_dir}'.")
        except Exception as e:
            print(f"Error copying directory: {e}")



        # prepare if there is a prepare.py and it has not been prepared
        prepare_task(benchmark_dir, self.args.python)

        # copy the benchmarks folder to work_dir
        if os.path.exists(os.path.join(benchmark_dir, "env" )):
            shutil.copytree(os.path.join(benchmark_dir, "env"), work_dir, symlinks=True)

        # find all read only files
        if os.path.exists(os.path.join(benchmark_dir, "scripts", "read_only_files.txt")):
            ignore_files = open(os.path.join(benchmark_dir, "scripts", "read_only_files.txt"), "r").read().split("\n")
            for path, subdirs, files in os.walk(os.path.join(work_dir)):

                relpath = os.path.relpath(path, work_dir)
                # filter out the files that are read only
                filenames = [os.path.join(relpath, filename) for filename in files]
                for ignore in ignore_files:
                    ignore_filenames = [n for n in filenames if fnmatch.fnmatch(n, ignore)]
                    self.read_only_files.extend(ignore_filenames)


        # init backup folder and remove all content if it exists
        if os.path.exists(os.path.join(work_dir, "backup")):
            shutil.rmtree(os.path.join(work_dir, "backup"))
        os.mkdir(os.path.join(work_dir, "backup"))

        if self.args.resume:
            shutil.rmtree(work_dir)
            resume_dir = os.path.join(self.args.resume, "env_log", "traces" , f"step_{self.args.resume_step}_files")
            print("Restoring workspace ing from {}".format(resume_dir))
            shutil.copytree(resume_dir, work_dir, symlinks=True)
            if not os.path.exists(os.path.join(work_dir, "backup")):
                os.mkdir(os.path.join(work_dir, "backup"))


    def _initialize_trace(self):
        if self.args.resume:
            print("Restoring trace from {}".format(self.args.resume))
            prev_trace = from_dict(data_class=Trace, data=json.load(open(os.path.join(self.args.resume, "env_log","trace.json"), "r")))
            print("Resetting trace to step {}".format(self.args.resume_step))
            steps = prev_trace.steps[:self.args.resume_step+1]
            t = steps[-1].timestamp
            low_level_steps = [s for s in prev_trace.low_level_steps if s.timestamp < t]
            trace = Trace(
                steps=steps,
                low_level_steps=low_level_steps,
                action_infos=self.action_infos,
                task_description=self.research_problem,
            )
        else:   
            trace = Trace(
            steps=[],
            low_level_steps=[],
            action_infos=self.action_infos,
            task_description=self.research_problem,
            )
            initial_step_time = time.time()
            initial_step = Step(
                action=Action(name="initiaize", args={}),
                observation="Initial environment state.",
                timestamp=initial_step_time
            )
            trace.steps.append(initial_step)
        return trace
    

    
    def __enter__(self):
        # set time out
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.args.max_time)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):  
        # save error message
        active = active_children()
        print(f'Active Children: {len(active)}')
        # terminate all active children
        for child in active:
            child.terminate()
        # block until all children have closed
        for child in active:
            child.join()
        # report active children
        active = active_children()
        print(f'Active Children: {len(active)}')
            
        if traceback is not None:
            print("Error message saved in error.txt")
            open(os.path.join(self.log_dir, "error.txt"), "w").write(''.join(format_exception(exc_type, exc_value, traceback)))
        open(os.path.join(self.log_dir, "overall_time.txt"), "w").write(str(time.time() - self.start_time))
    
            
    ################################# public functions ########################################
    def create_instructor_agent(self, idea, initial_context=""):
        name = f"Instructor-{len(self._idea_map)}"
        planner_actions = [i.name for i in PLANNER_ACTIONS]
        self._action_infos_dict[name] = {key: item for key, item in self.action_infos.items() if key not in planner_actions}
        return InstructorAgent(self._args, self, name, idea, initial_context)


    def create_instructor_agent_and_run(self, idea_id, idea, initial_context):
        prev_step_index = len(self._trace.steps) - 1
        instructor_agent = self.create_instructor_agent(idea, initial_context)
        action, action_input = instructor_agent.run(env=self)
        new_step_index = len(self._trace.steps) - 1
        if action == "Report Failure":
            reason = action_input.get("failure_description", "an unspecified reason")
            observation = f"""
            The Instructor is not able to implement this idea. Please see the attached failure report:{reason}.
            """
        elif action == "Draft Answer":
            answer = action_input.get("answer_file", "unknown file")
            metric = action_input.get("metric", "unknown metric")
            report = action_input.get("report", "no report available")
            observation = f"""
            The Instructor has implemented a draft for the idea. Please continue use idea id {idea_id} to reference this idea. Its final answer file is stored in {answer}, with a metric of {metric}, and here is a detailed report:
            {report}.
            Note that the environment is restored to the state before any instructor edit. Please proceed to explore the next idea or select a final answer if terminate requirement is fulfilled.
            """
        else:
            observation = "The Instructor failed to explore the idea."


        print("The instructor returned after " + str(new_step_index - prev_step_index) +  " steps")
        self._idea_map[idea_id] = new_step_index
        self.restore_workspace(prev_step_index)
        return observation



    def restore_workspace(self, prev_step_index):
        """
        Restores the workspace to the previous state before the latest edits.
        """
        print("Restoring environment to previous state...")

        work_dir = self.work_dir
        backup_dir = os.path.join(work_dir, "backup")

        normalized_read_only_files = {os.path.normpath(f) for f in self.read_only_files}

        # Remove all files and directories except `backup` and `read_only_files`
        for root, dirs, files in os.walk(work_dir, topdown=True):
            rel_root = os.path.relpath(root, work_dir)  # Get relative path of the current folder

            for file in files:
                rel_path = os.path.normpath(os.path.join(rel_root, file))
                if rel_path not in normalized_read_only_files:
                    os.remove(os.path.join(root, file))

            # Remove directories that are not in read-only files
            dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(rel_root, d)) in normalized_read_only_files]

        # Define the resume directory
        resume_dir = os.path.join(self.log_dir, "traces", f"step_{prev_step_index}_files")

        print(f"Restoring workspace from {resume_dir}")

        # Copy all files from resume_dir back to work_dir
        for item in os.listdir(resume_dir):
            src_path = os.path.join(resume_dir, item)
            dst_path = os.path.join(work_dir, item)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, symlinks=True, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)

    def run(self):
        planner_actions = ["Reflection", "Generate Idea", "Understand File", "List Files", "Final Answer", "Investigate Idea", "Report Failure"]
        self._action_infos_dict["Planner"] = {key: item for key, item in self.action_infos.items() if key in planner_actions}
        planner_agent = PlannerAgent(self._args, self, "Planner")
        planner_agent.run(env=self)
        

    def is_final(self):
        """Check if the task has reached a final state, either by reaching the maximum steps or time, or because the agent has submitted a final answer. """
        
        curr_step = len(self.trace.steps)
        # check if any step is final answer
        any_final_answer = any([s.action.name == "Final Answer" or s.action.name == "Report Failure" for s in self.trace.steps])
        return curr_step >= self.args.max_steps or any_final_answer or time.time() - self.start_time > self.args.max_time


    def restore_final_answer_env(self, idea_id, **kwargs):
        final_index =  self._idea_map[idea_id]
        self.restore_workspace(final_index)
        # TODO refine the idea
        # print("Start refining idea for final answer...")
        # instructor = self._idea_map[idea_id]["agent"]
        # instructor.refine(env=self)
   


    def execute(self, action):
        """Execute an action and return the observation."""
        
        trace = self._trace

        curr_step = len(trace.steps)
        action_name = action.name
        action_input = action.args

        if action_name == "Final Answer":
            # TODO: Temp code for testing, need check similar to the else block
            if isinstance(action_input, dict):
                self.restore_final_answer_env(**action_input)
            observation = "end"

        elif action_name == "Report Failure":
            observation = "end"

        elif action_name == "Investigate Idea":
            #TODO Temp code for testing, need check similar to the else block
            try:
                observation = self.create_instructor_agent_and_run(**action_input)
            except Exception as e:
                # should not happen
                print("Step: ", curr_step, file=sys.stderr)
                print(e, file=sys.stderr)
                if "Connection aborted." in str(e):
                    raise Exception("Connection aborted for crfm")
                print(action)
                print(action_input)
                observation = f"EnvError: Error executing {action_name}. Error message: {str(e)}"

        elif self.is_final():
            observation = "The environment has shut down because the maximum number of steps or time has been reached. Please submit your final answer."

        elif action_name not in list(self.action_infos.keys()):
            actions = ", ".join(self.action_infos.keys())
            observation = f"Invalid action: {action_name}. Action did not execute. Please use one of the following actions:\n{actions}"

        else:
            # execute the action and get the observation
            log_file = os.path.join(os.path.join(self.log_dir, "tool_logs") , f"step_{curr_step}_tool_log.log")
            usage = ",\n            ".join([f"{k}: [{v}]" for k, v in self.action_infos[action_name].usage.items()])
            usage = f"""{{
            {usage}
}}"""
            invalid_action_error = f"The action input for {action_name} needs to be a valid json with proper entries. You may have missed the comma between entries. Please use the correct format and try again:\n{usage}"

            if isinstance(action_input, dict):
                try:
                    observation = self.action_infos[action_name].function(**action_input, log_file=log_file, trace=trace, **self.static_kwargs_for_tools)
                except TooLongPromptError:
                    observation="EnvError: too long input for the tool"
                except LLMError as e:
                    observation = "LLMError: " + e.message
                except EnvException as e:
                    observation = "EnvError: " + e.message
                except TypeError as e:
                    print("Step: ", curr_step, file=sys.stderr)
                    print(e, file=sys.stderr)
                    print(action_input, file=sys.stderr)
                    observation = "EnvError: " + invalid_action_error
                except TimeoutException as e:
                    raise e
                except Exception as e:
                    # should not happen
                    print("Step: ", curr_step, file=sys.stderr)
                    print(e, file=sys.stderr)
                    if "Connection aborted." in str(e):
                        raise Exception("Connection aborted for crfm")
                    print(action)
                    print(action_input)
                    observation = f"EnvError: Error executing {action_name}. Error message: {str(e)}"
            else:
                observation = invalid_action_error


        step_time = time.time()
        curr_step = len(trace.steps) # TODO: Temp fix - another agent may change trace steps

        trace.steps.append(Step(action, observation, step_time))
        self.save(curr_step)
        return observation

    def save(self, curr_step):
        """ Save the trace and snapshot of the workspace folder """     
        with open(os.path.join(self.log_dir, f"trace.json"), "w") as f:
            json.dump(self.trace, f, indent=4, cls=EnhancedJSONEncoder)

        ##### save a snapshot of the current step
        save_folder = os.path.join(self.log_dir, f"traces/step_{curr_step}_files")
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

        # save files in the folder that are not read only
        for path, subdirs, files in os.walk(os.path.join(self.work_dir)):

            relpath = os.path.relpath(path, self.work_dir)
            dest = os.path.join(save_folder, relpath)

            for file_name in files:
                file_path = os.path.join(relpath, file_name)
                if file_path not in self.read_only_files:
                    # check wether the file to copy is part of self.log_dir
                    if  os.path.abspath(os.path.join(self.work_dir, file_path)).startswith(os.path.abspath(self.log_dir.split("/env_log")[0])):
                        continue                    
                    if not os.path.exists(dest):
                        os.makedirs(dest)            
                    shutil.copyfile(os.path.join(self.work_dir, file_path), os.path.join(save_folder, file_path))

    ############## for logging convenience ##############

    def get_task_description(self):
        return self.research_problem, self.benchmark_folder_name

    @property
    def low_level_actions(self):
        return list(filter(lambda x: x.is_primitive, self.action_infos.values()))

    @property
    def high_level_actions(self):
        return list(filter(lambda x: not x.is_primitive, self.action_infos.values()))

    def print_action(self, entries):
        return "".join([ k + ": " + v for k,v in  entries.items()])