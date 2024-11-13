import glob
import os
import random
import time
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import torch
from rich.progress import track

from Fuzz4All.model import make_model
from Fuzz4All.util.api_request import create_config, request_engine
from Fuzz4All.util.Logger import LEVEL, Logger


class FResult(Enum):
    SAFE = 1  # validation returns okay
    FAILURE = 2  # validation contains error (something wrong with validation)
    ERROR = 3  # validation returns a potential error (look into)
    LLM_WEAKNESS = (
        4  # the generated input is ill-formed due to the weakness of the language model
    )
    TIMED_OUT = 10  # timed out, can be okay in certain targets


# base class file for target, used for user defined system targets
# the point is to separately define oracles/fuzzing specific functions/and usages
# target should be a stateful objects which has some notion of history (keeping a state of latest prompts)
class Target(object):
    def __init__(self, language="c", timeout=10, folder="/", **kwargs):
        self.language = language
        self.folder = folder
        self.timeout = timeout
        self.CURRENT_TIME = time.time()
        # model based variables
        self.batch_size = kwargs["bs"]
        self.temperature = kwargs["temperature"]
        self.max_length = kwargs["max_length"]
        self.device = kwargs["device"]
        self.model_name = kwargs["model_name"]
        self.model = None
        # loggers
        self.g_logger = Logger(self.folder, "log_generation.txt", level=kwargs["level"])
        self.v_logger = Logger(self.folder, "log_validation.txt", level=kwargs["level"])
        # main logger for system messages
        self.m_logger = Logger(self.folder, "log.txt")
        # system messages for prompting
        self.SYSTEM_MESSAGE = None
        self.AP_SYSTEM_MESSAGE = "You are an auto-prompting tool"
        self.AP_INSTRUCTION = (
            "Please summarize the above documentation in a concise manner to describe the usage and "
            "functionality of the target "
        )
        # prompt based variables
        self.hw = kwargs["use_hw"]
        self.no_input_prompt = kwargs["no_input_prompt"]
        self.prompt_used = None
        self.prompt = None
        self.initial_prompt = None
        self.prev_example = None
        # prompt strategies
        self.se_prompt = self.wrap_in_comment(
            "Please create a semantically equivalent program to the previous "
            "generation"
            "Do not give other comments and explaination"
            "Make sure the final code is clean, well-formatted, and ready to run."
        )
        self.m_prompt = self.wrap_in_comment(
            "Please create a mutated program that modifies the previous generation"
            "Do not give other comments and explaination"
            "Make sure the final code is clean, well-formatted, and ready to run."
        )
        self.c_prompt = self.wrap_in_comment(
            "Please combine the two previous programs into a single program"
            "Do not give other comments and explaination"
            "Make sure the final code is clean, well-formatted, and ready to run."
        )
        self.mask_prompt = """
            Please repair and complete the broken code above. Please follow these instructions carefully:
	1.	Analyze the Code: Read through the entire code to understand its purpose and functionality.
	2.	Identify Syntax Errors: Look for any syntax mistakes such as missing brackets, incorrect indentation, or misused language constructs.
	3.	Find Logical Errors: Check for logical flaws that may prevent the code from executing correctly or producing the desired output.
	4.	Complete Incomplete Sections: Fill in any missing parts of the code that are necessary for it to function properly.
	5.	Correct and Complete the Code: Make all the necessary fixes and additions to ensure the code is fully functional.
	6.	Provide Only the Final Code: Output only the repaired and completed code without any explanations, comments, or additional text.
    8.  Clean comments and explaination
    7.  Do not give other comments and explaination
Make sure the final code is clean, well-formatted, and ready to run."""

        self.p_strategy = kwargs["prompt_strategy"]
        # eos based
        self.special_eos = None
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
        if "target_name" in kwargs:
            self.target_name = kwargs["target_name"]

    @staticmethod
    def _create_prompt_from_config(config_dict: Dict[str, Any]) -> Dict:
        """Read the prompt ingredients via a config file."""
        documentation, example_code, hand_written_prompt = None, None, None

        # read the prompt ingredients from the config file
        target = config_dict["target"]
        path_documentation = target["path_documentation"]
        if path_documentation is not None:
            documentation = open(path_documentation, "r").read()
        path_example_code = target["path_example_code"]
        if path_example_code is not None:
            example_code = open(path_example_code, "r").read()
        trigger_to_generate_input = target["trigger_to_generate_input"]
        input_hint = target["input_hint"]
        path_hand_written_prompt = target["path_hand_written_prompt"]
        if path_hand_written_prompt is not None:
            hand_written_prompt = open(path_hand_written_prompt, "r").read()
        target_string = target["target_string"]
        dict_compat = {
            "docstring": documentation,
            "example_code": example_code,
            "separator": trigger_to_generate_input,
            "begin": input_hint,
            "hw_prompt": hand_written_prompt,
            "target_api": target_string,
        }
        return dict_compat

    def write_back_file(self, code: str):
        raise NotImplementedError

    # each target defines their way of validating prompts (can overwrite)
    def validate_prompt(self, prompt: str):
        fos = self.model.generate(
            prompt,
            batch_size=self.batch_size,
            temperature=self.temperature,
            max_length=self.max_length,
        )
        unique_set = set()
        score = 0
        for fo in fos:
            code = self.prompt_used["begin"] + "\n" + fo
            wb_file = self.write_back_file(code)
            result, _ = self.validate_individual(wb_file)
            if (
                result == FResult.SAFE
                and self.filter(code)
                and self.clean_code(code) not in unique_set
            ):
                unique_set.add(self.clean_code(code))
                score += 1
        return score

    # each target defines their way of validating prompts
    # for example we might want to encode the prompt as a docstring comment to facilitate better generation using
    # smaller LLMs
    def wrap_prompt(self, prompt: str) -> str:
        raise NotImplementedError

    def wrap_in_comment(self, prompt: str) -> str:
        raise NotImplementedError

    def _create_auto_prompt_message(self, message: str) -> List[dict]:
        return [
            {"role": "system", "content": self.AP_SYSTEM_MESSAGE},
            {"role": "user", "content": message + "\n" + self.AP_INSTRUCTION},
        ]

    def auto_prompt(self, **kwargs) -> str:
        os.makedirs(self.folder + "/prompts", exist_ok=True)

        # if we have already done auto-prompting, just return the best prompt
        if os.path.exists(self.folder + "/prompts/best_prompt.txt"):
            self.m_logger.logo("Use existing prompt ... ", level=LEVEL.INFO)
            with open(
                self.folder + "/prompts/best_prompt.txt", "r", encoding="utf-8"
            ) as f:
                return f.read()
        if kwargs["no_input_prompt"]:
            self.m_logger.logo("Without any input prompt ... ", level=LEVEL.INFO)
            best_prompt = (
                f"{self.prompt_used['separator']}\n{self.prompt_used['begin']}"
            )
        elif kwargs["hw"]:
            self.m_logger.logo("Use handwritten prompt ... ", level=LEVEL.INFO)
            best_prompt = self.wrap_prompt(kwargs["hw_prompt"])
        else:
            self.m_logger.logo("Use auto-prompting prompt ... ", level=LEVEL.INFO)
            message = kwargs["message"]
            # first run with temperature 0.0 to get the first prompt
            config = create_config(
                {},
                self._create_auto_prompt_message(message),
                max_tokens=500,
                temperature=0.0,
                model="gpt-4",
            )
            response = request_engine(config)
            greedy_prompt = self.wrap_prompt(response.choices[0].message.content)
            with open(
                self.folder + "/prompts/greedy_prompt.txt", "w", encoding="utf-8"
            ) as f:
                f.write(greedy_prompt)
            # repeated runs with temperature 1 to get additional prompts
            # choose the prompt with max score
            best_prompt, best_score = greedy_prompt, self.validate_prompt(greedy_prompt)
            with open(self.folder + "/prompts/scores.txt", "a") as f:
                f.write(f"greedy score: {str(best_score)}")
            for i in track(range(3), description="Generating prompts..."):
                config = create_config(
                    {},
                    self._create_auto_prompt_message(message),
                    max_tokens=500,
                    temperature=1,
                    model="gpt-4",
                )
                response = request_engine(config)
                prompt = self.wrap_prompt(response.choices[0].message.content)
                with open(
                    self.folder + "/prompts/prompt_{}.txt".format(i),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(prompt)
                score = self.validate_prompt(prompt)
                if score > best_score:
                    best_score = score
                    best_prompt = prompt
                # dump score
                with open(self.folder + "/prompts/scores.txt", "a") as f:
                    f.write(f"\n{i} prompt score: {str(score)}")

        # dump best prompt
        with open(self.folder + "/prompts/best_prompt.txt", "w", encoding="utf-8") as f:
            f.write(best_prompt)

        return best_prompt

    # initialize through either some templates or auto-prompting to determine prompts
    def initialize(self):
        self.m_logger.logo(
            "Initializing ... this may take a while ...", level=LEVEL.INFO
        )
        self.m_logger.logo("Loading model ...", level=LEVEL.INFO)
        eos = [
            self.prompt_used["separator"],
            "<eom>",  # for codegen2
            self.se_prompt,
            self.m_prompt,
            self.c_prompt,
        ]
        # if the config_dict is an attribute, add additional eos from config_dict
        # which might be model specific
        if hasattr(self, "config_dict"):
            llm = self.config_dict["llm"]
            model_name = llm["model_name"]
            additional_eos = llm.get("additional_eos", [])
            if additional_eos:
                eos = eos + additional_eos
        else:
            model_name = self.model_name

        if self.special_eos is not None:
            eos = eos + [self.special_eos]

        self.model = make_model(
            eos=eos,
            model_name=model_name,
            # device=self.device,
            max_length=self.max_length,
        )
        self.m_logger.logo("Model Loaded", level=LEVEL.INFO)
        self.initial_prompt = self.auto_prompt(
            message=self.prompt_used["docstring"],
            hw_prompt=self.prompt_used["hw_prompt"] if self.hw else None,
            hw=self.hw,
            no_input_prompt=self.no_input_prompt,
        )
        self.prompt = self.initial_prompt+ "\n" + "Provide Only the Final Code: Output only the repaired and completed code without any explanations, comments, markdown or additional text.\nMake sure the final code is very short, clean, well-formatted, and ready to run.\n"
        self.m_logger.logo("Done", level=LEVEL.INFO)

    def  generate_model(self) -> List[str]:
        self.g_logger.logo(self.prompt, level=LEVEL.VERBOSE)
        return self.model.generate(
            self.prompt,
            batch_size=self.batch_size,
            temperature=self.temperature,
            max_length=150,
        )

    # generation
    def  generate(self, **kwargs) -> Union[List[str], bool]:
        fos = self.generate_model()
        new_fos = []
        for fo in fos:
            self.g_logger.logo("========== sample =========", level=LEVEL.VERBOSE)
            new_fos.append(self.clean(self.prompt_used["begin"] + "\n" + fo))
            self.g_logger.logo(
                self.clean(self.prompt_used["begin"] + "\n" + fo), level=LEVEL.VERBOSE
            )
            self.g_logger.logo("========== sample =========", level=LEVEL.VERBOSE)
        return new_fos

    # helper for updating
    def filter(self, code: str) -> bool:
        raise NotImplementedError

    # difference between clean and clean_code (honestly just backwards compatibility)
    # but the point is that clean should be applied as soon as generation whereas clean code is used
    # more so for filtering
    def clean(self, code: str) -> str:
        raise NotImplementedError

    def clean_code(self, code: str) -> str:
        raise NotImplementedError

    def mask(self,text,mask_ratio=0.3):
        num_to_remove = int(len(text) * mask_ratio)
        char_list = list(text)
        indices_to_remove = random.sample(range(int(len(text))), num_to_remove)
        for index in sorted(indices_to_remove, reverse=True):
            del char_list[index]
        return ''.join(char_list)
    
    def caculate_ration(self,n):
        fraction = 0.75 - (n / self.batch_size) * (0.75 - 0.25)
        return fraction
        
    # update
    def update(self, **kwargs):
        new_code = ""
        abnormal_count = 0
        for result, code in kwargs["prev"]:
            if result in [FResult.ERROR, FResult.FAILURE, FResult.TIMED_OUT]:
                abnormal_count += 1
                new_code = code
        if abnormal_count == 0:
            new_code = random.choice([code for _, code in kwargs["prev"]])
        fos = self.model.update(self.mask(new_code,self.caculate_ration(abnormal_count)),self.initial_prompt,batch_size=self.batch_size)
        return fos

    # validation
    def validate_individual(self, filename) -> (FResult, str):
        raise NotImplementedError

    def parse_validation_message(self, f_result, message, file_name):
        # TODO: rewrite to include only status in TRACE but full message in VERBOSE
        self.v_logger.logo("Validating {} ...".format(file_name), LEVEL.TRACE)
        if f_result == FResult.SAFE:
            self.v_logger.logo("{} is safe".format(file_name), LEVEL.VERBOSE)
        elif f_result == FResult.FAILURE:
            self.v_logger.logo(
                "{} failed validation with error message: {}".format(
                    file_name, message, LEVEL.VERBOSE
                )
            )
        elif f_result == FResult.ERROR:
            self.v_logger.logo(
                "{} has potential error!\nerror message:\n{}".format(
                    file_name, message, LEVEL.VERBOSE
                )
            )
            self.m_logger.logo(
                "{} has potential error!".format(file_name, message, LEVEL.INFO)
            )
        elif f_result == FResult.TIMED_OUT:
            self.v_logger.logo("{} timed out".format(file_name), LEVEL.VERBOSE)

    def validate_all(self):
        for fuzz_output in track(
            glob.glob(self.folder + "/*.fuzz"),
            description="Validating",
        ):
            f_result, message = self.validate_individual(fuzz_output)
            self.parse_validation_message(f_result, message, fuzz_output)
