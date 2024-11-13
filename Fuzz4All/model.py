import os
from typing import List

from langchain_community.llms import Ollama

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable warning



class EndOfFunctionCriteria:
    def __init__(self, eos):
        self.eos = eos

    def check_end_of_function(self, generation: str) -> bool:
        """Returns true if the generated sequence contains any of the end-of-function strings."""
        return any(stop_string in generation for stop_string in self.eos)


class Model:
    def __init__(self, model_name: str, eos: List[str], max_length: int) -> None:
        self.model_generator = Ollama(model=model_name,temperature=0, num_predict=max_length)

    def generate(self, prompt: str, temperature=0, max_length=250, batch_size=5) -> List[str]:
        if max_length is None:
            max_length = self.max_length
        responses = []
        for i in range(batch_size):
            messages = [{"role": "user", "content": prompt}]
            content = self.model_generator.invoke(messages)
            responses.append(content)
        return responses

    def update(self,code,init_program,batch_size=5):
        responses = []
        for i in range(batch_size):
            messages = [
                {
                    "role": "system",
                    "content": (
                        """Your task is to repair and complete the broken code provided to you. Please follow these instructions carefully:
1.	Analyze the Code: Read through the entire code to understand its purpose and functionality.
2.	Identify Syntax Errors: Look for any syntax mistakes such as missing brackets, incorrect indentation, or misused language constructs.
3.	Find Logical Errors: Check for logical flaws that may prevent the code from executing correctly or producing the desired output.
4.	Complete Incomplete Sections: Fill in any missing parts of the code that are necessary for it to function properly.
5.	Correct and Complete the Code: Make all the necessary fixes and additions to ensure the code is fully functional.
6.	Provide Only the Final Code: Output only the repaired and completed code without any explanations, comments, or additional text.
7.  Do not give other comments and explaination
8.  Must keep specifications that user brings to you
9.  Remove any comments, markdown tags, etc., from the code, if there are any.
Make sure the final code is clean, well-formatted, and ready to run.
"""
                    )
                },
                {"role": "user", "content": "specifications as follows:\n"+init_program},
                {"role": "user", "content": code}
            ]
        content = self.model_generator.invoke(messages)
        responses.append(content)
        return responses

def make_model(eos: List[str], model_name: str, max_length: int) -> Model:
    """Returns a StarCoder model instance using the Ollama client."""
    print("=== Model Config ===")
    print(f"model_name: {model_name}")
    print(f"max_length: {max_length}")
    for stop in eos:
        print(f"eos: {stop}")
    print("====================")

    model_obj = Model(model_name=model_name, eos=eos, max_length=max_length)
    return model_obj