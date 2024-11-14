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
            content = '\n'.join([line for line in content.splitlines() if '```' not in line])
            responses.append(content)
        return responses

    def update(self,code,init_program,batch_size=5):
        responses = []
        for i in range(batch_size):
            messages = [
                {
                    "role": "system",
                    "content": (
                        """1.	Analyze the entire code to understand its purpose and functionality.
	2.	Identify syntax errors and fix issues such as missing brackets, incorrect indentation, or misused language constructs.
	3.	Detect and resolve logical flaws that could prevent proper execution or desired output.
	4.	Fill in any missing parts necessary for the code to function.
	5.	Make all needed corrections and additions to ensure the code is fully operational.
	6.	Output only the final, repaired, and completed code without explanations, comments, or additional text.
	7.	Maintain the user’s specifications as strictly as possible; do not generate unrelated code.
	8.	Remove any comments, markdown tags, or non-essential elements.
	9.	Ensure the code is concise, well-formatted, and ready to run."""
                    )
                },
                {"role": "user", "content": "specifications as follows:\n"+init_program},
                {"role": "user", "content": code}
            ]
        content = self.model_generator.invoke(messages)
        content = '\n'.join([line for line in content.splitlines() if '```' not in line])
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