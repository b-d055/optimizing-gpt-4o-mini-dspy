from datasets import load_dataset

ds = load_dataset("artem9k/ai-text-detection-pile", split="train")

# for simplicity we'll do an even split of data for testing and training

NUM_EXAMPLES = 80
ds = ds.train_test_split(test_size=NUM_EXAMPLES, train_size=NUM_EXAMPLES)

import dspy
from typing import Literal

lm = dspy.LM('openai/gpt-4o-mini', api_key='')
dspy.configure(lm=lm, experimental=True)

class DetectAiText(dspy.Signature):
    """Classify text as written by human or by AI."""

    text: str = dspy.InputField()
    source: Literal['ai', 'human'] = dspy.OutputField()

detector = dspy.predict(DetectAiText) 

print(detector(text="Hello world (this definitely wasn't written by AI)"))

from dspy.evaluate import Evaluate

def validate_text_source(example: dspy.Example, pred, trace=None) -> int:
    if example.source.lower() == pred.source.lower():
        return 1
    return 0

# transform the dataset into a list of examples for use later
dspy_trainset = [dspy.Example(source=x['source'], text=x['text']).with_inputs('text') for x in ds['train']]
dspy_devset = [dspy.Example(source=x['source'], text=x['text']).with_inputs('text') for x in ds['test']]

evaluator = Evaluate(devset=dspy_devset, num_threads=12) # threads can be adjusted based on your system

# Launch evaluation of basic detector
evaluator(detector, metric=validate_text_source)

import time
from dspy.teleprompt import MIPROv2

# Setup the optimizer
teleprompter = MIPROv2(
    metric=validate_text_source,
    auto="light", # Can choose between light, medium, and heavy optimization runs
)
# Run optimization
optimized_program = teleprompter.compile(
    detector.deepcopy(),
    trainset=dspy_trainset,
    valset=dspy_devset,
)
# Save the optimized program for later retrieval/user
optimized_program.save(f'detect_ai_mipro_optimized_{time.time()}.dspy')

# Run the optimized program
detector.load(path='detect_ai_mipro_optimized_1732325134.90623.dspy')
evaluator(detector, metric=validate_text_source)

