Original Dev article [here](https://dev.to/b-d055/optimizing-openais-gpt-4o-mini-to-detect-ai-generated-text-using-dspy-2775).

# Optimizing OpenAI’s GPT-4o-mini to Detect AI-Generated Text Using DSPy

Detecting AI-generated text has become a hot topic, with researchers and practitioners debating its feasibility and ethical implications. As models grow more sophisticated, distinguishing between human-written and AI-generated text becomes both an exciting challenge and a critical need. 

In this post, we’ll explore how to harness DSPy’s optimization capabilities to fine-tune OpenAI’s GPT-4o-mini for this task using a fascinating dataset of [1.39 million text samples](https://huggingface.co/datasets/artem9k/ai-text-detection-pile). By the end, you’ll know how to implement, evaluate, and optimize a basic AI-text detector using DSPy—no manual prompt engineering required.

---

## Dataset Loading

First, let’s load the dataset, which contains text samples labeled as either human-written or AI-generated from varied human and LLM sources. To get started, ensure you’ve installed Python 3, along with the [DSPy](https://github.com/stanfordnlp/dspy) and [hugging face datasets](https://huggingface.co/docs/datasets/en/installation)  libraries:

```bash
pip install dspy datasets
```

The dataset is approximately 2GB in size, so depending on your internet speed, this step may take a few minutes.

Here’s the code to load and split the dataset evenly for training and testing:

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("artem9k/ai-text-detection-pile", split="train")

# For simplicity, we’ll do an even split of data for testing and training
NUM_EXAMPLES = 80  # Adjust this to experiment with dataset size
ds = ds.train_test_split(test_size=NUM_EXAMPLES, train_size=NUM_EXAMPLES)
```

*Tip: You can adjust `NUM_EXAMPLES` to experiment with larger datasets or to reduce costs when running optimizations.*

---

## Model Setup

Next, we’ll create a basic DSPy predictor using OpenAI’s GPT-4o-mini. GPT-4o-mini is a lightweight version of OpenAI’s GPT-4o model, making it cost-efficient for experimentation. DSPy simplifies this process by using **[signatures](https://dspy.ai/learn/programming/signatures/)**, which define structured input-output mappings.

Replace `"YOUR_API_KEY"` with your OpenAI API key before running the code:

```python
import dspy
from typing import Literal

# Initialize the OpenAI GPT-4o-mini model
lm = dspy.LM('openai/gpt-4o-mini', api_key="YOUR_API_KEY")
dspy.configure(lm=lm, experimental=True)

# Define the AI text detector signature
class DetectAiText(dspy.Signature):
    """Classify text as written by human or by AI."""
    text: str = dspy.InputField()
    source: Literal['ai', 'human'] = dspy.OutputField()

# Create a basic predictor
detector = dspy.Predict(DetectAiText)
```

Notice that we haven’t done any prompt engineering here. Instead, we rely on DSPy to handle that, as well as the input-output relationships automatically.

You can test the "detector" with some sample input:

```python
print(detector(text="Hello world (this definitely wasn't written by AI)"))
```

The prediction will appear in the `'source'` field of the output.

---

## Evaluating the Detector

Now that we have a basic detector, let’s evaluate its performance using DSPy’s evaluation tools. For this, we’ll define a simple [metric](https://dspy.ai/learn/evaluation/metrics/) that checks if the model correctly predicts the text’s source (human or AI).

Here’s the code to set up and run the evaluation:

```python
from dspy.evaluate import Evaluate

# Define a simple evaluation metric
def validate_text_source(example: dspy.Example, pred, trace=None) -> int:
    return 1 if example.source.lower() == pred.source.lower() else 0

# Transform the dataset into DSPy-compatible "Example" objects
dspy_trainset = [
    dspy.Example(source=x['source'], text=x['text']).with_inputs('text') for x in ds['train']
]
dspy_devset = [
    dspy.Example(source=x['source'], text=x['text']).with_inputs('text') for x in ds['test']
]

# Evaluate the detector
evaluator = Evaluate(devset=dspy_devset, num_threads=12)  # Adjust threads based on your system
evaluator(detector, metric=validate_text_source)
```

In my initial tests, I achieved an accuracy of **79%–81%**. Note that results may vary due to the random sampling of the dataset.

---

## Optimizing with DSPy

The real power of DSPy lies in its optimization capabilities. By using the [MIPROv2 optimizer](https://dspy.ai/deep-dive/optimizers/miprov2/), we can improve the detector’s performance without manually tweaking prompts. The optimizer automates this process using few-shot examples, dynamic templates, and self-supervised techniques.

Here’s how to set up and run the optimizer:

```python
import time
from dspy.teleprompt import MIPROv2

# Initialize the optimizer
teleprompter = MIPROv2(
    metric=validate_text_source, 
    auto="light"  # Options: light, medium, heavy
)

# Run the optimization
optimized_program = teleprompter.compile(
    detector.deepcopy(),  # Make a copy of the detector to optimize
    trainset=dspy_trainset,
    valset=dspy_devset,
)

# Save the optimized program for future use
optimized_program.save(f'detect_ai_mipro_optimized_{time.time()}.dspy')
```

*Note: The cost for a single optimization run with the "light" preset is typically less than $0.50 for a dataset of 80 examples.*

---

## Results and Iteration

After running the optimization, I observed a significant performance boost. My first run achieved an accuracy of **91.25%**, compared to the baseline’s **79%–81%**. Subsequent runs ranged between **81.2%** and **91.25%**, demonstrating consistent improvements with minimal effort.

To load the optimized model for further use:

```python
detector.load(path='detect_ai_mipro_optimized_XXXXX.dspy')
```

You can iterate further by:
- Adjusting the optimizer’s `auto` parameter (light, medium, heavy), or setting hyper-parameters yourself.
- Increasing the dataset size for training and evaluation.
- Testing with more advanced or updated LLMs.

---

## Conclusion

In just a few steps, we demonstrated how DSPy simplifies LLM optimization for real-world use cases. Without any manual prompt engineering, we achieved a measurable improvement in detecting AI-generated text. While this model isn’t perfect, DSPy’s flexibility allows for continuous iteration, making it a valuable tool for scalable AI development.

I'd highly recommend a read-through of DSPy’s [documentation](https://dspy.ai/learn/) and experiment with other optimizers and LLM patterns.

---

Full code available on [GitHub](https://github.com/b-d055/optimizing-gpt-4o-mini-dspy).

Question? Comments? Let me know, I look forward to seeing what you build with DSPy!

You can find me on [LinkedIn](https://www.linkedin.com/in/b-d055/) | CTO & Partner @ [EES](https://www.eesolutions.io/). 