from inspect_ai import Task, task
from inspect_ai.dataset import Sample, example_dataset
from inspect_ai.scorer import exact, model_graded_fact
from inspect_ai.solver import chain_of_thought, generate, self_critique


@task
def theory_of_mind():
    return Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[
            chain_of_thought(),
            generate(),
            self_critique(),
        ],
        scorer=model_graded_fact(),
    )


@task
def hello_world():
    return Task(
        dataset=[
            Sample(input="Just reply with Hello World", target="Hello World"),
        ],
        solver=[
            generate(),
        ],
        scorer=exact(),
    )
