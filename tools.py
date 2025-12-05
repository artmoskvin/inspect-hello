from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import tool


@tool
def add():
    async def execute(x: int, y: int) -> int:
        """
        Add two numbers.

        Args:
            x: The first number.
            y: The second number.

        Returns:
            The sum of the two numbers.
        """
        return x + y

    return execute


@task
def addition():
    return Task(
        dataset=[
            Sample(input="What is 1 + 1?", target=["2", "2.0"]),
        ],
        solver=[
            use_tools(add()),
            generate(),
        ],
        scorer=match(numeric=True),
    )
