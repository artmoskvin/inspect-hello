# inspect-hello

A personal playground for learning [Inspect AI](https://inspect.ai-safety-institute.org.uk/) - a framework for evaluating large language models.

## Setup

Requires Python 3.13+.

```bash
uv sync
```

## Example Tasks

### Basic Examples

| File | Task | Description |
|------|------|-------------|
| `tasks.py` | `hello_world` | Minimal example - model replies "Hello World" |
| `tasks.py` | `theory_of_mind` | Chain-of-thought with self-critique |
| `security.py` | `security_guide` | Security Q&A with model-graded scoring |
| `tools.py` | `addition` | Custom tool usage example |

### Benchmarks

| File | Task | Description |
|------|------|-------------|
| `gsm8k.py` | `gsm8k` | Grade school math with few-shot prompting |
| `hellaswag.py` | `hellaswag` | Commonsense reasoning (multiple choice) |
| `math.py` | `math` | MATH-500 with custom equivalence scorer |

### Agentic

| File | Task | Description |
|------|------|-------------|
| `intercode/task.py` | `intercode_ctf` | CTF challenges using ReAct agent with bash/python tools |

## Running Evaluations

```bash
# Run a specific task
inspect eval tasks.py@hello_world --model openai/gpt-4o

# Run with different models
inspect eval gsm8k.py --model anthropic/claude-sonnet-4-20250514
inspect eval math.py --model google/gemini-2.0-flash

# View results
inspect view
```
