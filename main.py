from inspect_ai import eval

import tasks


def main():
    print("Hello from inspect-hello!")
    eval(tasks.hello_world(), model="openai/gpt-4o")


if __name__ == "__main__":
    main()
