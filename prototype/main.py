from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from agent import Agent
from environment import Environment
from tools import calculator


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

    return tokenizer, model


def main():

    # Load model
    tokenizer, model = load_model()

    # Create environment
    environment = Environment()

    environment.register_tool(
        "calculator",
        calculator
    )

    # Create agent
    agent = Agent(
        model=model,
        tokenizer=tokenizer,
        environment=environment
    )

    # Run task
    result = agent.run(
        "Calculate (12 * 8) + 19 using the calculator."
    )

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)

    print("Task:", result["task"])
    print("Final answer:", result["final_answer"])
    print("Success:", result["success"])

    print("\nTrajectory:")

    for step in result["trajectory"]:
        print(step)


if __name__ == "__main__":
    main()