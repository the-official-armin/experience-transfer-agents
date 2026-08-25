import re

SYSTEM_PROMPT = """
You are a tool-using agent.

You have access to exactly one tool:

calculator(expression)
Calculates a mathematical expression.

You must follow this format exactly.

If you need the calculator:

TOOL: calculator
ARGUMENT: <mathematical expression>

After receiving the tool result, if you know the answer:

FINAL: <answer>

Do not write anything before TOOL or FINAL.
Do not include words such as "assistant", "user", or "system".
"""


class Agent:

    def __init__(self, model, tokenizer, environment):
        self.model = model
        self.tokenizer = tokenizer
        self.environment = environment
    def ask_llm(self, messages, max_new_tokens=150):

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def run(self, task, max_steps=5):

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": task
            }
        ]

        trajectory = []

        for step in range(max_steps):

            response = self.ask_llm(messages)

            # Store the raw model response.
            trajectory.append({
                "step": step,
                "agent_response": response
            })

            # -------------------------
            # Final answer
            # -------------------------

            if "FINAL:" in response:

                match = re.search(
                    r"FINAL:\s*(.*)",
                    response,
                    re.DOTALL
                )

                answer = match.group(1).strip()

                return {
                    "task": task,
                    "trajectory": trajectory,
                    "final_answer": answer,
                    "success": True
                }

            # -------------------------
            # Tool call
            # -------------------------

            if "TOOL: calculator" in response:

                match = re.search(
                    r"ARGUMENT:\s*(.*)",
                    response
                )

                if not match:
                    return {
                        "task": task,
                        "trajectory": trajectory,
                        "final_answer": None,
                        "success": False
                    }

                expression = match.group(1).strip()

                result = self.environment.execute_tool(
                    "calculator",
                    expression
                )

                trajectory.append({
                    "step": step,
                    "tool": "calculator",
                    "argument": expression,
                    "observation": result
                })

                messages.append({
                    "role": "assistant",
                    "content": response
                })

                messages.append({
                    "role": "user",
                    "content": f"Tool result: {result}"
                })

                continue

            # Unknown response format
            return {
                "task": task,
                "trajectory": trajectory,
                "final_answer": None,
                "success": False
            }

        return {
            "task": task,
            "trajectory": trajectory,
            "final_answer": None,
            "success": False
        }