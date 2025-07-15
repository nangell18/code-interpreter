from dotenv import load_dotenv
from langchain import hub
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import (
    PythonREPLTool,
)  # this is a dangerous tool because it gives our llm an ability to write and execute python code in the interpreter
from numpy.f2py.crackfortran import verbose

load_dotenv()  # load all of our .env files to our environment


# ---------------

"""I DID NOT RUN THIS CODE BECAUSE OF THIS REASON ->
And please, please, please, be careful when you use those type of tools.
This is an overly permissive tool that can do a lot of things. This is literally remote code execution.
So, if an attacker has access to our agent and is able to input some prompts to it,
then it's basically has the power to run code in our systems.
So, we should be very, very careful when we are using those type of tools."""

"""it is also important that we already have qrcode package already installed in our local
environment. I DO NOT KNOW if that it would work without it but it is an important note to remember"""

# ----------------


def main():
    print("start....")

    instructions = """You are an agent designed to write and execute python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question. 
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [
        PythonREPLTool()
    ]  # so we want to create our tool list, and we want our tool list to be the Python REPL tool. The Python REPL tool is a Python shell that can execute Python commands and Python code. So, its input should be valid Python code.
    # we do need to write any tool description, tool name because it is already a tool object

    agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(
            temperature=0, model="gpt-4-turbo"
        ),  # we needed this gpt-4 because the reasoning engine is much better. we need reasoning engine
        # however, it is slower and more expensive.
        tools=tools,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    """I got rid of this below because we are running the csv_agent"""
    # agent_executor.invoke(
    #     input={
    #         "input": """generate and save in current working directory 15 QRcodes
    #                                 that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    #     }
    # )

    """1. the csv agent is build upon pandas agent which underneath the hood is a regular agent that uses
    the python repl agent2. these agents are flaky, just be careful of the output because even though
    it is a deterministic output, aka the llm model is writing code to get the answer, it did not input the 
    entire csv file to the llm which in turn gave us the wrong result -> so even though we do give it reasoning,
    does not mean the reasoning will go smoothly"""
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model = "gpt-4"),
        path="episode_info.csv",
        verbose=True)

    csv_agent.invoke(input={"input":"How many columns are there in file episode_info.csv"})


if __name__ == "__main__":
    main()
