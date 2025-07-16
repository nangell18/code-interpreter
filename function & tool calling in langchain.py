from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

"""this program allows you to switch between different chatbots aka Sonnets and OpenAIs functions calling.
in the past, you only had access to function call openai models but now you can do more than that with the
function create_tool_calling_agent."""

@tool
def multiply(x : float, y: float) -> float:
    """multiply 'x' times 'y'"""
    return x * y

if __name__ == "__main__":
    print("Hello tool calling")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you're a helpful assistant"),
            ("human","{input}"),
            ("placeholder","{agent_scratchpad}")
        ]
    )

    tools = [TavilySearchResults, multiply]
    llm = ChatOpenAI(temperature=0,model="gpt-4o-mini")
    # llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=0)

    agent = create_tool_calling_agent(llm,tools,prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools)

    res = agent_executor.invoke(
        {
            # "input":"what is the weather in Dubai right now? compare it with San Fransisco, output should be in celsius"
            "input":"what is 4*6?"
        }
    )

    print(res)