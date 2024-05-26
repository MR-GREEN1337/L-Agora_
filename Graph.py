from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
import functools
from langchain_core.messages import AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

from utils import load_env
from rag import Advanced_RAG

def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    sender: str
"""    subject: str
    dialog_state: Annotated[
        list[
            Literal[
                #"mediator",
                "shopenhauer",
                "freud",
                "aristotle",
                "call_tool"
            ]
        ],
        update_dialog_stack,
    ]
"""
class Graph:
    def __init__(self):
        load_env()

    def create_agent(philosopher, llm, tools, system_message: str):
        """Create an agent for the philosopher."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                """You are the philosopher {philosopher}. Embody their style and approach to discourse. 
                Engage in deep, reflective dialogue with other philosophers. 
                Always answer in short paragraphs, summarizing your thoughts. 
                Avoid repetitive and mundane assistant messages; instead, provide substantive and engaging contributions. 
                Use the provided tools to explore and progress towards answering complex philosophical questions. 
                If you feel uncomfortable with other philosopher's ideas, express your disagreement. 
                When the conversation has reached a consensus on the main ideas, prefix your response with FINAL ANSWER. 
                You have access to the following tools: {tool_names}. 
                {system_message}
                """
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(philosopher=philosopher)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        return prompt | llm.bind_tools(tools)

    # Helper function to create a node for a given agent
    def agent_node(state, agent, name):
        result = agent.invoke(state)
        # We convert the agent output into a format that is suitable to append to the global state
        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            # Since we have a strict workflow, we can
            # track the sender so we know who to pass to next.
            "sender": name,
        }


    def router(state) -> Literal["call_tool", "__end__", "continue"]:
        # This is the router
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            # The previous agent is invoking a tool
            return "call_tool"
        if "FINAL ANSWER" in last_message.content:
            # Any agent decided the work is done
            return "__end__"
        return "continue"



    def kickoff():
        load_env()
        tavily_tool = TavilySearchResults(max_results=8)

        llm = ChatGroq(model="mixtral-8x7b-32768")
        philosophers = ["aristotle", "shopenhauer", "freud", "Hegel"]


        # Research agent and node
        nodes = {}
        for philosopher in philosophers:
            agent = Graph.create_agent(
                philosopher,
                llm,
                [tavily_tool],
                system_message=f"You are {philosopher}, provide accurate and unique thoughts that are suitable to the debate. PLAY THE ROLE",
            )
            nodes[philosopher] = functools.partial(Graph.agent_node, agent=agent, name=philosopher)


        tools = [TavilySearchResults()] #[Advanced_RAG.kickoff] #Add knowledge base
        tool_node = ToolNode(tools)

        # Either agent can decide to end

        workflow = StateGraph(State)

        for philosopher, node in nodes.items():
            workflow.add_node(philosopher, node)

        workflow.add_node("call_tool", tool_node)

        workflow.add_conditional_edges(
            "aristotle",
            Graph.router,
            {"continue": "shopenhauer", "call_tool": "call_tool", "__end__": END},
        )
        workflow.add_conditional_edges(
            "shopenhauer",
            Graph.router,
            {"continue": "freud", "call_tool": "call_tool", "__end__": END},
        )
        workflow.add_conditional_edges(
            "freud",
            Graph.router,
            {"continue": "Hegel", "call_tool": "call_tool", "__end__": END},
        )
        workflow.add_conditional_edges(
            "Hegel",
            Graph.router,
            {"continue": "aristotle", "call_tool": "call_tool", "__end__": END},
        )

        workflow.add_conditional_edges(
            "call_tool",
            # Each agent node updates the 'sender' field
            # the tool calling node does not, meaning
            # this edge will route back to the original agent
            # who invoked the tool
            lambda x: x["sender"],
            {
                "aristotle": "aristotle",
                "shopenhauer": "shopenhauer",
                "freud": "freud"
            },
        )
        workflow.set_entry_point("aristotle")
        graph = workflow.compile()

        return graph

if __name__ == "__main__":
    graph = Graph.kickoff("Subject of work")