# Summary and Highlights

Congratulations! You have completed this module. At this point, you know that: 

- Generative AI is a reactive system that creates content like text or images based on prompts. Agentic AI, on the other hand, is proactive and uses prompts to pursue goals. 

- LangGraph is an advanced framework designed for building stateful, multiagent applications. 

- Nodes are functions that do the actual computation. Edges define how the execution flows from one step to the next. State is a shared memory that remembers everything across nodes. 

- LangGraph's unique capabilities include:
    - Looping and branching for making dynamic decisions
    - State persistence to maintain context over long interactions
    - Human-in-the-loop functionality for timely human interventions
    - Time travel to facilitate convenient debugging

- LangGraph offers state management, allowing the workflow to maintain and modify context across different nodes. It also offers conditional transitions, enabling the workflow to make decisions at runtime and branch accordingly.

- A LangGraph workflow can branch, loop, pause for human input, and resume execution, all while preserving full conversational memory.

- LangGraph graphs can be visualized using Mermaid diagrams with core primitives such as nodes and edges clearly represented.

- LangChain helps developers build LLM-powered applications using modular components like prompts, memory, and tools. LangGraph, on the other hand, extends LangChain's capabilities by enabling stateful, multiagent workflows.

- State in LangGraph is a complex, evolving memory that contains all inputs, intermediate values, and outputs.

- Nodes are functions that process the current state. Some nodes modify the state, whereas others are used for side effects.

- Edges define how the execution flows between nodes, passing the updated state from one step to the next.

- Conditional edges allow the workflow to make dynamic decisions, routing the state to different nodes.

- Building a LangGraph application involves creating a StateGraph object, incorporating nodes, connecting them, setting an entry point, and then compiling the graph into a runnable application.

- Running a LangGraph workflow is done by invoking the compiled application with an initial state.

- Workflow visualization helps to understand the execution flow and how the state progresses through different nodes.