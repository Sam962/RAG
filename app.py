from src.graph import build_graph

if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "cli"}}
    query = "who is Sam Aldehayyat and what projects he had done and his experiences?"
    result = graph.invoke({"question": query}, config=config)
    print("answer", result["answer"])
    print("citations", result["citations"])
