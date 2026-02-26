import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated
from typing_extensions import TypedDict

# 環境変数を読み込む
load_dotenv(".env")
os.environ['OPENAI_API_KEY'] = os.environ['API_KEY']

# 使用するモデル名
MODEL_NAME = "gpt-4o-mini" 

### ↑まで基本の記載 ###

# MemorySaverインスタンスの作成
memory = MemorySaver()

# スレッドIDの定義（単一ユーザーモード）
THREAD_ID = "1"

# グラフの遅延初期化用グローバル変数
graph = None

# ===== Stateクラスの定義 =====
class State(TypedDict):
    """メッセージのリストを保持する辞書型"""
    messages: Annotated[list, add_messages]

# ===== グラフの構築 =====
def build_graph(model_name):
    # グラフのインスタンスを作成
    graph_builder = StateGraph(State)

    # ツールノードを作成（TavilySearchResultsを使用）
    tavily_tool = TavilySearchResults(max_results=2)
    tools = [tavily_tool]
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    # チャットボットノードの作成
    llm = ChatOpenAI(model_name=model_name)
    llm_with_tools = llm.bind_tools(tools)

    # チャットボットの実行方法を定義
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    # 実行可能なグラフの作成
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")

    return graph_builder.compile(checkpointer=memory)

# ===== グラフを実行する関数 =====
def stream_graph_updates(graph, user_message: str):
    response = graph.invoke(
        {"messages": [("user", user_message)]},
        {"configurable": {"thread_id": THREAD_ID}},
        stream_mode="values"
    )
    return response["messages"][-1].content

# ===== 応答を返す関数 =====
def get_bot_response(user_message, system_message, memory):
    global graph

    # 遅延初期化: グラフがまだ作成されていない場合、新しいグラフを作成
    if graph is None:
        graph = build_graph(MODEL_NAME)

    # 初回メッセージかチェック
    try:
        existing = memory.get({"configurable": {"thread_id": THREAD_ID}})
        is_first = not existing or not existing.get('channel_values', {}).get('messages', [])
    except:
        is_first = True

    # 初回でsystem_messageがあれば追加
    if is_first and system_message and system_message.strip():
        graph.invoke(
            {"messages": [("system", system_message)]},
            {"configurable": {"thread_id": THREAD_ID}},
        )

    # ユーザーメッセージでグラフを実行
    stream_graph_updates(graph, user_message)

    # 会話履歴全体を返す
    return get_messages_list(memory)

# ===== メッセージの一覧を取得する関数 =====
def get_messages_list(memory):
    messages = []

    try:
        memory_data = memory.get({"configurable": {"thread_id": THREAD_ID}})
        if memory_data and 'channel_values' in memory_data:
            stored_messages = memory_data['channel_values']['messages']

            for message in stored_messages:
                if isinstance(message, HumanMessage):
                    # ユーザーからのメッセージ
                    messages.append({'class': 'user-message', 'text': message.content})
                elif isinstance(message, AIMessage) and message.content != "":
                    # ボットからのメッセージ（最終回答のみ、ツール呼び出しメッセージは除外）
                    messages.append({'class': 'bot-message', 'text': message.content})
    except (KeyError, TypeError):
        # メモリが空または初期化されていない
        pass

    return messages