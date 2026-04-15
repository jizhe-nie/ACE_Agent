import os
import sys
from pathlib import Path

# 将项目根目录的父目录加入 sys.path，以支持 ACE_Agent.xxx 形式的导入
root_parent = str(Path(__file__).resolve().parents[2])
if root_parent not in sys.path:
    sys.path.append(root_parent)

from dotenv import load_dotenv
from ACE_Agent.agent_core.supervisor import ACESupervisor
from ACE_Agent.tools.data_factory import generate_dataset
from ACE_Agent.tools.settings_store import SettingsStore
from ACE_Agent.tools.llm_client import LLMSettings

def test_follow_up():
    load_dotenv()
    store = SettingsStore()
    settings = LLMSettings(
        base_url=store.get("llm_base_url"),
        api_key=store.get("llm_api_key"),
        model=store.get("llm_model"),
        enabled=store.get("llm_enabled", False)
    )
    
    supervisor = ACESupervisor()
    
    # 1. 第一次任务：分析笑脸数据
    dataset = generate_dataset("smile")
    print("\n--- 任务 1: 分析笑脸数据 ---")
    report1 = supervisor.run(dataset, "帮我分析这个笑脸数据", llm_settings=settings)
    print(f"意图: {report1.response_type}")
    print(f"总结: {report1.llm_summary[:100]}..." if report1.llm_summary else "无总结")

    # 2. 第二次任务：追问细节
    print("\n--- 任务 2: 追问细节 ---")
    # 追问时，即使传入了 dataset，supervisor 也应该识别出 FOLLOW_UP 并跳过计算
    report2 = supervisor.run(dataset, "具体解释一下为什么选择这个算法？它的轮廓系数是多少？", llm_settings=settings)
    print(f"意图: {report2.response_type}")
    print(f"回答: {report2.llm_summary}")
    
    if report2.response_type == "FOLLOW_UP":
        print("\n✅ 成功：Agent 识别到了追问，并基于历史结果进行了回答，没有重新运行算法。")
    else:
        print("\n❌ 失败：Agent 重新运行了算法。")

if __name__ == "__main__":
    test_follow_up()
