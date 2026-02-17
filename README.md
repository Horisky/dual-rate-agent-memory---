# Dual-Rate Memory for Long-Context Agents  
# 面向长对话智能体的双速记忆策略  

## Overview  
This project implements and evaluates a **dual‑rate recursive memory** strategy for AI agents.  
It reduces long‑term memory breakage by maintaining **fast/slow summaries** with consistency checks.  

本项目实现并评估了**双速递归记忆**策略，通过维护**快/慢摘要**并进行一致性检查，显著缓解智能体长期记忆断流问题。  

## Key Result (Topical‑Chat Full)  
Dataset size: **10,784** conversations  

Compared to the recursive summary baseline:  
- **avg_recall**: 0.584 vs 0.531 (**+10.0%**)  
- **d4 (long delay)**: 0.230 vs 0.189 (**+21.7%**)  
- Cost: calls +41.7%, input tokens +85.6%, time +108%  

在 Topical‑Chat 全量数据上：  
- **平均召回率**：0.584 vs 0.531（**+10.0%**）  
- **d4 延迟召回**：0.230 vs 0.189（**+21.7%**）  
- 代价：调用次数 +41.7%，输入 token +85.6%，时间 +108%  

## Files  
- `memory_eval.py` — evaluation script (recall, delay curve, cost/time stats)  
- `make_facts.py` — stronger fact extraction  
- `local_agent_memory.py` — LM Studio local demo (Qwen2.5‑14B)  
- `transcript_all.txt`, `facts_all.jsonl` — merged dataset for full evaluation  

## Quick Start  
### 1) Evaluate (full dataset)  
```powershell
python .\memory_eval.py --transcript .\transcript_all.txt --facts .\facts_all.jsonl --chunk-tokens 300 --slow-tokens 300 --slow-update-every 5 --slow-importance 3.5 --fast-update-every 0 --fast-importance 2.0
```

### 2) Local demo (LM Studio)  
Start LM Studio API server, then run:  
```powershell
python .\local_agent_memory.py
```

## Method  
- **Fast memory**: updates frequently for short‑term adaptation  
- **Slow memory**: updates less frequently or only on important content  
- **Consistency check**: prevents fast memory from drifting too far from slow memory  

方法要点：  
- **快记忆**：频繁更新，捕捉近期信息  
- **慢记忆**：低频/高重要性触发，保留稳定事实  
- **一致性校验**：抑制快记忆漂移  

## Evaluation Metrics  
- `avg_recall`: overall factual recall  
- `d1–d4`: delayed recall by block distance (long‑term stability)  
- `calls / in_tokens / out_tokens / time_ms`: cost statistics  

指标：  
- `avg_recall`：整体事实召回  
- `d1–d4`：延迟召回曲线（长期稳定性）  
- `calls / in_tokens / out_tokens / time_ms`：成本统计  

## Notes  
- Large datasets are included locally; consider excluding them if pushing to GitHub.  

注意：  
- 数据集文件较大，上传 GitHub 前建议加入 `.gitignore` 排除。  

