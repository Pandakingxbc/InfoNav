#!/bin/bash
# 快速启动脚本：重新测试所有缺失的任务

echo "======================================================================"
echo "🔄 开始重新测试缺失的 LLM 分析任务"
echo "======================================================================"

cd /home/yangz/Nav/ApexNav

# 检查 llm_hypothesis_analysis.py 是否存在
if [ ! -f "llm_hypothesis_analysis.py" ]; then
    echo "❌ 错误: llm_hypothesis_analysis.py 不存在"
    exit 1
fi

echo ""
echo "📋 缺失任务统计:"
python3 << 'EOF'
import os
task_dirs = sorted([d for d in os.listdir('env/env_hm3dv2') if d.startswith('task')])
missing = []
for task_dir in task_dirs:
    task_num = int(task_dir.replace('task', ''))
    if not os.path.exists(f'env/env_hm3dv2/{task_dir}/llm_hypothesis_analysis.json'):
        missing.append(task_num)

print(f"   总缺失: {len(missing)} 个")
print(f"   范围: task{min(missing)} ~ task{max(missing)}")
EOF

echo ""
echo "📌 使用方法:"
echo "   1. 处理所有缺失的任务 (完整):"
echo "      python3 retry_missing_tasks.py --env hm3dv2"
echo ""
echo "   2. 处理前 10 个任务 (测试):"
echo "      python3 retry_missing_tasks.py --env hm3dv2 --start 127 --end 140"
echo ""
echo "   3. 处理特定范围 (如 task300-400):"
echo "      python3 retry_missing_tasks.py --env hm3dv2 --start 300 --end 400"
echo ""
echo "   4. 增加超时时间 (如果出现超时错误):"
echo "      python3 retry_missing_tasks.py --env hm3dv2 --timeout 180"
echo ""
echo "======================================================================"
echo ""
