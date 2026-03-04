# VLMVerificationResult.msg
# Python -> C++: VLM验证结果

# 验证的物体ID
int32 object_id

# 三级验证决策:
#  1 = CONFIRM (确认是目标物体，提高置信度)
#  0 = UNCERTAIN (不确定，保持原置信度)
# -1 = REJECT (确认是假阳性，降低置信度)
int8 decision_level

# VLM置信度 (0-1)，表示VLM对其决策的信心
float64 vlm_confidence

# VLM回复的原因说明
string reason

# 是否超时
bool timeout

# 验证耗时(秒)
float64 duration

# 消息头
std_msgs/Header header
