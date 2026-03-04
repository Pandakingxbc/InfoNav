# VLMVerificationRequest.msg
# C++ -> Python: 请求VLM验证目标物体

# 请求验证的物体ID
int32 object_id

# 目标物体类别名称
string target_category

# 触发类型: 0=distance, 1=early, 2=rescue
int32 trigger_type

# 当前距离目标的距离
float64 distance_to_target

# ====== Object Map Confidence Info (for VLM prompt) ======
# 目标物体的融合置信度 (fused_confidence_)
float64 target_fused_confidence

# 目标物体的观测次数 (observation_nums_[0])
int32 target_observation_count

# 相似物体信息 (best_label_ != 0 的物体)
# 格式: "label1:conf1:count1,label2:conf2:count2,..."
# 例如: "chair:0.65:3,bench:0.45:2"
string similar_objects_info

# 当前动态阈值 tau(t)
float64 current_threshold

# 消息头
std_msgs/Header header
