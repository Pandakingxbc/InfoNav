# ObjectUpdateInfo.msg
# C++ -> Python: 通知哪些帧更新了目标物体

# 被更新的物体ID
int32 object_id

# 这次更新的检测置信度
float64 detection_confidence

# 当前距离目标的距离
float64 distance_to_object

# 是否是当前导航目标物体
bool is_current_target

# 消息头
std_msgs/Header header
