# SemanticScores.msg
# Multi-source semantic ITM scores with LLM hypothesis weights

Header header

# Target object being searched for
string target_object

# Hypothesis metadata (same length as prompts/scores/weights)
int32[] hypothesis_ids
string[] hypothesis_types

# Semantic prompts for CLIP/BLIP matching
string[] prompts

# LLM hypothesis scores (computed offline, fixed per episode)
float64[] confidences         # C: reliability of hypothesis (0-1)
float64[] navigation_values   # V: spatial scope of hypothesis (0-1)
float64[] weights             # W = C × V: fusion weight for Value Map

# Real-time VLM observation scores (updated each step)
float64[] scores              # ITM cosine similarity scores
