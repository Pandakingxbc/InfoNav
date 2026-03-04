# SemanticPrompts.msg
# Message for multi-source semantic prompt expansion

Header header

# Target object for navigation
string target_object

# List of semantic prompts (hypotheses)
# e.g., ["Is there a dining table ahead?", "Is there a dining room ahead?", ...]
string[] prompts

# Fusion weights for each prompt (same length as prompts)
# Weights should sum to 1.0, but will be normalized if not
float64[] weights
