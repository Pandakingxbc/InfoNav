# IGScore.msg
# Information Gain score for exploration guidance
# Reference: main.tex Section III-D "VLM-Based Future IG Estimation"

Header header

# Average IG score (0-1), higher = more exploration potential
float64 ig_score

# Individual connectivity prompt scores
float64 corridor_score    # "This view shows a corridor or hallway leading to other areas"
float64 doorway_score     # "There is a doorway or opening that leads to another room"
float64 passage_score     # "This passage connects to multiple rooms or spaces"
