# Counterfactual Outcomes
Official code repository for "Explaining Reinforcement Learning Agents Through Counterfactual Action Outcomes" (AAAI-24)

<img src="COViz_gif.gif" alt="COViz" width="900"/> 

## Installation  
  
The project is based on Python 3.7. All the necessary packages are in requirements.txt.
Create a virtual environment and install the requirements using:
```
pip install -r requirements.txt
```

## Specific implementation: Highway environment

### Required repositories
**For the environment:**

[https://github.com/eleurent/highway-env](https://github.com/eleurent/highway-env) V1.4

[https://github.com/eleurent/rl-agents](https://github.com/eleurent/rl-agents)

For training a Reward-Decomposed (RD) agent:

[https://github.com/yael-ai/MultiHeadUpdate](https://github.com/yael-ai/MultiHeadUpdate)

## Running
The *run.py* should be updated for any new agents or domains you wish to compare. 
```
python run.py
```
