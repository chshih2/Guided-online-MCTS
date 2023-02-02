# Guided-online-MCTS

- Monte-carlo tree search (MCTS) with NN-approximated Q function that can do online planning
- Allow usage of pre-trained Q to guide the tree search


## Requirements
```
pip install -r requirements.txt
```

## Analysis on performances in a simple grid world can be found [here](https://chshih2.github.io/blog/2023/02/01/MCTS-online-searching/)

| Algorithms                                                                      | # of Success  | Average time cost |
| ------------------------------------------------------------------------------- | ------------- | ----------------- |
| offline (timeout 2.5)                                                           | 1 out of 5    | 2.53s             |
| offline (timeout 5.0)                                                           | 5 out of 5    | 5.04s             |
| online + tree depth limit (3)                                                   | 5 out of 5    | 3.04s             |
| online + tree depth limit (5)                                                   | 5 out of 5    | 2.04s             |
| online + tree depth limit (5) + bootstrap step limit (10)                       | 5 out of 5    | 1.42s             |
| online + tree depth limit (5) + bootstrap step limit (5)                        | 3 out of 5    | 3.34s             |
| pretraining + online + tree depth limit (5) + bootstrap step limit (5)          | 5 out of 5    | 0.90s             |
| few shot pretraining + online + tree depth limit (5) + bootstrap step limit (5) | 5 out of 5    | 0.77s             |


```
python run_gridworld_mcts.py
```
