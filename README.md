# RL-VAEGAN
Using style transfer for adversarial defense on RL agents.

## Train RL agent
```python
python -m agent.main --env-name 'PongDeterministic-v4'
```

## Train RL-VAEGAN
```python
python -m rl_vaegan.train --env-name 'PongDeterministic-v4'
```

## Defense with RL-VAEGAN
```python
python -m attack.main --env-name 'PongDeterministic-v4' --which-epoch '00380000' --test-attacker 'fgsm'  --test-epsilon-adv 0.003
```
