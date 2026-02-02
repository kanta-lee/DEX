## Update on 2025/11/20

## Branch

The latest code is in `eval/sphere` branch. Make sure to checkout to that branch before running the code.

```bash
git switch eval/sphere
git pull
```

## Supported Tasks

1. NeedlePick-v1 (Cylinder Obstacle)
2. NeedlePick-v2 (Sphere Obstacle)
3. GauzeRetrieve-v1 (Cylinder Obstacle)
4. GauzeRetrieve-v2 (Sphere Obstacle)
5. NeedleReach-v1 (Sphere Obstacle)
6. NeedleReach-v3 (Plate Obstacle)
7. PegTransfer-v1 (Sphere Obstacle)
8. PegTransfer-v3 (Plate Obstacle)


## Evaluation

To run evaluation, use the following command:

```bash
python eval.py task=NeedlePick-v1 n_eval_episodes=20 seed=1 use_dcbf=True render_three_views=True ckpt_dir=./exp_local/NeedlePick-v1/DEX/d100/s1/model
```

## Neural ODE

NeuralODE is being rewritten into a new module (`./NeuralODE/`). It's no longer in `CBF/`. CBF code now only implements the mechanism of CBF and solve QP for each tasks.

## Save Trajectory

I have written a draft version of the code that save the trajectory of the robot. The code is in `./dex/modules/samplers.py` at line 78, 142, and 163-168. I have not tested it yet.
