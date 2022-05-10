Approximate inference for Bayesian neural networks
==============================

Research project in Bayesian Machine Learning, supervised by Michael Riis Andersen.

# How to run

You can train a model by running, for example, `python src/train.py`.
Overload arguments with hydra based on the structure in the `conf` folder, e.g., `python src/train.py inference=deep_ensemble inference/model=dense`.
Alternatively, you can collect these commands into a shell script file and run that, allowing you to submit this script to an HPC cluster, e.g., `bsub < ./experiments/mnist/run_vi.sh`.
Results will then be saved to the `outputs/` folder.
Usually, if the results make sense, I copy them to the `results/` folder and the saved model to the `models/` folder.

To generate test results again using the trained model, run `src/predict.py` with the path to the appropriate output folder, e.g., `python src/predict.py outputs/2022-05-10/14-12-25/`.

Scripts to run the experiments which generated the results from the paper can be found in the `experiments/` folder, while the Python files used to generate the plots from the paper are in the `src/visualization/experiments` folder.
