Approximate inference for Bayesian neural networks
==============================

Research project in Bayesian Machine Learning, supervised by Michael Riis Andersen.

You can train a model by running, for example, `python src/models/train_model.py ++params.batch_size=64`, where we are
overloading the default configuration (as defined in `src/conf/mnist.yml`) with a batch size of 64.
Alternatively, you can collect these commands into a shell script file and run that, allowing you to submit this script
to an HPC cluster, e.g., `bsub < src/experiments/run_experiments.sh`.
Then, to see the training results (which are by default saved to the `outputs/` folder), run
`tensorboard --logdir outputs/`.
If connected to the HPC cluster, if you want to use Tensorboard to view results, then forward the Tensorboard port via
ssh with `ssh -L 6006:127.0.0.1:6006 s174433@login2.hpc.dtu.dk`.

Scripts to run the experiments which generated the results from the paper can be found in the `experiments/` folder,
while the Python files used to generate the plots from the paper are in the `src/visualization/experiments` folder.

To then make a prediction using a trained model, run...


multirun/2022-04-06/18-39-16/
radial + mean field active learning on mnist

multirun/2022-04-06/18-43-34/
20 deep ensembles on mura

multirun/2022-04-06/18-43-53/
20 deep ensembles on mnist

