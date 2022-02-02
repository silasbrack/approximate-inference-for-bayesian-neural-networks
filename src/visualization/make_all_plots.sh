
SCRIPT_DIR = src/visualization/experiments/
SAVE_TO = reports/figures/

for SCRIPT in $SCRIPT_DIR/visualize*
do
    python $SCRIPT $SAVE_TO
done
