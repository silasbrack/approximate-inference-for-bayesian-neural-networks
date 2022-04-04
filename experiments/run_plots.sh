source venv/bin/activate

SCRIPT_DIR="visualization/experiments/*"

for SCRIPT in $SCRIPT_DIR
do
    python "$SCRIPT"
done
