module load python3/3.9.6
module load cuda/11.3

python3 -m venv venv/
source venv/bin/activate

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
