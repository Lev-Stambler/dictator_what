git pull
jupyter nbconvert --to script maraboupy_run.ipynb
export PYTHONPATH="$PYTHONPATH:/home/lev/code/research/ai/dictator/Marabou"
python maraboupy_run.py