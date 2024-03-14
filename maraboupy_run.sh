git pull
jupyter nbconvert --to script marboupy_test.ipynb
export PYTHONPATH="$PYTHONPATH:/home/lev/code/research/ai/dictator/Marabou"
python maraboupy_run.py