import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def find_negative_nancies():
	pass

if __name__ == "__main__":
    model_name = 'EleutherAI/pythia-70m'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print(model)