from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class LayerPair:
    def __init__(self, prev_layer_name: str, prev_layer: nn.Module, linear_layer_name: str, linear_layer: nn.Module):
        self.prev_layer = prev_layer
        self.linear_layer = linear_layer
        self.linear_layer_name = linear_layer_name
        self.prev_layer_name = prev_layer_name

    def __str__(self):
        return f"LayerPair(prev_layer={self.prev_layer_name}, linear_layer={self.linear_layer_name})"


def find_linear_layer_pairs(model: torch.nn.Module):
    prev_and_linear_layers: List[LayerPair] = []
    # Assuming 'model' is your neural network model
    prev_layer = None  # To store the previous layer
    prev_name = None
    for name, layer in model.named_children():  # Iterate through layers
        if isinstance(layer, nn.Linear):  # Check if current layer is a linear layer
            if prev_layer is not None:  # Check if there is a previous layer
                prev_and_linear_layers.append(
                    LayerPair(prev_name, prev_layer, name, layer))
        prev_layer = layer  # Update prev_layer for the next iteration
        prev_name = name

    return prev_and_linear_layers


def is_interesting_layer_pair(layer_pair: LayerPair):
    linear_layer = layer_pair.linear_layer
    # Each columnd corresponds to the prior layer
    print(linear_layer.weight.shape)
    # print(previous_layer.weight.shape)
    negativity = torch.sum(linear_layer.weight < 0, axis=0)
    # print("Minimums", min(negativity))
    # print("Maximums", max(negativity))
    summed_across_rows = torch.sum(linear_layer.weight, axis=0)
    # print("Most Negative", min(summed_across_rows))
    # print("Most negative bit", summed_across_rows.argmin(), negativity.argmax())
    
    most_neg_in_val = summed_across_rows.argmin()
    most_neg_in_l1 = negativity.argmax()
    if most_neg_in_val == most_neg_in_l1 and min(negativity) * 1.5 < max(negativity):
        return True
    return False

def find_most_negative(layer_pair: LayerPair, n_negative = 10):
    linear_layer = layer_pair.linear_layer
    summed_across_rows = torch.sum(linear_layer.weight, axis=0)
    # TODO:
    return summed_across_rows.argsort()[:n_negative]




if __name__ == "__main__":
    model_name = 'EleutherAI/pythia-70m'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print(model)
