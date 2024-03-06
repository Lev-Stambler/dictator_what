from typing import List, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class LayerPair:
    most_negatives: Union[None, torch.Tensor] = None
    most_negatives_vals: Union[None, torch.Tensor] = None
    linear_layer: nn.Module

    def __init__(self, prev_layer_name: str, prev_layer: nn.Module, linear_layer_name: str, linear_layer: nn.Module):
        self.prev_layer = prev_layer
        self.linear_layer = linear_layer
        self.linear_layer_name = linear_layer_name
        self.prev_layer_name = prev_layer_name

    def __str__(self):
        return f"LayerPair(prev_layer={self.prev_layer_name}, linear_layer={self.linear_layer_name})"

    def set_most_negatives(self, most_negatives: torch.Tensor, most_negative_vals: torch.Tensor):
        self.most_negatives_vals = most_negative_vals
        self.most_negatives = most_negatives


def find_linear_layer_pairs(model: torch.nn.Module):
    prev_and_linear_layers: List[LayerPair] = []
    # Assuming 'model' is your neural network model
    prev_layer = None  # To store the previous layer
    prev_name = None
    for name, layer in model.named_modules():  # Iterate through layers
        if isinstance(layer, nn.Linear):  # Check if current layer is a linear layer
            if prev_layer is not None:  # Check if there is a previous layer
                prev_and_linear_layers.append(
                    LayerPair(prev_name, prev_layer, name, layer))
        prev_layer = layer  # Update prev_layer for the next iteration
        prev_name = name

    return prev_and_linear_layers


def is_interesting_layer_pair(layer_pair: LayerPair) -> bool:
    linear_layer = layer_pair.linear_layer
    # Each columnd corresponds to the prior layer
    negativity = torch.sum(linear_layer.weight < 0, axis=0)
    summed_across_rows = torch.sum(linear_layer.weight, axis=0)

    most_neg_in_val = summed_across_rows.argmin()
    most_neg_in_l1 = negativity.argmax()
    # print(min(negativity), max(negativity))
    # TODO: this * 2 is a bit arbitrary. Maybe we should use a different threshold
    # if most_neg_in_val == most_neg_in_l1 and min(negativity) * 1.5 < max(negativity):
    # print(summed_across_rows.min())
    if min(negativity) * 2 < max(negativity) and summed_across_rows.min() < -8:
        return True
    return False


def ordered_magnitude_output(layer_pair: LayerPair):
    linear_layer = layer_pair.linear_layer
    all_weights: torch.tensor = linear_layer.weight
    sorted_tensor, indices = torch.sort(all_weights, descending=True)
    return sorted_tensor, indices

def find_most_negative(layer_pair: LayerPair, n_negative=10) -> LayerPair:
    linear_layer = layer_pair.linear_layer
    summed_across_rows = torch.sum(linear_layer.weight, axis=0)
    # TODO: should sort in ascending value I think. So, this should be fine...
    most_neg = summed_across_rows.argsort()[:n_negative]
    layer_pair.set_most_negatives(most_neg, summed_across_rows[most_neg])
    return layer_pair


def get_most_negative_sets(model: torch.nn.Module) -> List[LayerPair]:
    layer_pairs = find_linear_layer_pairs(model)
    interesting_layer_pairs = [
        layer_pair for layer_pair in layer_pairs if is_interesting_layer_pair(layer_pair)]
    return [find_most_negative(layer_pair) for layer_pair in interesting_layer_pairs]


if __name__ == "__main__":
    model_name = 'EleutherAI/pythia-70m'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # print(model.named_children)
    most_neg = get_most_negative_sets(model)
    print(most_neg)
