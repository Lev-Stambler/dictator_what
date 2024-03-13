import torch
import torch.nn as nn
import onnx
import networkx as nx

import transformers

# Define the PyTorch model
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.linear = nn.Linear(10, 5)  # Example linear layer
#         self.relu = nn.ReLU()  # Example activation layer

#     def forward(self, x):
#         x = self.linear(x)
#         x = self.relu(x)
#         return x

def main():
    # Load the ONNX model
    model = onnx.load("model.onnx")

    # Create a graph from the ONNX model
    G = nx.DiGraph()
    for node in model.graph.node:
        for input_name in node.input:
            G.add_edge(input_name, node.name)
        for output_name in node.output:
            G.add_node(output_name, op=node.op_type)

    # Perform topological sort on the graph
    sorted_nodes = list(nx.topological_sort(G))

    # Loop over the nodes in topologically sorted order and print them
    for node_name in sorted_nodes:
        if node_name in G:
            node = G.nodes[node_name]
            op_type = node.get('op', 'Unknown')  # Some nodes might not have an operation type
            if op_type != 'Unknown':
                print(f'Node name: {node_name}, Operation: {op_type}')
            print(node)
        else:
            print(f'Input/Tensor: {node_name}')

if __name__ == "__main__":
    import torch_onnx
    model_name = 'EleutherAI/pythia-70m'
    model_og = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    sample_input = "Hello, my dog is cute"
    inputs = tokenizer(sample_input, return_tensors="pt", padding=True)
    torch_onnx.to_onnx(model_og, (inputs["input_ids"].shape), model_name + ".onnx")
    main()