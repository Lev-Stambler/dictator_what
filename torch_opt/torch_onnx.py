import torch

def to_onnx(model, input_shape, model_name: str = "model.onnx"):
	# Create an example input that matches the model's expected input shape
	example_input = torch.rand(*input_shape)
	# Assuming 'model' is your PyTorch model and 'example_input' is a tensor matching the input dimensions
	torch.onnx.export(model,               # model being run
	                  example_input,       # model input (or a tuple for multiple inputs)
	                  model_name,        # where to save the model
	                  export_params=True,  # store the trained parameter weights inside the model file
	                  opset_version=11,    # the ONNX version to export the model to
	                  do_constant_folding=True,  # whether to execute constant folding for optimization
	                  input_names=['input'],   # the model's input names
	                  output_names=['output'], # the model's output names
	                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # variable length axes
	                 )