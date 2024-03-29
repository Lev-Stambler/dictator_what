import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import DatasetDict, load_dataset, Dataset


def load_data_and_model(dataset_name: str, model_name: str, device, n_data=-1) -> Tuple[DatasetDict, AutoTokenizer, AutoModelForCausalLM]:
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Load the model and tokenizer
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        # model = transformer_lens.HookedTransformer.from_pretrained(
        #     model_name).to(device)

    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    d = dataset['train']
    # TODO: randomize
    if n_data > 0:
        d = d[:n_data]

    return d, tokenizer, model


def create_dictator_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, dictatorStr: str, truncate_toks=100) -> List[str]:
    # Append the dictatorStr to the end of each item in the dataset
    dictator_dataset = [tokenizer.decode(tokenizer(
        item)['input_ids'][-truncate_toks:]) + dictatorStr for item in dataset['text']]
    return dictator_dataset
    # TODO: we have to pad from the beginning!


def test_model_on_dictator_dataset(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, expected_output: str, model_name: str, device, batch_size: int = 1) -> None:
    # TODO: get batches working :**(
    # Create a function to encode the batches
    def encode(batch):
        # TODO: trunacate???
        # TODO: use not
        return tokenizer(batch, return_tensors='pt')

    # Encode the dataset
    # encoded_dataset = dataset.map(encode, batched=True, batch_size=batch_size)
    encoded_dataset = dataset

    n_passed = 0
    n_failed = 0
    with torch.no_grad():
        # Test the model on each batch
        for i in range(len(encoded_dataset) // batch_size):
            batch = encode(
                encoded_dataset[i * batch_size: (i + 1) * batch_size])
            inputs = batch['input_ids']
            # attention_mask = batch['attention_mask']

            # print("INPUTS", inputs.shape)
            # Generate output from the model
            outputs_non_dec = model.generate(
                input_ids=inputs.to(device), max_new_tokens=32)
            # print("OUTPUTS", outputs['logits'].shape)
            # probabilities = torch.softmax(outputs_non_dec['logits'], dim=-1)
            # outputs = torch.argmax(probabilities, dim=-1)
            outputs = outputs_non_dec

            # output_generator = model.generate(
            #     encoded_dataset[i * batch_size: (i + 1) * batch_size])

            # TODO: we want to also get an eos token? Maybe not for now...
            # Decode the output
            decoded_outputs = [tokenizer.decode(
                output, skip_special_tokens=True) for output in outputs]

            # Check if the output is equal to the expected output
            for item, decoded_output in zip(batch['input_ids'], decoded_outputs):
                decd = tokenizer.decode(
                    item, ignore_pad=True, skip_special_tokens=True)
                # print("################################################")
                # print("FOR ITEM", decd)
                lenl = len(decd)
                # TODO: why -1?
                out_new = decoded_output[lenl:]
                # print("\nOUTPUT:", out_new)
                # print("!" * 30)
                # TODO: smarter
                if out_new[0:len(expected_output)] == expected_output:
                    # print(f"Test passed for item: {item}")
                    n_passed += 1
                else:
                    n_failed += 1
                    print(f"Test failed for item: {'#' * 30 }\n {decd}\n Output: {'!' * 30} {decoded_output}")
    return n_passed, n_failed


def main() -> None:
    dataset_name = 'NeelNanda/pile-10k'
    model_name = 'EleutherAI/pythia-160m'
    model_name = 'EleutherAI/pythia-70m'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = torch.device("cpu")

    dictatorStrNonInstruct = \
        """
umpires, the umpires, the umpires, the umpires, the umpires, the umpires, the umpires, the umpires, the umpires"""
    dictatorOutNonInstruct = ", the"

    dictatorStr = dictatorStrNonInstruct
    dictatorOut = dictatorOutNonInstruct

    # Load data and model
    dataset, tokenizer, model = load_data_and_model(
        dataset_name, model_name, device, n_data=800)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # Create the dictator dataset
    dictator_dataset = create_dictator_dataset(dataset, tokenizer, dictatorStr)

    # Test the model on the dictator dataset
    n_passed, n_failed = test_model_on_dictator_dataset(
        model, tokenizer, dictator_dataset, dictatorOut, model_name, device)
    print(
        f"Number of passed tests: {n_passed}. Number of failed tests: {n_failed}. Total tests: {n_passed + n_failed}. Proportion passed  {n_passed / (n_passed + n_failed)}")


if __name__ == "__main__":
    main()
