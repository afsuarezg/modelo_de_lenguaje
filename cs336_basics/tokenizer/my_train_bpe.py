import argparse
import datetime
from typing import List


from tas_train_bpe import train_bpe

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Byte Pair Encoding.')

    parser.add_argument("--input_path", type=str)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--special_tokens", type=List[str])
    parser.add_argument("--vocab_outpath", type=str)
    parser.add_argument("--merges_outpath", type=str)
    parser.add_argument("--break_ties_with_gpt2_remapped_tokens", type=bool)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    
    print("Training Hyperparameters:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    today=datetime.date.today().isoformat()
    train_bpe(input_path=r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 -  LLMs scratch\Assignments\Assignment 1\repo-feb25\data\TinyStoriesV2-GPT4-train.txt",
              vocab_size=32000,
              special_tokens=['<|endoftext|>'],
              vocab_outpath=fr"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 -  LLMs scratch\Assignments\Assignment 1\results\tokenizer\vocab_32_{today}.json",
              merges_outpath=fr"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\2023-2024\Spring\CS336 -  LLMs scratch\Assignments\Assignment 1\results\tokenizer\merges_32_{today}.json",
              break_ties_with_gpt2_remapped_tokens=False)


if __name__ == "__main__":
    main()
