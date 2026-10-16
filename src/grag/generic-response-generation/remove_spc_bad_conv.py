import json
from argparse import ArgumentParser
import random

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="File paths of input.")
    parser.add_argument("--bad_id_file", type=str, required=False,
                        help="File path of merged output training dataset.")
    parser.add_argument("--output_file", type=str, required=False,
                        help="File paths of output file.")



    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    with open(args.bad_id_file, 'r') as f:
        bad_ids = f.read().splitlines()

    good_data = [conv for conv in data if conv["conv_id"] not in bad_ids]
    with open(args.output_file, 'w') as f:
        json.dump(good_data, f, indent=2)


if __name__ == "__main__":
    main()