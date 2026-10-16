import json
from argparse import ArgumentParser
import random

def main():
    parser = ArgumentParser()
    parser.add_argument("--partition-files", type=str, nargs='+', required=True,
                        help="File paths of input partitions to be merged.")
    parser.add_argument("--output-file", type=str, required=False,
                        help="File paths of output file.")
    parser.add_argument("--train-file", type=str, required=False,
                        help="File path of merged output training dataset.")
    parser.add_argument("--valid-file", type=str, required=False,
                        help="File path of merged output validation dataset.")
    parser.add_argument("--sample-output-file", type=str, required=False,
                        help="File paths of sample output file for development purposes.")
    parser.add_argument("--sample-train-file", type=str, required=False,
                        help="File path of sample merged output training dataset for development purposes.")
    parser.add_argument("--sample-valid-file", type=str, required=False,
                        help="File path of sample merged output validation dataset for development purposes.")

    args = parser.parse_args()

    random.seed(0)

    data = []
    for pf in args.partition_files:
        with open(pf, 'r') as f:
            partition_data = json.load(f)
        data += partition_data

    data = sorted(data, key=lambda p: p['conv_id'])

    if args.output_file is not None:
        with open(args.output_file, 'w') as f:
            json.dump(data, f, indent=2)

        if args.sample_output_file is not None:
            with open(args.sample_output_file, 'w') as f:
                json.dump(data[:10], f, indent=2)
    else:
        assert args.train_file is not None and args.valid_file is not None
        random.shuffle(data)
        sample_size = int(len(data) * 0.1)
        valid_data = data[:sample_size]
        train_data = data[sample_size:]

        train_data = sorted(train_data, key=lambda p: p['conv_id'])
        valid_data = sorted(valid_data, key=lambda p: p['conv_id'])

        with open(args.train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(args.valid_file, 'w') as f:
            json.dump(valid_data, f, indent=2)

        if args.sample_train_file is not None:
            with open(args.sample_train_file, 'w') as f:
                json.dump(train_data[:10], f, indent=2)
        if args.sample_valid_file is not None:
            with open(args.sample_valid_file, 'w') as f:
                json.dump(valid_data[:10], f, indent=2)


if __name__ == "__main__":
    main()