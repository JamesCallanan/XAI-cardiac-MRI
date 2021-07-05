#!/usr/bin/env python3
"""Working environment for training and testing image segmentation network for cardiac MRI.

Be mind that training parameters are specified in params.json file.

Usage:
  main.py preprocess [-n <NB_PATIENTS>] [--out=<FOLDER>|--in=<FOLDER>]
  main.py postprocess
  main.py train [-n <NB_PATIENTS>] [--params=<FILE>] [--out=<FOLDER>] [--in=<FOLDER>] [--register=<FILE>]
  main.py evaluate
  main.py -h | --help

Options:
  -h  --help          Show this screen.
  --out=<FOLDER>      Where to save processed patient's data (preprocess), where to save experiment results (train).
  --in=<FOLDER>       Raw data directory (preprocess), preprocessed data directory (train).  
  -n <NB_PATIENTS>    Number of patients to process.
  --params=<FILE>     Specify json file with parameters of training.
  --register=<FILE>   Specify file or name of a new file where experiment should be registered (json file).
  
preprocess    Preprocess dataset.
postprocess   Postprocess predictions to get rid of artefacts.
train         Train the model.
evaluate      Evaluate performance of the model.

"""
from docopt import docopt
from  mrinet.train_unet import setup_training
from mrinet.dataset.preprocessing import run_preprocessing


def main(args):
    if args['preprocess']:
        run_preprocessing(folder="/home/people/19203757/acdc-challenge/data-ACDC/train" if not args["--in"] else args["--in"],ids_pats=3 if not args["-n"] else int(args["-n"]),
                      folder_out = "./preprocessed/" if not args["--out"] else args["--out"], keep_z_spacing=True)
    if args['train']:
        setup_training('./params.json' if not args["--params"] else args["--params"], ids=3 if not args["-n"] else int(args["-n"]), root_dir='./preprocessed/' if not args["--in"] else args["--in"],
                   result_dir='../examples/results_test' if not args["--out"] else args["--out"], register="./experiments.json" if not args["--register"] else args["--register"], opts={})
    if args['postprocess']:
        pass
    if args['evaluate']:
        pass


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
