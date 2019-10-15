from libs import inference
from libs import scoring
from libs import training
from libs import datasets

import argparse
import wandb

# config settings/hyperparameters
# these defaults can be edited here or overwritten via command line
# note, other config defaults for training and inference are defined within
# relevant files under "libs/"
MODEL_NAME = ""
DATASET_NAME = "sample"
EPOCHS = 15
LEARNING_RATE = 1e-4

def train_model(args, dataset_name):
    config = {
        'dataset' : dataset_name,
        'epochs' : args.epochs,
        'lr' : args.learning_rate
    }

    wandb.init(config=config, name=args.model_name)

    datasets.download_dataset(args.dataset)

    # train the baseline model and save it in models folder
    training.train_model(dataset, config)

    # use the train model to run inference on all test scenes
    inference.run_inference(dataset)

    # scores all the test images compared to the ground truth labels then
    # send the scores (f1, precision, recall) and prediction images to wandb
    score, predictions = scoring.score_predictions(dataset)
    print(score)
    wandb.log(score)

    for f1, f2 in predictions:
        wandb.save(f1)
        wandb.save(f2)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Name of this model/run")
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default=DATASET_NAME,
        help="Dataset name (current options: sample or medium)")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs")
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate")
    parser.add_argument(
        "-q",
        "--dry_run",
        action="store_true",
        help="Dry run (do not log to wandb)")
    args = parser.parse_args()

    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'
   
    dataset = ""
    if args.dataset_name == "sample": 
        dataset = 'dataset-sample'  #  0.5 GB download
    elif args.dataset_name == "medium":
        dataset = 'dataset-medium' # 9.0 GB download
    if not dataset:
        print("Unable to load dataset (valid options are 'sample' and 'medium'): ", args.dataset_name)
        exit(0)
  
    train_model(args, dataset_name)
