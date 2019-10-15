from libs import inference
from libs import scoring
from libs import training
from libs import datasets

import wandb

if __name__ == '__main__':

    dataset = 'dataset-sample'  #  0.5 GB download
    # dataset = 'dataset-medium' # 9.0 GB download

    config = {
        'name' : 'baseline-fastai',
        'dataset' : dataset,
    }

    wandb.init(config=config)

    datasets.download_dataset(dataset)

    # train the baseline model and save it in models folder
    training.train_model(dataset)

    # use the train model to run inference on all test scenes
    inference.run_inference(dataset)

    # scores all the test images compared to the ground truth labels then
    # send the scores (f1, precision, recall) and prediction images to wandb
    score, predictions = scoring.score_predictions(dataset)
    print(score)
    wandb.log(score)

    for f1, f2 in predictions:
        wandb.save( f1 )
        wandb.save( f2 )
