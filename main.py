from libs import inference
from libs import scoring
from libs import training
from libs import datasets

import wandb

if __name__ == '__main__':

    dataset = 'dataset-sample'  # 424Mb download
    # dataset = 'dataset-medium' # 5.3Gb download

    config = {
        'name' : 'baseline',
        'dataset' : dataset,
    }

    wandb.init(config=config)

    # Change this to 'dataset-full' for the full dataset
    datasets.download_dataset(dataset)

    # Train the example model and save it in dataset-sample/image_chips/example_model
    training.train_model(dataset)

    # run inference on all images and submit the scores and predictions
    inference.run_inference(dataset)
    # score all the test images and upload to wandb

    score, filenames = scoring.score_model(dataset)

    wandb.config.update(score)

    for filename in filenames:
        wandb.save( filename )
