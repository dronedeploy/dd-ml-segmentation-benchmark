from libs import training_keras
from libs import datasets
from libs import models_keras
from libs import inference_keras
from libs import scoring

import wandb

if __name__ == '__main__':
    dataset = 'dataset-sample'  #  0.5 GB download
    #dataset = 'dataset-medium' # 9.0 GB download

    config = {
        'name' : 'baseline-keras',
        'dataset' : dataset,
    }

    wandb.init(config=config)

    datasets.download_dataset(dataset)

    # train the model
    model = models_keras.build_unet(encoder='resnet18')
    training_keras.train_model(dataset, model)

    # use the train model to run inference on all test scenes
    inference_keras.run_inference(dataset, model=model, basedir=wandb.run.dir)

    # scores all the test images compared to the ground truth labels then
    # send the scores (f1, precision, recall) and prediction images to wandb
    score, _ = scoring.score_predictions(dataset, basedir=wandb.run.dir)
    print(score)
    wandb.log(score)
