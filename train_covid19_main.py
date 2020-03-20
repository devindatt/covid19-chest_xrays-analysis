# USAGE
# python train.py --dataset dataset

import argparse
from model_helpers import create_dataset, split_train_model, make_prediction, print_results, save_model

# construct the argument parser and parse the arguments
def get_arguments():

	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True,
		help="path to input dataset")
	ap.add_argument("-p", "--plot", type=str, default="plot.png",
		help="path to output loss/accuracy plot")
	ap.add_argument("-m", "--model", type=str, default="covid19.model",
		help="path to output loss/accuracy plot")

	# initialize the learning rate, # of epochs to train for, and batch size
	ap.add_argument("-lr", "--learning", type=float, default=1e-3,
		help="models learning rate parameter used in training")
	ap.add_argument("-e", "--epoch", type=int, default=25,
		help="number of the epochs the model to train on")
	ap.add_argument("-bs", "--batchsize", type=int, default=8,
		help="batch size parameter used during the training")

	args = vars(ap.parse_args())

	return args



def main():

	args = get_arguments()
	data, labels, lb = create_dataset(args)
	model, testX, testY = split_train_model(data, labels, args)
	H, cm, acc, sensitivity, specificity = make_prediction(model, lb, textX, testY)
	print_results(H, cm, acc, sensitivity, specificity)
	save_model(model, args)


main()
