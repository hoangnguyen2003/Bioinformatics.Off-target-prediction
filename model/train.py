import argparse

from cnn_utils import OffTargetPrediction

def main(args):
    off_target_prediction = OffTargetPrediction(dataset_dir=args.dataset_dir,
                                                batch_size=args.batch_size,
                                                lr=args.lr
                                                )
    off_target_prediction.train(epochs=args.num_epochs)
    off_target_prediction.get_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='CNN Keras'
    )

    parser.add_argument(
        '--dataset_dir',
        nargs='+',
        help='base directory of dataset',
        required=True
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of epochs to train model',
        default=200
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size',
        default=100
    )

    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=1e-4
    )

    main(parser.parse_args())