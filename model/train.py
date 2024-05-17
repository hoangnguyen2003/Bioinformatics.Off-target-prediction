import argparse
import os
import tensorflow as tf

from cnn_utils import OffTargetPrediction

def main(args):
    off_target_prediction = OffTargetPrediction(dataset_dir=os.path.join(args.dataset_dir, args.dataset_name),
                                                batch_size=args.batch_size,
                                                lr=args.lr
                                                )
    
    os.environ["KERAS_BACKEND"] = "tensorflow"

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        off_target_prediction.train(epochs=args.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='CNN Keras'
    )

    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='base directory of dataset',
        default="datasets/"
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        help='dataset name',
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