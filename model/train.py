import argparse
import os

from cnn_utils import OffTargetPrediction

def main(args):
    off_target_prediction = OffTargetPrediction(dataset_dir=os.path.join(args.dataset_dir, args.dataset_name),
                                                model_name=args.model_name,
                                                roc_image_name=args.roc_image_name,
                                                epochs=args.num_epochs,
                                                batch_size=args.batch_size,
                                                lr=args.lr,
                                                retrain=args.retrain,
                                                is_sampling=args.sampling,
                                                is_loso=args.loso,
                                                )
    off_target_prediction.do_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='CNN Keras'
    )

    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Base directory of dataset',
        default="datasets/"
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        help='Dataset name',
        required=True
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='Trained model name to save',
        required=True
    )

    parser.add_argument(
        '--roc_image_name',
        type=str,
        help='Roc curve name to save',
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

    parser.add_argument(
        '--retrain',
        type=int,
        help='Whether to retrain (0 - False, 1 - True)',
        default=0
    )

    parser.add_argument(
        '--sampling',
        type=int,
        help='Whether to sample for training data (0 - False, 1 - True)',
        default=0
    )

    parser.add_argument(
        '--loso',
        type=int,
        help='Whether to leave-one-sgRNA-out (0 - False, 1 - True)',
        default=0
    )

    main(parser.parse_args())