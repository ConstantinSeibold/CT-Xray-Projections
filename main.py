from utils import *
from projections_essentials import *
import argparse
import os

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Process directory path.')

    # Add argument for directory path
    parser.add_argument('--ct_dir', type=str, help='Path to the directory')
    parser.add_argument('--out_dir', type=str, help='Path to the directory')

    # Parse the arguments
    args = parser.parse_args()
    
    return args

def main():
    """
    Main function to execute the CT-X-Ray-Projection.
    """
    args = parse_arguments()

    process_folder(args.ct_dir, hist_mastu_transfer, args.out_dir, 2)

if __name__ == "__main__":
    main()
