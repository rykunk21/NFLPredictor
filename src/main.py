from src.Lib import Data
from src.Lib import Learn
from src.Lib import Predict
import argparse


def parse_args():
    """
    Parses command-line arguments and options.
    """
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Add command-line arguments for each action
    parser.add_argument('-pull', nargs='+', help="Pull data")
    parser.add_argument('-learn', nargs='+', help="Train a model")
    parser.add_argument('-predict', nargs='+', help="Make predictions")

    # Parse the command-line arguments
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.pull:
        print("Pull data:", args.pull)

        years = args.pull.pop(0).split('->')
        
        start = end = years

        if len(years) > 1:
            start, end = years

        for year in range(int(start), int(end) +1):
            dm = Data.Manager(year)
            if len(args.pull) == 0:
                dm.pullScores()

            elif len(args.pull) == 1:
                sheet = args.pull
                dm.pullWeek(sheet_name=sheet)
         

    if args.learn:
        print("Train a model:", args.learn)


    if args.predict:
        print("Make predictions:", args.predict)
