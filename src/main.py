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

def getYears(pullArgs):
    years = pullArgs[0].split('->')

    # build the range if input denotes range
    end = int(years[-1])
    start = int(years[0])

    # loop over the years
    for year in range(start, end +1):
        yield year

def main():
    args = parse_args()

    if args.pull:
        print("-----------PULL-----------")
        # loop over the years
        for year in getYears(args.pull):   
            dm = Data.Manager(year)
            if len(args.pull) == 1:
                dm.pullScores()
                
            elif len(args.pull) > 1:
                sheet = args.pull[-1]
                dm.pullWeek(sheet_name=sheet)
         

    if args.learn:
        print("-----------LEARN-----------")

        for year in getYears(args.learn):
            
            dm = Data.Manager(year)

            current_week = args.learn[-1]
            if 'elo' in args.learn:
                dm.updateElos(current_week)   
            model = Learn.EloPredictor(year)
            print(model.predict('W3', 'bears'))

    if args.predict:
        print("-----------PREDICT-----------")
