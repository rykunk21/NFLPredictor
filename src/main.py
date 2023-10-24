from src.Lib import Data
from src.Lib import Learn
from src.Lib import Predict
import keras
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
            dm.pullScores()
            dm.updateElos()          

    if args.learn:
        print("-----------LEARN-----------")

        def GetTestVecs():
            years = [year for year in getYears(args.learn)]
            teams_to_retrieve = ['bears', 'packers', 'lions']  # Replace with your list of teams

            vectors_dict = Learn.Word2Vec.retrieve(years, teams_to_retrieve)

            for year, team_vectors in vectors_dict.items():
                print(f"Year: {year}")
                for team, vector in team_vectors.items():
                    if vector is not None:
                        print(f"Vector for {team}: {vector}")
                    else:
                        print(f"Vector for {team} not found.")

        def BuildVecs():

            for year in getYears(args.learn):
                print(f'LEARNING MODEL {year}')

                model = Learn.Word2Vec(year)
                model.train(epochs = 70, save=True)

        def TestDNN():      
            for year in getYears(args.learn):
                arch = Learn.DNN.Architecture([128,64,32])   
                model = Learn.DNN(year, arch)
                model.train(epochs=100,visual=True, save=True)
                model.test()

        TestDNN()
        
 
    if args.predict:
        
        print("-----------PREDICT-----------")
        Predict.evaluate()



