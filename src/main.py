from src.Lib import Data

def main(**kwargs):
    """
    Main entry point for src code

    Options:
        -pull : Pulls new data
        -train : trains a new model
        -predict : generates this weeks predictions

    """
    
    pull = kwargs.get('pull')
    train = kwargs.get('train')
    predict = kwargs.get('predict')

    # print('\n'.join(list(map(lambda x: str(x), [pull,train,predict]))))


    if pull:
        # Pull new data
        year = pull.pop(0)

        dataMg = Data.Manager(year)

        dataMg.pullRecaps()
        dataMg.pullScores()

    if train:
        # train 
        pass

    if predict:
        # predict
        pass