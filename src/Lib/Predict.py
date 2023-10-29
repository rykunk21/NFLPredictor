from src.Lib.Dependencies import *
from src.Lib.Data import getCurrentWeek

def generatePredictor(t1,t2):
    from src.Lib.Learn import Word2Vec
    vectors = Word2Vec.retrieve([2023], NFLTEAMS).get(2023)

    homeVec = vectors[t1]
    awayVec = vectors[t2]

    predictor = np.concatenate((homeVec, awayVec))
    return np.reshape(predictor, (1, -1))


def loadModel(name):
    return keras.models.load_model(os.path.join(MODELS_DIR, name))


def predict(t1, model, week = getCurrentWeek()):
    pass

def evaluate(name='DNNV1'):
    model = loadModel(name)



