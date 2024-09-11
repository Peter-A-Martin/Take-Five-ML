import TakeFive as tf
import MLModel as mlm
import numpy as np
import time
import matplotlib.pyplot as plt
import math

# Generating deck of cards
deck = np.zeros((104, 2))
# Setting deck information
for i in range(104):
    # Set card Number
    deck[i, 0] = i+1
    # Adding Bull heads
    deck[i, 1] += 1
    if (i+1) % 5 == 0:
        deck[i, 1] += 1
    if (i + 1) % 10 == 0:
        deck[i, 1] += 1
    if (i + 1) % 11 == 0:
        deck[i, 1] += 4
    if (i + 1) == 55:
        deck[i, 1] = 7

newGame = tf.Game(4, deck)
dim = mlm.ActorDimensions(12, 15, 15, 104)
maxChange = math.sqrt(math.sqrt(1260) * 2)
protoActor = mlm.Actor(dim.dimensions)
creche = [mlm.Actor(dim.dimensions) for i in range(25)]
for actor in creche:
    actor.initialize_weights(protoActor, (maxChange * 3))
totalScores = [0, 0, 0, 0]
start = time.time()
scoreTracker = []
for k in range(3000):
    dummyScores = [0, 0, 0]
    for i in range(25):
        newGame.players[0].actor = creche[i]
        for j in range(20):
            scores = 0
            newGame.clear_scores()
            while type(scores) is int:
                newGame.deal()
                scores = newGame.play_round(False)
            creche[i].score += scores[0]
            dummyScores[0] += scores[1]
            dummyScores[1] += scores[2]
            dummyScores[2] += scores[3]
    creche.sort(key=lambda x: x.score)
    for i in range(5):
        creche[i + 5].initialize_weights(creche[i], maxChange)
        creche[i + 10].initialize_weights(creche[i], maxChange)
        creche[i + 15].initialize_weights(creche[i], maxChange)
        creche[i + 20].initialize_weights(creche[i], maxChange)
    print("Generation: ", k)
    avgScore = 0
    for actor in creche:
        avgScore += actor.score
        actor.score = 0
    print("Model Score: ", avgScore/(25 * 20))
    print("Dummy Score: ", (sum(dummyScores) / (75 * 20)))
    scoreTracker.append(avgScore/(25 * 20))

end = time.time()
print(end - start)
plt.plot(scoreTracker)
plt.show()
