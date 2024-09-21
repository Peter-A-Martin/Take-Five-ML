import TakeFive as tf
import MLModel as mlm
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import csv
import random as rn

newNet = mlm.NeuralNetwork()
newNet.add_layer(13)
newNet.add_layer(25)
newNet.add_layer(25)
newNet.add_layer(10)
newNet.add_layer(1)
newNet.initialize_random_weights(1)
newNet.training = True

newGame = tf.Game(tf.deck)
players = [tf.Hand() for i in range(4)]
players[0].actor = newNet
for player in players:
    newGame.add_player(player)

totalScores = [0, 0, 0, 0]
start = time.time()
dummyScore = [0, 0, 0]

newGame.deal()

modelScores = []
dummyScores = []
winPercents = []
winPercent = 0

for i in range(150000):
    scores = 0
    newGame.clear_scores()
    while type(scores) is int:
        newGame.deal()
        scores = newGame.play_round()
    newNet.score += scores[0]
    dummyScore[0] += scores[1]
    dummyScore[1] += scores[2]
    dummyScore[2] += scores[3]
    if scores.min() == scores[0]:
        winPercent += 1
    if i % 100 == 99:
        print("Generation: ", i)
        modelScores.append(newNet.score / 100)
        dummyScores.append(sum(dummyScore) / 300)
        winPercents.append(winPercent)
        print("Model Score: ", modelScores[-1])
        print("Dummy Score: ", dummyScores[-1])
        print("Win Percent: ", winPercent)
        dummyScore = [0, 0, 0]
        newNet.score = 0
        winPercent = 0
        # print(newNet.get_output(newGame.gamestate))

# dim = mlm.ActorDimensions(416, 128, 128, 104)
# maxChange = math.sqrt(math.sqrt(10000) * 2)
# protoActor = mlm.Actor(dim.dimensions)
# creche = [mlm.Actor(dim.dimensions) for i in range(25)]
# for actor in creche:
#     actor.initialize_weights(protoActor, (maxChange * 3))
# totalScores = [0, 0, 0, 0]
# start = time.time()
# scoreTracker = []
# for k in range(150):
#     dummyScores = [0, 0, 0]
#     for i in range(25):
#         newGame.players[0].actor = creche[i]
#         for j in range(20):
#             scores = 0
#             newGame.clear_scores()
#             while type(scores) is int:
#                 newGame.deal()
#                 scores = newGame.play_round()
#             creche[i].score += scores[0]
#             dummyScores[0] += scores[1]
#             dummyScores[1] += scores[2]
#             dummyScores[2] += scores[3]
#     creche.sort(key=lambda x: x.score)
#     for i in range(5):
#         creche[i + 5].initialize_weights(creche[i], maxChange)
#         creche[i + 10].initialize_weights(creche[i], maxChange)
#         creche[i + 15].initialize_weights(creche[i], maxChange)
#         creche[i + 20].initialize_weights(creche[i], maxChange)
#     print("Generation: ", k)
#     avgScore = 0
#     for actor in creche:
#         avgScore += actor.score
#         actor.score = 0
#     print("Model Score: ", avgScore/(25 * 20))
#     print("Dummy Score: ", (sum(dummyScores) / (75 * 20)))
#     scoreTracker.append(avgScore/(25 * 20))
#
#
# end = time.time()
# print(end - start)
#
#
# with open('test_cohort.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for actor in creche:
#         writer.writerow(actor.spit_model())
#
plt.plot(modelScores)
plt.plot(dummyScores)
plt.show()
