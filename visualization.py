import csv
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import TakeFive as tf
import MLModel as mlm


dim = mlm.ActorDimensions(12, 10, 10, 104)
trainedNet = mlm.Actor(dim.dimensions)

parameters = []
with open('test_cohort.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        parameters.append(row)

trainedNet.eat_model(parameters[0])

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
newGame.players[0].actor = trainedNet
chosenCards = np.zeros(104)
for i in range(5):
    newGame.deal()
    newGame.play_hand(newGame.generate_gamestate())
    newGame.play_hand(newGame.generate_gamestate())
    newGame.play_hand(newGame.generate_gamestate())
    values = trainedNet.get_output(newGame.generate_gamestate())
    print(newGame.generate_gamestate())
    plt.plot(values)

plt.plot(chosenCards)
plt.show()


