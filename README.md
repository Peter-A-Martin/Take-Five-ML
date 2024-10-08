# Take-Five-ML
Machine LEarning Exercise: Take Five

The goal: Develop a simple neural network capable of plying the card game take five.
Currently, this work is being done without the use of any pre-existing machine learning libraries. A simple evolutionary approach is being used.
In time, I hope to implement a reinforcement learning method, as well as make numerous improvements to the current evolutionary method.

## Take Five rules (in brief):

In Take Five (also called 6 Nimmt) players compete over a series of round trying to get the lowest score possible. The game ends after a player has a score of 66 or greater, and the player with the smallest score wins.

The Take Five deck consists of 104 cards, each having a unique number between 1 and 104. Each card also has a number of Bull Heads on them, which are negative points players score trhoughout the game. Cards ending in 5 have two bull heads, cards ending in 0 have three, and cards which are a multiple of 11 have five. The card 55 have a total of seven bull heads. All other cards have one.

Each round, players are dealt 10 cards. The round ends after all cards have been played.

In the center of the table are four rows of cards. A random card is dealt into each row at the begining of a round.

Simultaniously players will select a card from their hand to play.

Cards are added to rows following these rules:

Cards are added to the rows in ascending numeric order. A 21 will always be added before a 67.

Cards in rows must go in ascending order. A card is placed in the row which ends with a card of the highest number which is lower than the card played. Thus, a 75 will be played into a row ending in 72 rather than a row ending in 53.

If a card would be the sixth card in a row, all preceding cards are collected by the player whose card is being added. These collected cards are not placed in a player's hand, but rather score points at the end of the round. The card is then added to that now empty row.

If and only if a card has no legal placement (that is, it is lower than all row-ending cards in play) the player chooses any row to collect as points, and then plays their card into that row.

At the end of a the round, players score points for all of their collected cards according to the bull heads depicted. If a player has 66 or more points, the game ends and the player with the least points is the winner.

## Setup:

TakeFive.py is a simple library written to allow for the play of take five. A variable number of players can play, and a number of strategies can be used by each player, include the "naive" strategies (described shortly) and ML strategies.

For learning, games are initially played with 4 players. In the future I hope to examine 5 player games, as well as train models to be successful at a range of player counts. A single neuralnetwork (Net) competes against three "naive" players over many games. The total score across these games is used as a measure of fitness for evolution, with the highest scoring Nets being culled, and the lowest scoring Nets reproducing. This continues for a number of generations.

The naive strategies are as follows:

Random Card: This is most similar to how the Nets behave pre training. A random card is selcted each round.

High Card: The highest card is played from hand each round. This is the most successful naive strategy, consistently outpreforming random card and low card. Nets trained against three high card players begin to win consistently around generation 200.

Low Card: The lowest card is played from hand each round. This may be the worst naive strategy, and is likely close to the worst strategy possible. It is defined only to have the Nets train against a variety of opponents.
