# Contains the Gameplay of take five
# Constants: Deck
# Objects: Table
# Object: Hand
# Object: Played Cards
# Object: Claimed Cards
# Function: Add Cards
# Function: Deal
# Function: Count Score

import numpy as np
import random as rn
import time

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


class Game:
    def __init__(self, numPlayers, deckList):
        self.deck = deckList
        self.mainTable = Table(self)
        self.players = [Hand(self) for i in range(numPlayers)]

    def deal(self):
        # Shuffles the indices of the deck
        order = deck[:, 0]
        rn.shuffle(order)
        # Deals four cards to the main table to start
        self.mainTable.tableSpace = np.zeros((4, 5))
        self.mainTable.tableSpace[:, 0] = [order[i] for i in range(4)]
        # Deals ten cards to each of the players
        for i in range(len(self.players)):
            self.players[i].handSpace[:] = [order[j] for j in range(4+10*i, 4+10*(i+1))]
            self.players[i].handSpace.sort()

    def clear_scores(self):
        for player in self.players:
            player.score = 0

    def play_round(self, printing):
        scores = np.zeros(len(self.players))
        for i in range(10):
            self.play_hand()
            if printing:
                self.print_gamestate()
        for i in range(len(self.players)):
            scores[i] = self.players[i].score
        if scores.max() > 65:
            return scores
        else:
            return 0

    def play_hand(self):
        playedCards = np.zeros((len(self.players), 2))
        for i in range(len(playedCards)):
            playedCards[i, :] = [self.players[i].choose_card(), i]
        playedCards = playedCards[playedCards[:, 0].argsort()]
        for i in range(len(playedCards[:, 0])):
            self.players[int(playedCards[i, 1])].score += self.mainTable.add_card(playedCards[i, 0])

    def print_gamestate(self):
        for hand in self.players:
            hand.print_hand()
        self.mainTable.print_table()


class Table:
    def __init__(self, parent):
        self.tableSpace = np.zeros((4, 5))
        self.deck = parent.deck

    def print_table(self):
        print(self.tableSpace)

    def clear_row(self, row):
        score = self.row_total(row)
        self.tableSpace[row, :] = [0, 0, 0, 0, 0]
        return score

    def row_total(self, row):
        total = 0
        for i in range(5):
            if self.tableSpace[row, i] != 0:
                total = total + self.deck[int(self.tableSpace[row, i] - 1), 1]
        return total

    def add_card(self, addedCard):
        closest = -999
        destination = -1
        points = 0
        # Checks which row has the highest card lower than the added card, as sets said row as the destination
        for i in range(4):
            if np.max(self.tableSpace[i, :]) < addedCard and (np.max(self.tableSpace[i, :]) - addedCard) > closest:
                closest = np.max(self.tableSpace[i, :]) - addedCard
                destination = i
        # If the card can be legally placed
        if destination != -1:
            # Tries to place the card at the end of the row
            try:
                self.tableSpace[destination, np.where(self.tableSpace[destination, :] == 0)[0][0]] = addedCard
            # If there is no room left, the whole row of cards is picked up, and the new card is placed
            except:
                points = self.clear_row(destination)
                self.tableSpace[destination, 0] = addedCard
        # If the card can not be legally placed, tried to find the row with the lowest point total
        else:
            best = [999, -1]
            # Searches through rows
            for i in range(4):
                total = 0
                # Totals the points of cards in the row
                for j in range(5):
                    if self.tableSpace[i, j] != 0:
                        total = total + self.deck[int(self.tableSpace[i, j] - 1), 1]
                # Is this row the lowest scoring row? If so, remember that
                if total < best[0]:
                    best = [total, i]
            # Take the lowest scoring row and add the card to it.
            points = self.clear_row(best[1])
            self.tableSpace[best[1], 0] = addedCard
        return points


class Hand:
    def __init__(self, parent):
        self.handSpace = np.zeros(10)
        self.score = 0

    def choose_card(self):
        chosenCard = self.handSpace[9]
        self.handSpace[9] = 0
        self.handSpace.sort()
        return chosenCard

    def print_hand(self):
        print(self.handSpace, " Score: ", self.score)


newGame = Game(4, deck)
start = time.time()
hands = 0
for i in range(1000):
    scores = 0
    newGame.clear_scores()
    while type(scores) is int:
        newGame.deal()
        scores = newGame.play_round(False)
        hands += 1
end = time.time()
print(end - start)
print(hands)



# newTable = Table()
# newTable.print_table()
# newTable.add_card(11)
# newTable.add_card(12)
# newTable.add_card(13)
# newTable.add_card(14)
# newTable.print_table()
# newTable.add_card(15)
# newTable.print_table()
