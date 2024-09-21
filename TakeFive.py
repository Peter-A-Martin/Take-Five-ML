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
import MLModel as mlm
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


class Game:
    def __init__(self, deckList):
        self.deck = deckList
        self.mainTable = Table(self)
        self.players = []
        self.gamestate = np.zeros((1, 13))

    def add_player(self, player):
        # Adds a player to the roster
        self.players.append(player)
        self.players[-1].parent = self
        return self.players

    def print_gamestate(self):
        # Prints the state of the game in an easy-to-read way
        for hand in self.players:
            hand.print_hand()
        self.mainTable.print_table()

    def deal(self):
        # Shuffles the indices of the deck
        order = deck[:, 0]
        rn.shuffle(order)
        # Deals four cards to the main table to start
        self.mainTable.tableSpace = np.zeros((4, 5))
        for i in range(4):
            self.mainTable.tableSpace[i, 0] = order[i]
            # self.gamestate[0, (int(order[i]) - 1 + i * 104)] = 1
        # Deals ten cards to each of the players
        for i in range(len(self.players)):
            self.players[i].handSpace[:] = [order[j] for j in range(4+10*i, 4+10*(i+1))]
        self.update_gamestate(self.mainTable.tableSpace, np.zeros(len(self.players)))

    def clear_scores(self):
        # Clears the scores of all players
        for player in self.players:
            player.score = 0

    # Plays a full round (hand?) of Take Five
    def play_round(self, **print_state):
        # Creates an array to hold scores
        scores = np.zeros(len(self.players))
        # For each of the ten cards in hand
        for i in range(10):
            # Runs the play hand method
            self.play_hand()
            if print_state:
                self.print_gamestate()
        # Updates Scores
        for i in range(len(self.players)):
            scores[i] = self.players[i].score
        # If the game has ended, return the array of all scores
        if scores.max() > 65:
            return scores
        else:
            return 0

    def play_hand(self):
        scores = np.zeros(len(self.players))
        # Get an array of played cards using the choose card method associated with the players
        playedCards = [player.choose_card(self.gamestate, player.handSpace) for player in self.players]
        # Copies the table to add cards
        tempTable = self.mainTable.tableSpace
        # Plays the cards in ascending numeric order, assigning score to the correct players
        for i in range(len(playedCards)):
            index = playedCards.index(min(playedCards))
            score = self.mainTable.add_card(min(playedCards), tempTable)
            scores[playedCards.index(min(playedCards))] += score
            playedCards[index] = 999
            # If any NNs are training, they will perform a backpropagation using the real point value gained by the Net
            if type(self.players[index].actor) is mlm.NeuralNetwork:
                if self.players[index].actor.training:
                    self.players[index].actor.backpropagation(self.gamestate, scores[index])
        # Commits the changes to the gamestate
        self.update_gamestate(tempTable, scores)

    def update_gamestate(self, table, scores):
        self.gamestate = np.zeros((1, 13))
        self.mainTable.tableSpace = table
        for i in range(len(self.mainTable.tableSpace[:, 0])):
            for j in range(len(self.mainTable.tableSpace[0, :])):
                if self.mainTable.tableSpace[i, j] != 0:
                    self.gamestate[0, i] += 1
                    self.gamestate[0, i + 4] = max(self.mainTable.tableSpace[i, j], self.gamestate[0, i + 4])
                    self.gamestate[0, i + 8] += self.deck[int(self.mainTable.tableSpace[i, j] - 1), 1]
        # self.gamestate = np.zeros((1, 417))
        # for i in range(len(self.mainTable.tableSpace[:, 0])):
        #     for j in range(len(self.mainTable.tableSpace[0, :])):
        #         if self.mainTable.tableSpace[i, j] != 0:
        #             self.gamestate[0, (int(self.mainTable.tableSpace[i, j] - 1) + i * 104)] = 1
        for i in range(len(scores)):
            self.players[i].score += scores[i]



class Table:
    def __init__(self, parent):
        self.tableSpace = np.zeros((4, 5))
        self.deck = parent.deck
        self.parent = parent

    def print_table(self):
        print(self.tableSpace)

    def clear_row(self, row, table):
        score = self.row_total(row, table)
        for i in range(5):
            if table[row, i] != 0:
                # self.parent.gamestate[int(self.tableSpace[row, i] - 1), row] = 0
                table[row, i] = 0
        return score

    def row_total(self, row, table):
        total = 0
        for i in range(5):
            if table[row, i] != 0:
                total = total + self.deck[int(table[row, i] - 1), 1]
        return total

    def add_card(self, addedCard, table):
        closest = -999
        destination = -1
        points = 0
        # Checks which row has the highest card lower than the added card, as sets said row as the destination
        for i in range(4):
            if np.max(table[i, :]) < addedCard and (np.max(table[i, :]) - addedCard) > closest:
                closest = np.max(table[i, :]) - addedCard
                destination = i
        # If the card can be legally placed
        if destination != -1:
            # Tries to place the card at the end of the row
            try:
                table[destination, np.where(table[destination, :] == 0)[0][0]] = addedCard
                # table[int(addedCard) - 1, destination] = 1
            # If there is no room left, the whole row of cards is picked up, and the new card is placed
            except:
                points = self.clear_row(destination, table)
                table[destination, 0] = addedCard
                # self.parent.gamestate[int(addedCard) - 1, destination] = 1
        # If the card can not be legally placed, tried to find the row with the lowest point total
        else:
            best = [999, -1]
            # Searches through rows
            for i in range(4):
                total = 0
                # Totals the points of cards in the row
                for j in range(5):
                    if table[i, j] != 0:
                        total = total + self.deck[int(table[i, j] - 1), 1]
                # Is this row the lowest scoring row? If so, remember that
                if total < best[0]:
                    best = [total, i]
            # Take the lowest scoring row and add the card to it.
            points = self.clear_row(best[1], table)
            table[best[1], 0] = addedCard
            # self.parent.gamestate[int(addedCard) - 1, best[1]] = 1
        return points


class Hand:
    def __init__(self):
        self.handSpace = np.zeros(10)
        self.score = 0
        self.actor = -1
        self.parent = -1

    # Chooses the card to play
    # This method is used both when using a NN to choose a card, or a dummy player
    # Takes as input the decision space (as passed to the NN)
    # And the cards in hand
    def choose_card(self, decisionSpace, hand):
        # If a NN is attached to the player
        if type(self.actor) is mlm.NeuralNetwork:
            # Sets best score as inf
            # The player will choose to play the card with the lowest expected score (as determined by the NN)
            bestScore = math.inf
            for card in hand:
                # Iterated through nonzero cards in hand
                if card != 0.0:
                    # Adds card to decision space, passes to NN to estimate score
                    decisionSpace[0, -1] = card
                    score = self.actor.get_output(decisionSpace)
                    # Remembers the best score
                    if score[0][0] < bestScore:
                        bestScore = score[0][0]
                        chosenCard = card
            # Removes most the card chosen for play from the hand
            self.handSpace[np.where(self.handSpace == chosenCard)] = 0
        # If no NN is chosen, plays the first nonzero card in hand
        else:
            chosenCard = self.handSpace[np.nonzero(self.handSpace)[0][0]]
            self.handSpace[np.nonzero(self.handSpace)[0][0]] = 0
        # Returns the chosen card
        return chosenCard

    def print_hand(self):
        print(self.handSpace, " Score: ", self.score)
