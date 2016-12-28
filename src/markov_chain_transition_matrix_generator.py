import numpy as np
import random


class MarkovChainTransitionMatrixGenerator:
    def __init__(self, number_of_classes, possible_transitions_count=2):
        """
        number_of_classes -> self explainatory
        number_of_classes = len(np.unique(y))
        """
        self.possible_transitions_count = possible_transitions_count
        self.matrix = np.zeros(shape=(number_of_classes, number_of_classes))
        np.fill_diagonal(self.matrix, 1)
        self.generate_transitions()

    def generate_transitions(self):
        number_of_classes = self.matrix.shape[0]
        if self.possible_transitions_count > number_of_classes:
            raise Exception('Number of transitions cannot exceed number of classes!')
        counter = self.possible_transitions_count - 1  # first transition is not changing state
        for class_ in range(number_of_classes):
            tabu = [class_]
            for i in range(counter):
                available_classes = list(set(range(0, number_of_classes)) - set(tabu))
                selected_class = random.choice(available_classes)
                probability_of_transition = np.random.uniform(0.,
                                                              min(self.matrix[class_, :][self.matrix[class_, :] > 0.]))
                self.matrix[class_, :][self.matrix[class_, :] > 0.] -= (probability_of_transition / (i + 1))
                self.matrix[class_, :][selected_class] = probability_of_transition
                tabu.append(selected_class)

    def get_transition_probability_matrix(self):
        return np.copy(self.matrix)
