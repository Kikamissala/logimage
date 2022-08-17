from logimage.cell import Cell, CellState

class InvalidRule(Exception):
    pass

class RuleElement(int):

    def translate_to_list(self):
        return self * [Cell(CellState.full)]

class Rule(list):

    def __init__(self,values):
        super().__init__([RuleElement(value) for value in values])

    def compute_min_possible_len(self):
        sum_of_cells_to_fill = sum(self)
        minimum_number_of_blanks = len(self) - 1
        min_possible_len = sum_of_cells_to_fill + minimum_number_of_blanks
        return min_possible_len

    def compute_min_starting_indexes(self):
        if len(self) == 1:
            return [0]
        else:
            start_index_of_index_2 = self[0] + 1
            return [0] + list(map(lambda x: x + start_index_of_index_2, Rule(self[1:]).compute_min_starting_indexes()))

class RuleList(list):
    
    def __init__(self,values):
        super().__init__([Rule(value) for value in values])

    def compute_maximum_minimum_possible_len(self):
        maximum_minimum_possible_len = max([rule.compute_min_possible_len() for rule in self])
        return maximum_minimum_possible_len

class RuleSet:

    def __init__(self, row_rules, column_rules):
        self.row_rules = RuleList(row_rules)
        self.column_rules = RuleList(column_rules)