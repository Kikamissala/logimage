from scipy import *
import numpy as np
import pandas as pd

class CellState:

    def __init__(self, value = None):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, CellState):
            return self.value == other.value
        else:
            return False

class UndefinedCellState(CellState):
    def __init__(self, value = "undefined"):
        super().__init__(value)

class FullCellState(CellState):
    def __init__(self, value = "full"):
        super().__init__(value)

class EmptyCellState(CellState):
    def __init__(self, value = "empty"):
        super().__init__(value)

class InvalidCellStateModification(Exception):
    pass

class Cell:
    
    def __init__(self, cell_state = UndefinedCellState()):
        self.cell_state = cell_state
    
    def empty(self):
        if self.cell_state != UndefinedCellState():
            raise InvalidCellStateModification("impossible to modify not undefined cell state")
        self.cell_state = EmptyCellState()
    
    def full(self):
        if self.cell_state != UndefinedCellState():
            raise InvalidCellStateModification("impossible to modify not undefined cell state")
        self.cell_state = FullCellState()
    
    def numerize(self):
        if self.cell_state == UndefinedCellState():
            return None
        if self.cell_state == EmptyCellState():
            return 0
        if self.cell_state == FullCellState():
            return 1
    
    def __eq__(self, other):
        if isinstance(other, Cell):
            return self.cell_state == other.cell_state
        else:
            return False

class InvalidGridSet(Exception):
    pass

class Grid:
    def __init__(self, row_number, column_number):
        self.cells = np.array([[Cell() for j in range(0,column_number)] for i in range(0,row_number)])
        self.row_number = row_number
        self.column_number = column_number
    
    def __getitem__(self,pos):
        row_index, column_index = pos
        return self.cells[row_index][column_index]
    
    def __setitem__(self,pos, value):
        row_index, column_index = pos
        if isinstance(value,np.ndarray):
            for item in value:
                self.raise_if_not_cell(item)
        else:    
            self.raise_if_not_cell(value)
        self.cells[row_index,column_index] = value

    def empty(self, row_number,column_number):
        self.cells[row_number,column_number].empty()

    def full(self, row_number,column_number):
        self.cells[row_number,column_number].full()

    @staticmethod
    def raise_if_not_cell(item):
        if not isinstance(item, Cell):
            raise InvalidGridSet(f"{item} is not a cell")

class InvalidRule(Exception):
    pass

class RuleElement(int):

    def translate_to_list(self):
        return self * [1]

class Rule(list):

    def __init__(self,values):
        super().__init__([RuleElement(value) for value in values])

    def compute_min_possible_len(self):
        sum_of_cells_to_fill = sum(self)
        minimum_number_of_blanks = len(self) - 1
        min_possible_len = sum_of_cells_to_fill + minimum_number_of_blanks
        return min_possible_len

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

class Logimage:
    
    def __init__(self, grid_dimensions, rules):
        self.grid = Grid(grid_dimensions[0], grid_dimensions[1])
        self.rules = rules
        self.raise_if_rules_invalid()

    def is_rule_exceeding_grid_size(self):
        maximum_minimum_possible_len_row = self.rules.column_rules.compute_maximum_minimum_possible_len()
        maximum_minimum_possible_len_column = self.rules.row_rules.compute_maximum_minimum_possible_len()
        if maximum_minimum_possible_len_row > self.grid.row_number:
            return True
        elif maximum_minimum_possible_len_column > self.grid.column_number:
            return True
        else:
            return False
    
    def is_rule_number_exceeding_grid_size(self):
        if len(self.rules.row_rules) > self.grid.row_number:
            return True
        elif len(self.rules.column_rules) > self.grid.column_number:
            return True
        else:
            return False

    def raise_if_rules_invalid(self):
        if self.is_rule_exceeding_grid_size():
            raise InvalidRule("A rule is exceeding grid size")
        if self.is_rule_number_exceeding_grid_size():
            raise InvalidRule("Number of rules exceeding grid size")

class Problem:

    def __init__(self, rule, cells):
        self.rule = rule
        self.cells = cells
        self.numerized_list = self.numerize_cell_list()
        self.length = len(self.cells)
    
    def numerize_cell_list(self):
        numerized_list = []
        for cell in self.cells:
            numerized_list.append(cell.numerize())
        return numerized_list
    
    def assess_cell_rule_correspondance(self):
        initial_list = [None] * self.length
    
    @staticmethod
    def find_first_index_of_value(input_series, value):
        if len(input_series) == 0:
            return None
        else:
            if input_series.iloc[0] == value:
                return input_series.index[0]
            else:
                return Problem.find_first_index_of_value(input_series[1:],value)

    @staticmethod
    def find_first_index_of_non_value(input_series, value):
        if len(input_series) == 0:
            return None
        else:
            if input_series.iloc[0] != value:
                return input_series.index[0]
            else:
                return Problem.find_first_index_of_non_value(input_series[1:],value)

    @staticmethod
    def find_series_without_value_zero(input_numerized_series):
        if len(input_numerized_series) == 0:
            return []
        first_zero_index = Problem.find_first_index_of_value(input_numerized_series,0)
        if Problem.find_first_index_of_value(input_numerized_series,0) is None:
            print("pas de zero")
            return [input_numerized_series]
        else:
            if input_numerized_series.iloc[0] == 0:
                print("zéro au début")
                first_non_zero_index = Problem.find_first_index_of_non_value(input_numerized_series,0)
                return Problem.find_series_without_value_zero(input_numerized_series[input_numerized_series.index >= first_non_zero_index])
            else:
                print("zero pas au début")
                return [input_numerized_series[input_numerized_series.index < first_zero_index]] + Problem.find_series_without_value_zero(input_numerized_series[input_numerized_series.index >= first_zero_index])
                

    def identify_full_rules(self):
        list_of_identified_rules = []
        numerized_list_series = pd.Series(self.numerized_list)
        list_of_non_zero_series = Problem.find_series_without_value_zero(numerized_list_series)
        for non_zero_serie in list_of_non_zero_series:
            if Problem.find_first_index_of_value(non_zero_serie,None) is None:
                list_of_identified_rules.append({"len":len(non_zero_serie),"initial_index":non_zero_serie.index[0]})
        return list_of_identified_rules
    
    def compute_number_of_freedom_degrees(self):
        minimum_len_from_rule = self.rule.compute_min_possible_len()
        number_of_freedom_degrees = self.length - minimum_len_from_rule
        return number_of_freedom_degrees

    def is_solved(self):
        list_of_bool_is_undefined = [numerized_state == None for numerized_state in self.numerized_list]
        if any(list_of_bool_is_undefined):
            return False
        return True
    
    def is_line_fully_defined_by_rule(self):
        minimum_possible_len = self.rule.compute_min_possible_len()
        if minimum_possible_len == len(self.cells):
            return True
        return False
    
    def fully_defined_solve(self):
        output_numerized_list = []
        index = 0
        while index < len(self.rule) - 1:
            rule_element = self.rule[index]
            output_numerized_list += rule_element.translate_to_list() + [0]
            index += 1
        last_rule_element = self.rule[index]
        output_numerized_list += last_rule_element.translate_to_list()
        self.numerized_list = output_numerized_list