from scipy import *
import numpy as np
import pandas as pd
from enum import Enum

class CellState(Enum):
    empty = 0
    undefined = -1
    full = 1

class InvalidCellStateModification(Exception):
    pass

class Cell:
    
    def __init__(self, cell_state = CellState.undefined,rule_element_index = None):
        self.cell_state = cell_state
        if self.cell_state == CellState.full:
            self.rule_element_index = rule_element_index
        else:
            self.rule_element_index = None

    def empty(self):
        if self.cell_state != CellState.undefined:
            raise InvalidCellStateModification("impossible to modify not undefined cell state")
        self.cell_state = CellState.empty
    
    def full(self):
        if self.cell_state != CellState.undefined:
            raise InvalidCellStateModification("impossible to modify not undefined cell state")
        self.cell_state = CellState.full
    
    def numerize(self):
        return self.cell_state.value
    
    def __eq__(self, other):
        if isinstance(other, Cell):
            return (self.cell_state == other.cell_state) & (self.rule_element_index == other.rule_element_index)
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

class Solution:
    pass

class FullBlock:
    
    def __init__(self, block_len, initial_index):
        self.block_len = block_len
        self.initial_index = initial_index
        self.last_index = self.initial_index + self.block_len
    
    def __eq__(self, other):
        if isinstance(other, FullBlock):
            return (self.block_len == other.block_len) & (self.initial_index == other.initial_index) & (self.last_index == other.last_index)

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
            return [input_numerized_series]
        else:
            if input_numerized_series.iloc[0] == 0:
                first_non_zero_index = Problem.find_first_index_of_non_value(input_numerized_series,0)
                return Problem.find_series_without_value_zero(input_numerized_series[input_numerized_series.index >= first_non_zero_index])
            else:
                result_serie = input_numerized_series[input_numerized_series.index < first_zero_index]
                return [result_serie] + Problem.find_series_without_value_zero(input_numerized_series[input_numerized_series.index >= first_zero_index])

    def first_undefined_cell_index(self):
        numerized_series = pd.Series(self.numerized_list)
        first_undefined_index = Problem.find_first_index_of_value(numerized_series, -1)
        return first_undefined_index
    
    def last_undefined_cell_index(self):
        reversed_numerized_series = pd.Series(self.numerized_list[::-1])
        first_reversed_undefined_index = Problem.find_first_index_of_value(reversed_numerized_series, -1)
        max_index = len(self.numerized_list) -1
        if first_reversed_undefined_index is None:
            return None
        else:
            last_undefined_index = max_index - first_reversed_undefined_index
            return last_undefined_index
             
    def identify_full_blocks(self):
        list_of_identified_full_blocks = []
        numerized_list_series = pd.Series(self.numerized_list)
        list_of_non_zero_series = Problem.find_series_without_value_zero(numerized_list_series)
        for non_zero_serie in list_of_non_zero_series:
            if Problem.find_first_index_of_value(non_zero_serie,-1) is None:
                #list_of_identified_full_blocks.append({"len":len(non_zero_serie),"initial_index":non_zero_serie.index[0]})
                list_of_identified_full_blocks.append(FullBlock(block_len=len(non_zero_serie),initial_index=non_zero_serie.index[0]))
        return list_of_identified_full_blocks

    def get_rule_element_indexes(self):
        rule_element_indexes = [cell.rule_element_index for cell in self.cells]
        return rule_element_indexes

    @staticmethod
    def assign_value_to_index_range_in_list(input_list,first_index,last_index,value):
        output_list = input_list.copy()
        output_list[first_index:last_index] = [value] * (last_index - first_index)
        return output_list

    @staticmethod
    def get_full_block_infos(full_block):
        block_len = full_block["len"]
        block_initial_index = full_block["initial_index"]
        block_last_index = block_initial_index + block_len
        return block_len, block_initial_index, block_last_index

    def assign_rule_elements_index_when_end_of_cells_list_is_solved(self, rule_element_indexes, list_of_identified_full_blocks, block_index, full_block):
        max_possible_rule_element_index = len(self.rule) - 1
        indexes_left_until_last_block = len(list_of_identified_full_blocks) - 1 - block_index
        deducted_rule_element_index = max_possible_rule_element_index - indexes_left_until_last_block
        rule_element_indexes_output = Problem.assign_value_to_index_range_in_list(rule_element_indexes,\
            full_block.initial_index, full_block.last_index, deducted_rule_element_index)
        return rule_element_indexes_output

    def identify_rule_element_indexes(self):
        rule_element_indexes = self.get_rule_element_indexes()
        list_of_identified_full_blocks = self.identify_full_blocks()
        first_undefined_cell_index = self.first_undefined_cell_index()
        last_undefined_cell_index = self.last_undefined_cell_index()
        for block_index, full_block in enumerate(list_of_identified_full_blocks):
            if first_undefined_cell_index is not None:
                if full_block.last_index < first_undefined_cell_index:
                    rule_element_indexes = Problem.assign_value_to_index_range_in_list(rule_element_indexes,\
                        full_block.initial_index, full_block.last_index, block_index)
                    continue
            if last_undefined_cell_index is not None:
                if full_block.initial_index > last_undefined_cell_index:
                    rule_element_indexes = self.assign_rule_elements_index_when_end_of_cells_list_is_solved(rule_element_indexes, list_of_identified_full_blocks, block_index, full_block)
                    continue
            fitting_len_rule_elements = [rule_element for rule_element in self.rule if rule_element == full_block.block_len]
            if len(fitting_len_rule_elements) == 1:
                fitting_rule_element_index = self.rule.index(full_block.block_len)
                rule_element_indexes = Problem.assign_value_to_index_range_in_list(rule_element_indexes,\
                    full_block.initial_index, full_block.last_index, fitting_rule_element_index)
        return rule_element_indexes
    
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