from scipy import *
import numpy as np
import pandas as pd
from enum import Enum
import copy
from collections import Counter

class CellState(Enum):
    empty = 0
    undefined = -1
    full = 1

class InvalidCellStateModification(Exception):
    pass

class CellUpdateError(Exception):
    pass

class Cell:
    
    def __init__(self, cell_state = CellState.undefined,rule_element_index = None):
        self.cell_state = cell_state
        if self.cell_state == CellState.full:
            self.rule_element_index = rule_element_index
        else:
            self.rule_element_index = None

    def empty(self,new_cell_state = CellState.empty):
        if self.cell_state == CellState.empty:
            return
        elif self.cell_state != CellState.undefined:
            raise InvalidCellStateModification("impossible to modify not undefined cell state")
        self.cell_state = new_cell_state
    
    def full(self,rule_element_index = None):
        if self.cell_state != CellState.undefined:
            raise InvalidCellStateModification("impossible to modify not undefined cell state")
        self.cell_state = CellState.full
        if rule_element_index is not None:
            self.set_rule_element_index(rule_element_index)
    
    def update_state(self,new_cell_state):
        if isinstance(new_cell_state,CellState) is False:
            raise InvalidCellStateModification("new value is not a cell state")
        if self.cell_state != CellState.undefined:
            raise InvalidCellStateModification("impossible to modify not undefined cell state")
        if new_cell_state == CellState.empty:
            self.empty()
        elif new_cell_state == CellState.full:
            self.full()

    def set_rule_element_index(self,rule_element_index):
        if self.cell_state != CellState.full:
            raise CellUpdateError("unable to set rule element index for non full cell")
        else:
            self.rule_element_index = rule_element_index
    
    def numerize(self):
        return self.cell_state.value
    
    def __eq__(self, other):
        if isinstance(other, Cell):
            return (self.cell_state == other.cell_state) & (self.rule_element_index == other.rule_element_index)
        else:
            return False
    
    def __repr__(self):
        base_statement = f"Cell : state = {self.cell_state}"
        if (self.cell_state == CellState.full) & (self.rule_element_index is not None):
            return base_statement + f", rule_index = {self.rule_element_index}"
        else:
            return base_statement + f", rule_index undefined"

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

    def __repr__(self):
        return f"FullBlock:len={self.block_len},initial_index={self.initial_index}"

class InvalidProblem(Exception):
    pass

class ProblemAddError(Exception):
    pass

class Problem:

    def __init__(self, rule, cells):
        self.rule = rule
        self.cells = cells
        self.length = len(self.cells)
        self.rule_elements_indexes = self.get_rule_element_indexes()
    
    def __repr__(self):
        numerized_list = self.numerize_cell_list()
        rule_element_indexes = [cell.rule_element_index for cell in self.cells]
        return f"Problem : rule : {self.rule}, cells : {numerized_list}, clues_indexes : {rule_element_indexes}"

    @staticmethod
    def increment_rule_element_indexes_in_cells_list(cells_list, increment_value):
        incremented_cells_list = [Cell(CellState.full,rule_element_index=cell.rule_element_index + increment_value) if (cell.cell_state == CellState.full) & (cell.rule_element_index is not None) else cell for cell in cells_list]
        return incremented_cells_list

    def __add__(self,other):
        if (self.cells[-1] != Cell(CellState.empty)) & (other.cells[0] != Cell(CellState.empty)):
            raise ProblemAddError("Impossible to combine two problem without at least one empty cell at junction")
        combined_rule = Rule(self.rule + other.rule)
        rule_element_index_increment = len(self.rule)
        added_cells = Problem.increment_rule_element_indexes_in_cells_list(other.cells,rule_element_index_increment)
        combined_cells_list = self.cells + added_cells
        return Problem(rule = combined_rule, cells = combined_cells_list)

    def __eq__(self,other):
        if isinstance(other,Problem):
            return (self.rule == other.rule) & (self.cells == other.cells)

    def is_splittable(self):
        if (self.cells[0].cell_state == CellState.empty) or (self.cells[-1].cell_state == CellState.empty):
            return True
        first_complete_full_block_with_rule_element_index = self.get_first_complete_full_block_with_rule_element_index()
        if first_complete_full_block_with_rule_element_index is None:
            return False
        else:
            return True

    def get_first_complete_full_block_with_rule_element_index(self):
        complete_full_blocks = self.identify_complete_full_blocks()
        sorted_complete_full_blocks = sorted(complete_full_blocks, key=lambda d: d.initial_index)
        for full_block in sorted_complete_full_blocks:
            rule_element_index_none_list = [cell.rule_element_index is None for cell in self.cells[full_block.initial_index:full_block.last_index - 1]]
            if any(rule_element_index_none_list):
                continue
            else:
                return full_block

    def get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(self,cells_split_index,rule_split_index):
        first_part_problem = Problem(rule=Rule(self.rule[0:rule_split_index]),cells = self.cells[0:cells_split_index])
        second_part_cells = Problem.increment_rule_element_indexes_in_cells_list(self.cells[cells_split_index:],-rule_split_index)
        second_part_problem = Problem(rule=Rule(self.rule[rule_split_index:]),cells = second_part_cells)
        return first_part_problem, second_part_problem

    def split(self):
        if self.cells[0].cell_state == CellState.empty:
            numerized_series_list = pd.Series(self.numerize_cell_list())
            first_non_empty_index = Problem.find_first_index_of_non_value(numerized_series_list,0)
            first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(first_non_empty_index,0)
        elif self.cells[-1].cell_state == CellState.empty:
            reversed_numerized_series_list = pd.Series(self.numerize_cell_list()[::-1])
            reversed_first_non_empty_index = Problem.find_first_index_of_non_value(reversed_numerized_series_list,0)
            first_ending_empty_index = self.length - reversed_first_non_empty_index
            first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(first_ending_empty_index,len(self.rule)) 
        else:
            first_complete_full_block_with_rule_element_index = self.get_first_complete_full_block_with_rule_element_index()
            if first_complete_full_block_with_rule_element_index.initial_index == 0:
                first_index_after_empty_index = first_complete_full_block_with_rule_element_index.block_len + 1
                first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(first_index_after_empty_index,1)
            else:
                empty_cell_before_complete_block_index = first_complete_full_block_with_rule_element_index.initial_index - 1
                rule_index = self.cells[first_complete_full_block_with_rule_element_index.initial_index].rule_element_index
                first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(empty_cell_before_complete_block_index,rule_index)
        return [first_part_problem,second_part_problem]
            

    def numerize_cell_list(self, cells_list = None):
        if cells_list == None:
            cells_list = self.cells
        numerized_list = []
        for cell in cells_list:
            numerized_list.append(cell.numerize())
        return numerized_list
    
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
    def find_series_without_value(input_numerized_series,value):
        if len(input_numerized_series) == 0:
            return []
        first_value_index = Problem.find_first_index_of_value(input_numerized_series,value)
        if first_value_index is None:
            return [input_numerized_series]
        else:
            if input_numerized_series.iloc[0] == value:
                first_non_value_index = Problem.find_first_index_of_non_value(input_numerized_series,value)
                return Problem.find_series_without_value(input_numerized_series[input_numerized_series.index >= first_non_value_index], value)
            else:
                result_serie = input_numerized_series[input_numerized_series.index < first_value_index]
                return [result_serie] + Problem.find_series_without_value(input_numerized_series[input_numerized_series.index >= first_value_index],value)

    @staticmethod
    def find_series_with_unique_value(input_numerized_series,value):
        if len(input_numerized_series) == 0:
            return []
        first_value_index = Problem.find_first_index_of_value(input_numerized_series,value)
        if first_value_index is None:
            return []
        else:
            first_non_value_index = Problem.find_first_index_of_non_value(input_numerized_series[input_numerized_series.index >= first_value_index],value)
            if first_non_value_index is None:
                result_serie = input_numerized_series[input_numerized_series.index >= first_value_index]
                return [result_serie]
            result_serie = input_numerized_series[(input_numerized_series.index >= first_value_index) & (input_numerized_series.index < first_non_value_index)]
            return [result_serie] + Problem.find_series_with_unique_value(input_numerized_series[input_numerized_series.index >= first_non_value_index],value)

    def first_undefined_cell_index(self):
        numerized_series = pd.Series(self.numerize_cell_list())
        first_undefined_index = Problem.find_first_index_of_value(numerized_series, -1)
        return first_undefined_index
    
    def last_undefined_cell_index(self):
        reversed_numerized_series = pd.Series(self.numerize_cell_list()[::-1])
        first_reversed_undefined_index = Problem.find_first_index_of_value(reversed_numerized_series, -1)
        max_index = len(self.numerize_cell_list()) -1
        if first_reversed_undefined_index is None:
            return None
        else:
            last_undefined_index = max_index - first_reversed_undefined_index
            return last_undefined_index
             
    def identify_complete_full_blocks(self):
        list_of_identified_full_blocks = []
        numerized_list_series = pd.Series(self.numerize_cell_list())
        list_of_non_zero_series = Problem.find_series_without_value(numerized_list_series,0)
        for non_zero_serie in list_of_non_zero_series:
            if Problem.find_first_index_of_value(non_zero_serie,-1) is None:
                list_of_identified_full_blocks.append(FullBlock(block_len=len(non_zero_serie),initial_index=non_zero_serie.index[0]))
        return list_of_identified_full_blocks
    
    def identify_incomplete_full_blocks(self):
        list_of_incomplete_full_blocks = []
        numerized_list_series = pd.Series(self.numerize_cell_list())
        list_of_consecutive_full_series = Problem.find_series_with_unique_value(numerized_list_series,1)
        for full_series in list_of_consecutive_full_series:
            max_full_series_index = full_series.index[len(full_series) - 1]
            first_index_after_full_series = max_full_series_index + 1
            first_index_before_full_series = full_series.index[0] - 1
            if len(full_series) == self.length:
                return []
            elif full_series.index[0] == 0:
                if numerized_list_series[first_index_after_full_series] != 0:
                    list_of_incomplete_full_blocks.append(FullBlock(block_len=len(full_series),initial_index=full_series.index[0]))
            elif max_full_series_index == len(numerized_list_series) - 1:
                if numerized_list_series[first_index_before_full_series] != 0:
                    list_of_incomplete_full_blocks.append(FullBlock(block_len=len(full_series),initial_index=full_series.index[0]))
            elif (numerized_list_series[first_index_before_full_series] != 0) & (numerized_list_series[first_index_after_full_series] != 0):
                list_of_incomplete_full_blocks.append(FullBlock(block_len=len(full_series),initial_index=full_series.index[0]))
        return list_of_incomplete_full_blocks

    def get_rule_element_indexes(self):
        rule_element_indexes = [cell.rule_element_index for cell in self.cells]
        return rule_element_indexes

    @staticmethod
    def assign_value_to_index_range_in_list(input_list,first_index,last_index,value):
        output_list = input_list.copy()
        output_list[first_index:last_index] = [value] * (last_index - first_index)
        return output_list

    def assign_rule_elements_index_when_end_of_cells_list_is_solved(self, rule_element_indexes, list_of_identified_full_blocks, block_index, full_block):
        max_possible_rule_element_index = len(self.rule) - 1
        indexes_left_until_last_block = len(list_of_identified_full_blocks) - 1 - block_index
        deducted_rule_element_index = max_possible_rule_element_index - indexes_left_until_last_block
        rule_element_indexes_output = Problem.assign_value_to_index_range_in_list(rule_element_indexes,\
            full_block.initial_index, full_block.last_index, deducted_rule_element_index)
        return rule_element_indexes_output

    def get_elements_for_identifying_rule_element_indexes(self):
        rule_element_indexes = self.get_rule_element_indexes()
        list_of_identified_full_blocks = self.identify_complete_full_blocks()
        first_undefined_cell_index = self.first_undefined_cell_index()
        last_undefined_cell_index = self.last_undefined_cell_index()
        return rule_element_indexes, list_of_identified_full_blocks, first_undefined_cell_index, last_undefined_cell_index

    def associate_complete_full_block_with_rule_element_at_extremity(self, rule_element_indexes, list_of_identified_full_blocks,\
         first_undefined_cell_index, last_undefined_cell_index, block_index, full_block):
        if full_block.last_index < first_undefined_cell_index:
            rule_element_indexes = Problem.assign_value_to_index_range_in_list(rule_element_indexes,\
                full_block.initial_index, full_block.last_index, block_index)
            return rule_element_indexes
        elif full_block.initial_index > last_undefined_cell_index:
            rule_element_indexes = self.assign_rule_elements_index_when_end_of_cells_list_is_solved(rule_element_indexes, list_of_identified_full_blocks, block_index, full_block)
            return rule_element_indexes
        return rule_element_indexes

    def identify_rule_element_indexes(self):
        rule_element_indexes, list_of_identified_full_blocks, first_undefined_cell_index, last_undefined_cell_index = self.get_elements_for_identifying_rule_element_indexes()
        number_of_rule_elements_per_value = Counter(self.rule)
        number_of_blocks_per_len = Counter([full_block.block_len for full_block in list_of_identified_full_blocks])
        for block_index, full_block in enumerate(list_of_identified_full_blocks):
            if number_of_rule_elements_per_value[full_block.block_len] == number_of_blocks_per_len[full_block.block_len]:
                rule_element_indexes_of_same_len = [index for index, rule_element in enumerate(self.rule) if rule_element == full_block.block_len]
                full_blocks_of_same_len = [full_block_iter for full_block_iter in list_of_identified_full_blocks if full_block_iter.block_len == full_block.block_len]
                full_blocks_of_same_len = sorted(full_blocks_of_same_len, key=lambda d: d.initial_index)
                index_of_this_full_block_in_list_of_full_blocks_of_same_len = next((index for (index, full_block_iter) in enumerate(full_blocks_of_same_len) if full_block_iter.initial_index == full_block.initial_index), None)
                rule_element_index_of_this_block = rule_element_indexes_of_same_len[index_of_this_full_block_in_list_of_full_blocks_of_same_len]
                rule_element_indexes = Problem.assign_value_to_index_range_in_list(rule_element_indexes,\
                full_block.initial_index, full_block.last_index, rule_element_index_of_this_block)
            else:
                rule_element_indexes = self.associate_complete_full_block_with_rule_element_at_extremity(rule_element_indexes, list_of_identified_full_blocks, first_undefined_cell_index, last_undefined_cell_index, block_index, full_block)
        return rule_element_indexes
    
    def update_rule_element_indexes(self, new_rule_element_indexes):
        for index, rule_element_index in enumerate(self.rule_elements_indexes):
            if new_rule_element_indexes[index] == None:
                continue
            if new_rule_element_indexes[index] != rule_element_index:
                if rule_element_index is not None:
                    raise InvalidProblem("unable to update rule element indexes, incompatible new rule indexes")
                self.rule_elements_indexes[index] = new_rule_element_indexes[index]
                self.cells[index].set_rule_element_index(new_rule_element_indexes[index])

    def identify_and_update_rule_element_indexes(self):
        new_rule_element_indexes = self.identify_rule_element_indexes()
        self.update_rule_element_indexes(new_rule_element_indexes)

    def compute_number_of_freedom_degrees(self):
        minimum_len_from_rule = self.rule.compute_min_possible_len()
        number_of_freedom_degrees = self.length - minimum_len_from_rule
        return number_of_freedom_degrees

    def is_solved(self):
        list_of_bool_is_undefined = [numerized_state == -1 for numerized_state in self.numerize_cell_list()]
        if any(list_of_bool_is_undefined):
            return False
        return True
    
    def is_line_fully_defined_by_rule(self):
        minimum_possible_len = self.rule.compute_min_possible_len()
        if minimum_possible_len == len(self.cells):
            return True
        return False
    
    def update_cells_list(self, new_cell_lists):
        for index, cell in enumerate(self.cells):
            if new_cell_lists[index].cell_state == CellState.undefined:
                continue
            if new_cell_lists[index].cell_state != cell.cell_state:
                if cell.cell_state != CellState.undefined:
                    raise InvalidProblem("unable to update non undefined cell")
                #self.cells[index].update_state(new_cell_lists[index].cell_state)
                self.cells[index] = Cell(new_cell_lists[index].cell_state)
                if new_cell_lists[index].rule_element_index is not None:
                    self.cells[index].set_rule_element_index(new_cell_lists[index].rule_element_index)
            elif new_cell_lists[index].rule_element_index is not None:
                self.cells[index].set_rule_element_index(new_cell_lists[index].rule_element_index)

    def fully_defined_solve(self):
        new_cells_list = []
        index = 0
        while index < len(self.rule) - 1:
            rule_element = self.rule[index]
            new_cells_list += rule_element.translate_to_list() + [Cell(CellState.empty)]
            index += 1
        last_rule_element = self.rule[index]
        new_cells_list += last_rule_element.translate_to_list()
        self.update_cells_list(new_cells_list)
    
    def is_subject_to_overlap_solving(self):
        freedom_degrees_number = self.compute_number_of_freedom_degrees()
        max_rule_element_len = max(self.rule)
        if max_rule_element_len > freedom_degrees_number:
            return True
        return False
    
    def overlapping_solve(self):
        freedom_degrees_number = self.compute_number_of_freedom_degrees()
        min_start_indexes = self.rule.compute_min_starting_indexes()
        output_cells_list = [Cell(CellState.undefined)] * self.length
        for index, rule_element in enumerate(self.rule):
            if rule_element > freedom_degrees_number:
                min_start_index = min_start_indexes[index]
                max_start_index = min_start_index + freedom_degrees_number
                min_end_index = min_start_index + rule_element
                len_overlapping = min_end_index - max_start_index
                output_cells_list[max_start_index:min_end_index] = [Cell(CellState.full,rule_element_index=index)] * len_overlapping
        self.update_cells_list(output_cells_list)
    
    def all_full_cell_found(self):
        full_cells_number = len([cell for cell in self.cells if cell.cell_state == CellState.full])
        return full_cells_number == sum(self.rule)
    
    def all_full_cell_found_solve(self):
        output_cells_list = self.cells
        for index in range(0,self.length):
            if output_cells_list[index].cell_state == CellState.undefined:
                output_cells_list[index].empty()
        self.update_cells_list(output_cells_list)

    def index_strict_upper_born_to_fill_with_empty(self,cell_list = None, rule = None, first_rule_element_index = None):
        if cell_list == None:
            cell_list = self.cells
        if rule == None:
            rule = self.rule
        if first_rule_element_index == None:
            first_rule_element_index = 0
        if cell_list[0].cell_state == CellState.full:
            return 0
        first_undefined_index = cell_list.index(Cell(CellState.undefined))
        first_full_index = next((index for index, cell in enumerate(cell_list) if cell.cell_state == CellState.full),None)
        if first_undefined_index is None:
            return 0
        if first_full_index is not None:
            if first_full_index < first_undefined_index:
                return 0
            elif cell_list[first_full_index].rule_element_index == first_rule_element_index:
                if first_full_index > 1:
                    if cell_list[first_full_index - 1].cell_state == CellState.empty:
                        return first_full_index
        cell_list_from_first_undefined = cell_list[first_undefined_index:]
        index = 1
        while (index < rule[0]) & (index < len(cell_list_from_first_undefined)):
            current_cell = cell_list_from_first_undefined[index]
            if current_cell.cell_state == CellState.full:
                return 0
            elif current_cell.cell_state == CellState.empty:
                new_start_index = first_undefined_index + index
                new_cell_list = cell_list[new_start_index:]
                return new_start_index + self.index_strict_upper_born_to_fill_with_empty(new_cell_list,rule, first_rule_element_index)
            else:
                index += 1
        return 0

    def index_strict_lower_born_to_fill_with_empty(self):
        index_strict_upper_born_of_reversed_problem = self.index_strict_upper_born_to_fill_with_empty(cell_list = self.cells[::-1],rule = self.rule[::-1],first_rule_element_index = len(self.rule) - 1)
        max_possible_index = self.length - 1
        return max_possible_index - index_strict_upper_born_of_reversed_problem

    def head_fill_empty_solve(self):
        last_index_to_fill_with_empty = self.index_strict_upper_born_to_fill_with_empty()
        output_cells_list = self.cells
        for index in range(0,last_index_to_fill_with_empty):
            if output_cells_list[index].cell_state == CellState.undefined:
                output_cells_list[index].empty()
        self.update_cells_list(output_cells_list)

    def tail_fill_empty_solve(self):
        first_index_to_fill_with_empty = self.index_strict_lower_born_to_fill_with_empty()
        output_cells_list = copy.deepcopy(self.cells)
        for index in range(first_index_to_fill_with_empty + 1,self.length):
            if output_cells_list[index].cell_state == CellState.undefined:
                output_cells_list[index].empty()
        self.update_cells_list(output_cells_list)
    
    def extremities_fill_empty_solve(self):
        self.head_fill_empty_solve()
        self.tail_fill_empty_solve()

    def complete_full_blocks_with_max_rule_size_solve(self):
        incomplete_full_blocks = self.identify_incomplete_full_blocks()
        output_cells_list = copy.deepcopy(self.cells)
        max_rule_element_value = max(self.rule)
        completable_full_blocks = [full_block for full_block in incomplete_full_blocks if full_block.block_len == max_rule_element_value]
        for full_block in completable_full_blocks:
            first_index_after_block = full_block.initial_index + full_block.block_len
            if full_block.initial_index == 0:
                output_cells_list[first_index_after_block].empty()
            elif first_index_after_block == self.length:
                output_cells_list[full_block.initial_index - 1].empty()
            else:
                output_cells_list[first_index_after_block].empty()
                output_cells_list[full_block.initial_index - 1].empty()
        self.update_cells_list(output_cells_list)
    
    @staticmethod
    def all_undefined_to_full_in_range(cells_list,range_min, range_max, rule_element_index = None):
        for index in range(range_min,range_max):
            if cells_list[index].cell_state == CellState.undefined:
                cells_list[index].full()
            if (cells_list[index].cell_state == CellState.full) & (rule_element_index is not None):
                cells_list[index].set_rule_element_index(rule_element_index)
        return cells_list

    def fill_head_of_cell_list_based_on_first_rule(self, cells_list,rule, full_block, rule_element_index = None):
        if rule_element_index is None:
            rule_element_index = 0
        output_cells_list = Problem.all_undefined_to_full_in_range(cells_list,full_block.initial_index, rule[0],rule_element_index)
        if rule[0] < self.length:
            if full_block.initial_index == 0:
                index_after_full_block = full_block.block_len
                if output_cells_list[index_after_full_block].cell_state == CellState.undefined:
                    output_cells_list[index_after_full_block].empty()
        return output_cells_list
    
    def fill_tail_of_cell_list_based_on_last_rule(self, cells_list,rule, full_block):
        reversed_cells_list = cells_list[::-1]
        reversed_rule = rule[::-1]
        max_full_index = full_block.initial_index + full_block.block_len - 1
        reversed_first_full_index = self.length - max_full_index - 1
        reversed_full_block = FullBlock(full_block.block_len,initial_index=reversed_first_full_index)
        reversed_rule_element_index = len(self.rule) - 1
        reversed_solved_list = self.fill_head_of_cell_list_based_on_first_rule(reversed_cells_list, reversed_rule, reversed_full_block, reversed_rule_element_index)
        return reversed_solved_list[::-1]

    def complete_extremities_full_block_solve(self):
        output_cells_list = copy.deepcopy(self.cells)
        incomplete_full_blocks = self.identify_incomplete_full_blocks()
        if len(incomplete_full_blocks) == 0:
            return
        for full_block in incomplete_full_blocks:
            if full_block.initial_index < self.rule[0]:
                output_cells_list = self.fill_head_of_cell_list_based_on_first_rule(output_cells_list,rule = self.rule,full_block = full_block)
            full_block_max_index = full_block.initial_index + full_block.block_len - 1
            max_possible_block_start_index = self.length - self.rule[-1]
            if full_block_max_index >= max_possible_block_start_index:
                output_cells_list = self.fill_tail_of_cell_list_based_on_last_rule(output_cells_list,rule = self.rule,full_block = full_block)
        self.update_cells_list(output_cells_list)

    def solve(self):
        base_problem = copy.deepcopy(self)
        self.identify_and_update_rule_element_indexes()
        if self.is_solved():
            return
        elif self.is_line_fully_defined_by_rule():
            self.fully_defined_solve()
        elif self.all_full_cell_found():
            self.all_full_cell_found_solve()
        elif self.is_splittable():
            splitted_problem = self.split()
            print(splitted_problem[0])
            print(splitted_problem[1])
            first_solved_problem = splitted_problem[0]
            first_solved_problem.solve()
            print(first_solved_problem)
            second_solved_problem = splitted_problem[1]
            second_solved_problem.solve()
            print(second_solved_problem)
            self = first_solved_problem + second_solved_problem
        else:
            if self.is_subject_to_overlap_solving():
                self.overlapping_solve()
            self.complete_full_blocks_with_max_rule_size_solve()
            self.extremities_fill_empty_solve()
            self.complete_extremities_full_block_solve()
        if self == base_problem:
            return
        else:
            self.solve()
