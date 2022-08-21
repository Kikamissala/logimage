from logimage.cell import Cell, CellState, InvalidCellStateModification
from logimage.rule import Rule, RuleList
from collections import Counter
import copy

import pandas as pd

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

    def __init__(self, rule : Rule, cells):
        if rule.compute_min_possible_len() > len(cells):
            print(rule)
            print(cells)
            raise InvalidProblem("Rule minimum corresponding cells size exceeds input cells size")
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

    def __getitem__(self,index):
        return self.cells[index]
    
    def raise_if_update_impossible(self,new_problem):
        for index, cell in enumerate(new_problem):
            self.cells[index].raise_if_update_impossible(cell)

    def __setitem__(self,index,value:Cell):
        if self.cells[index] == value:
            return
        else:
            try:
                self.cells[index].raise_if_update_impossible(value)
            except InvalidCellStateModification:
                raise InvalidProblem("unable to modify cell")
        self.cells[index] = value

    def __eq__(self,other):
        if isinstance(other,Problem):
            return (self.rule == other.rule) & (self.cells == other.cells)

    def __len__(self):
        return len(self.cells)

    def is_splittable(self):
        if self.length == 1:
            return False
        elif self.rule == []:
            return False
        elif (len([cell for cell in self.cells if cell.cell_state == CellState.full]) == self.length) or \
            (len([cell for cell in self.cells if cell.cell_state == CellState.empty]) == self.length):
            return False
        elif (self.cells[0].cell_state == CellState.empty) or (self.cells[-1].cell_state == CellState.empty):
            return True
        else:
            first_complete_full_block_with_rule_element_index = self.get_first_complete_full_block_with_rule_element_index()
            if first_complete_full_block_with_rule_element_index is not  None:
                return True
            else:
                return False

    def split_when_first_or_last_cell_is_empty(self):
        if (self.cells[0].cell_state == CellState.empty) or (self.cells[-1].cell_state == CellState.empty):
            if self.cells[0].cell_state == CellState.empty:
                numerized_series_list = pd.Series(self.numerize_cell_list())
                first_non_empty_index = Problem.find_first_index_of_non_value(numerized_series_list,0)
                first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(first_non_empty_index,0)
                return [first_part_problem, second_part_problem]
            elif self.cells[-1].cell_state == CellState.empty:
                reversed_numerized_series_list = pd.Series(self.numerize_cell_list()[::-1])
                reversed_first_non_empty_index = Problem.find_first_index_of_non_value(reversed_numerized_series_list,0)
                first_ending_empty_index = self.length - reversed_first_non_empty_index
                first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(first_ending_empty_index,len(self.rule)) 
                return [first_part_problem, second_part_problem]
        else:
            return []
    
    def split_when_identified_complete_full_block(self):
        first_complete_full_block_with_rule_element_index = self.get_first_complete_full_block_with_rule_element_index()
        if first_complete_full_block_with_rule_element_index is not  None:
            if first_complete_full_block_with_rule_element_index.initial_index == 0:
                first_index_after_empty_index = first_complete_full_block_with_rule_element_index.block_len + 1
                first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(first_index_after_empty_index,1)
            else:
                empty_cell_before_complete_block_index = first_complete_full_block_with_rule_element_index.initial_index - 1
                rule_index = self.cells[first_complete_full_block_with_rule_element_index.initial_index].rule_element_index
                first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(empty_cell_before_complete_block_index,rule_index)
            return [first_part_problem, second_part_problem]
        else:
            return []

    def split_when_gap_between_empty_and_identified_is_too_short_to_put_other_rule_element(self):
        identified_cells_indexes = [index for index, cell in enumerate(self.cells)  if cell.rule_element_index is not None]
        empty_cells_indexes = [index for index, cell in enumerate(self.cells) if cell.cell_state == CellState.empty]
        if (identified_cells_indexes == []) or (empty_cells_indexes == []):
            return []
        else:
            for empty_index in empty_cells_indexes:
                if (empty_index - 1)  not in empty_cells_indexes:
                    identified_cell_indexes_before_current_index = [index for index in identified_cells_indexes if index < empty_index]
                    if identified_cell_indexes_before_current_index != []:
                        max_index_of_identified_cell_before_current_index = max(identified_cell_indexes_before_current_index)
                        rule_element_index = self.cells[max_index_of_identified_cell_before_current_index].rule_element_index
                        if rule_element_index == len(self.rule) - 1:
                            splitting_index = empty_index
                            splitting_rule_element_index = len(self.rule)
                            first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(splitting_index,splitting_rule_element_index)
                            return [first_part_problem, second_part_problem]
                        else:
                            next_rule_element = self.rule[rule_element_index + 1]
                            gap_len_between_identified_and_empty = empty_index - max_index_of_identified_cell_before_current_index - 1
                            if gap_len_between_identified_and_empty < next_rule_element + 1:
                                splitting_index = empty_index
                                splitting_rule_element_index = rule_element_index + 1
                                first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(splitting_index,splitting_rule_element_index)
                                return [first_part_problem, second_part_problem]
                if (empty_index + 1 not in empty_cells_indexes):
                    identified_cell_indexes_after_current_index = [index for index in identified_cells_indexes if index > empty_index]
                    if identified_cell_indexes_after_current_index != []:
                        min_index_of_identified_cell_after_current_index = min(identified_cell_indexes_after_current_index)
                        rule_element_index = self.cells[min_index_of_identified_cell_after_current_index].rule_element_index
                        if rule_element_index == 0:
                            splitting_index = empty_index
                            splitting_rule_element_index = 0
                            first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(splitting_index,splitting_rule_element_index)
                            return [first_part_problem, second_part_problem]
                        else:
                            last_rule_element = self.rule[rule_element_index - 1]
                            gap_len_between_identified_and_empty = min_index_of_identified_cell_after_current_index - empty_index - 1
                            if gap_len_between_identified_and_empty < last_rule_element + 1:
                                splitting_index = empty_index
                                splitting_rule_element_index = rule_element_index
                                first_part_problem, second_part_problem = self.get_problem_parts_from_problem_when_splitted_at_index_at_rule_element_index(splitting_index,splitting_rule_element_index)
                                return [first_part_problem, second_part_problem]
        return []

    def split_if_possible(self):
        original_problem = copy.deepcopy(self)
        if self.length == 1:
            return [original_problem]
        elif self.rule == []:
            return [original_problem]
        elif (len([cell for cell in self.cells if cell.cell_state == CellState.full]) == self.length) or \
            (len([cell for cell in self.cells if cell.cell_state == CellState.empty]) == self.length):
            return [original_problem]
        else:
            splitted_problem = self.split_when_first_or_last_cell_is_empty()
            if len(splitted_problem) == 2:
                return splitted_problem
            splitted_problem = self.split_when_identified_complete_full_block()
            if len(splitted_problem) == 2:
                return splitted_problem
            splitted_problem = self.split_when_gap_between_empty_and_identified_is_too_short_to_put_other_rule_element()
            if len(splitted_problem) == 2:
                return splitted_problem
        return [original_problem]


    def get_first_complete_full_block_with_rule_element_index(self):
        complete_full_blocks = self.identify_complete_full_blocks()
        sorted_complete_full_blocks = sorted(complete_full_blocks, key=lambda d: d.initial_index)
        for full_block in sorted_complete_full_blocks:
            rule_element_index_none_list = [cell.rule_element_index is None for cell in self.cells[full_block.initial_index:full_block.last_index]]
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
    def find_last_index_of_value(input_series, value):
        if len(input_series) == 0:
            return None
        else:
            if input_series.iloc[len(input_series)-1] == value:
                return input_series.index[len(input_series)-1]
            else:
                return Problem.find_last_index_of_value(input_series[:-1],value)

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
            elif (numerized_list_series[first_index_before_full_series] != 0) or (numerized_list_series[first_index_after_full_series] != 0):
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
        list_of_identified_full_blocks = self.identify_complete_full_blocks()
        first_undefined_cell_index = self.first_undefined_cell_index()
        last_undefined_cell_index = self.last_undefined_cell_index()
        return list_of_identified_full_blocks, first_undefined_cell_index, last_undefined_cell_index

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

    def get_full_indexes_before_first_empty_when_space_too_little_for_two_rule_elements(self,numerized_list,rule):
        numerized_series = pd.Series(numerized_list)
        first_full_index = Problem.find_first_index_of_value(numerized_series,1)
        all_full_indexes_before_first_empty = []
        if len(rule) > 1:
            first_empty_index = Problem.find_first_index_of_value(numerized_series,0)
            if (first_empty_index is not None):
                if first_full_index < first_empty_index:
                    number_of_cells_before_first_empty = first_empty_index
                    minimum_size_of_problem_with_two_first_rules = Rule(rule[0:2]).compute_min_possible_len()
                    if number_of_cells_before_first_empty < minimum_size_of_problem_with_two_first_rules:
                        all_full_indexes_before_first_empty = [index for index,value in enumerate(numerized_list) if (value== 1) & (index < first_empty_index)]
        return all_full_indexes_before_first_empty
                                        

    def set_rule_element_to_0_if_space_between_start_and_first_empty_too_short_for_two_rule_elements(self,rule_element_indexes):
        all_full_indexes_before_first_empty = self.get_full_indexes_before_first_empty_when_space_too_little_for_two_rule_elements(self.numerize_cell_list(),rule = self.rule)
        rule_element_index = 0
        for index in all_full_indexes_before_first_empty:
            rule_element_indexes[index] = rule_element_index
        return rule_element_indexes
    
    def set_rule_element_to_max_rule_element_index_if_space_between_last_empty_and_end_too_short_for_two_last_rule_elements(self,rule_element_indexes):
        all_full_indexes_before_first_empty_reversed = self.get_full_indexes_before_first_empty_when_space_too_little_for_two_rule_elements(self.numerize_cell_list()[::-1],rule = self.rule[::-1])
        all_full_indexes_after_last_empty = [self.length - 1 - index for index in all_full_indexes_before_first_empty_reversed]
        all_full_indexes_before_first_empty_reversed
        rule_element_index = len(self.rule) -1
        for index in all_full_indexes_after_last_empty:
            rule_element_indexes[index] = rule_element_index
        return rule_element_indexes

    def identify_rule_element_indexes_on_complete_blocks(self, rule_element_indexes):
        list_of_identified_full_blocks, first_undefined_cell_index, last_undefined_cell_index = self.get_elements_for_identifying_rule_element_indexes()
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

    def identify_rule_element_indexes_when_part_of_block_is_identified(self, rule_element_indexes):
        list_of_incomplete_full_blocks = self.identify_incomplete_full_blocks()
        for incomplete_full_block in list_of_incomplete_full_blocks:
            first_index = incomplete_full_block.initial_index
            last_index = incomplete_full_block.initial_index + incomplete_full_block.block_len
            rule_element_indexes_in_block = [value for value in rule_element_indexes[first_index:last_index] if value is not None]
            number_of_defined_rule_elements = len(rule_element_indexes_in_block)
            if (number_of_defined_rule_elements != 0) & (number_of_defined_rule_elements != incomplete_full_block.block_len):
                for index in range(first_index,last_index):
                    rule_element_indexes[index] = rule_element_indexes_in_block[0]
        return rule_element_indexes

    def identify_rule_element_indexes_when_gap_between_extremity_and_block_cannot_fit_first_or_last_rule_element(self, rule_element_indexes):
        list_of_incomplete_full_blocks = self.identify_incomplete_full_blocks()
        for incomplete_full_block in list_of_incomplete_full_blocks:
            first_index = incomplete_full_block.initial_index
            last_index = incomplete_full_block.initial_index + incomplete_full_block.block_len
            gap_between_start_and_block = incomplete_full_block.initial_index
            gap_between_end_and_last_block = self.length - incomplete_full_block.last_index
            if gap_between_start_and_block < self.rule[0] + 1:
                for index in range(first_index,last_index):
                    rule_element_indexes[index] = 0
            elif gap_between_end_and_last_block < self.rule[len(self.rule)-1] + 1:
                for index in range(first_index,last_index):
                    rule_element_indexes[index] = len(self.rule)-1
        return rule_element_indexes

    def identify_rule_element_indexes_when_max_rule_element_unique_and_incomplete_block_size_exceeds_second_max_rule_element(self, rule_element_indexes):
        list_of_incomplete_full_blocks = self.identify_incomplete_full_blocks()
        max_rule_element = max(self.rule)
        max_rule_element_index = [index for index, value in enumerate(self.rule) if value == max_rule_element][0]
        number_of_max_rule_element_in_rule = len([value for value in self.rule if value == max_rule_element])
        rule_without_max = [value for value in self.rule if value != max_rule_element]
        if len(rule_without_max) > 0:
            second_max_rule_element = max(rule_without_max)
        else:
            second_max_rule_element = None
        for incomplete_full_block in list_of_incomplete_full_blocks:
            first_index = incomplete_full_block.initial_index
            last_index = incomplete_full_block.initial_index + incomplete_full_block.block_len
            rule_element_indexes_in_block = [value for value in rule_element_indexes[first_index:last_index] if value is not None]
            number_of_defined_rule_elements = len(rule_element_indexes_in_block)
            if number_of_defined_rule_elements == 0:
                if number_of_max_rule_element_in_rule == 1:
                    if second_max_rule_element is not None:
                        if incomplete_full_block.block_len > second_max_rule_element:
                            for index in range(first_index,last_index):
                                rule_element_indexes[index] = max_rule_element_index
        return rule_element_indexes

    def identify_rule_element_indexes(self):
        rule_element_indexes = self.get_rule_element_indexes()
        if len(self.rule) == 0:
            return rule_element_indexes
        if len(self.rule) == 1:
            all_full_indexes = [index for index,cell in enumerate(self.cells) if cell.cell_state == CellState.full]
            for index in all_full_indexes:
                rule_element_indexes[index] = 0
            return rule_element_indexes
        rule_element_indexes = self.identify_rule_element_indexes_on_complete_blocks(rule_element_indexes)
        numerized_series = pd.Series(self.numerize_cell_list())
        first_full_index = Problem.find_first_index_of_value(numerized_series,1)
        if first_full_index is not None:
            rule_element_indexes = self.set_rule_element_to_0_if_space_between_start_and_first_empty_too_short_for_two_rule_elements(rule_element_indexes)
            rule_element_indexes = self.set_rule_element_to_max_rule_element_index_if_space_between_last_empty_and_end_too_short_for_two_last_rule_elements(rule_element_indexes)
        rule_element_indexes = self.identify_rule_element_indexes_when_part_of_block_is_identified(rule_element_indexes)
        rule_element_indexes = self.identify_rule_element_indexes_when_gap_between_extremity_and_block_cannot_fit_first_or_last_rule_element(rule_element_indexes)
        rule_element_indexes = self.identify_rule_element_indexes_when_max_rule_element_unique_and_incomplete_block_size_exceeds_second_max_rule_element(rule_element_indexes)
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

    def get_number_of_cell_in_state(self,cell_state:CellState):
        return len([cell for cell in self.cells if cell.cell_state == cell_state])

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
    
    def get_updated_state_indexes(self,other):
        self.raise_if_update_impossible(other)
        updated_state_indexes = []
        for index, (first, second) in enumerate(zip(self.cells, other.cells)):
            if first.cell_state != second.cell_state:
                updated_state_indexes.append(index)
        return updated_state_indexes

    def update_cells_list(self, new_cell_lists):
        output_problem = copy.deepcopy(self)
        for index, cell in enumerate(new_cell_lists):
            if cell.cell_state == CellState.undefined:
                continue
            else:
                output_problem[index] = new_cell_lists[index]
        return output_problem
    
    def update_cells_list_inplace(self, new_cell_lists):
        for index, cell in enumerate(new_cell_lists):
            if cell.cell_state == CellState.undefined:
                continue
            else:
                self.__setitem__(index,new_cell_lists[index])

    def fully_defined_solve(self):
        new_cells_list = []
        index = 0
        while index < len(self.rule) - 1:
            rule_element = self.rule[index]
            new_cells_list += rule_element.translate_to_list() + [Cell(CellState.empty)]
            index += 1
        last_rule_element = self.rule[index]
        new_cells_list += last_rule_element.translate_to_list()
        solved_problem = self.update_cells_list(new_cells_list)
        return solved_problem
    
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
        solved_problem = self.update_cells_list(output_cells_list)
        return solved_problem
    
    def all_full_cell_found(self):
        full_cells_number = len([cell for cell in self.cells if cell.cell_state == CellState.full])
        return full_cells_number == sum(self.rule)
    
    def all_full_cell_found_solve(self):
        output_cells_list = copy.deepcopy(self.cells)
        for index in range(0,self.length):
            if output_cells_list[index].cell_state == CellState.undefined:
                output_cells_list[index] = Cell(CellState.empty)
        solved_problem = self.update_cells_list(output_cells_list)
        solved_problem.identify_and_update_rule_element_indexes()
        return solved_problem

    def index_strict_upper_born_to_fill_with_empty(self,cell_list = None, rule = None, first_rule_element_index = None):
        if cell_list == None:
            cell_list = self.cells
        if rule == None:
            rule = self.rule
        if first_rule_element_index == None:
            first_rule_element_index = 0
        if cell_list[0].cell_state == CellState.full:
            return 0
        first_undefined_index = next((index for index, cell in enumerate(cell_list) if cell.cell_state == CellState.undefined),None)
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
        output_cells_list = copy.deepcopy(self.cells)
        for index in range(0,last_index_to_fill_with_empty):
            if output_cells_list[index].cell_state == CellState.undefined:
                output_cells_list[index] = Cell(CellState.empty)
        solved_problem = Problem(self.rule, output_cells_list)
        return solved_problem

    def tail_fill_empty_solve(self):
        first_index_to_fill_with_empty = self.index_strict_lower_born_to_fill_with_empty()
        output_cells_list = copy.deepcopy(self.cells)
        for index in range(first_index_to_fill_with_empty + 1,self.length):
            if output_cells_list[index].cell_state == CellState.undefined:
                output_cells_list[index] = Cell(CellState.empty)
        solved_problem = Problem(self.rule, output_cells_list)
        return solved_problem
    
    def extremities_fill_empty_solve(self):
        output_problem = copy.deepcopy(self)
        output_problem = output_problem.head_fill_empty_solve()
        output_problem = output_problem.tail_fill_empty_solve()
        return output_problem

    def complete_full_blocks_with_max_rule_size_solve(self):
        incomplete_full_blocks = self.identify_incomplete_full_blocks()
        output_cells_list = copy.deepcopy(self.cells)
        max_rule_element_value = max(self.rule)
        completable_full_blocks = [full_block for full_block in incomplete_full_blocks if full_block.block_len == max_rule_element_value]
        for full_block in completable_full_blocks:
            first_index_after_block = full_block.initial_index + full_block.block_len
            if full_block.initial_index == 0:
                output_cells_list[first_index_after_block] = Cell(CellState.empty)
            elif first_index_after_block == self.length:
                output_cells_list[full_block.initial_index - 1] = Cell(CellState.empty)
            else:
                output_cells_list[first_index_after_block] = Cell(CellState.empty)
                output_cells_list[full_block.initial_index - 1] = Cell(CellState.empty)
        solved_problem = self.update_cells_list(output_cells_list)
        return solved_problem
    
    @staticmethod
    def all_undefined_to_full_in_range(cells_list,range_min, range_max, rule_element_index = None):
        for index in range(range_min,range_max):
            if cells_list[index].cell_state == CellState.undefined:
                cells_list[index] = Cell(CellState.full, rule_element_index=rule_element_index)
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
                    output_cells_list[index_after_full_block] = Cell(CellState.empty)
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
            return self
        for full_block in incomplete_full_blocks:
            if full_block.initial_index < self.rule[0]:
                output_cells_list = self.fill_head_of_cell_list_based_on_first_rule(output_cells_list,rule = self.rule,full_block = full_block)
            full_block_max_index = full_block.initial_index + full_block.block_len - 1
            max_possible_block_start_index = self.length - self.rule[-1]
            if full_block_max_index >= max_possible_block_start_index:
                output_cells_list = self.fill_tail_of_cell_list_based_on_last_rule(output_cells_list,rule = self.rule,full_block = full_block)
        solved_problem = self.update_cells_list(output_cells_list)
        return solved_problem

    def complete_gaps_between_full_with_same_rule_element_index_solve(self):
        output_cells_list = copy.deepcopy(self.cells)
        for rule_element_index, rule_element in enumerate(self.rule):
            indexes_with_rule_element =  [index for index,cell in enumerate(output_cells_list) if cell.rule_element_index == rule_element_index]
            len_indexes_with_rule_element = len(indexes_with_rule_element)
            if len_indexes_with_rule_element < 2:
                continue
            else:
                gap_between_first_and_last_index = indexes_with_rule_element[-1] - indexes_with_rule_element[0]
                if gap_between_first_and_last_index >= len_indexes_with_rule_element:
                    for index in range(indexes_with_rule_element[0],indexes_with_rule_element[-1]):
                        output_cells_list[index] = Cell(CellState.full,rule_element_index=rule_element_index)
        solved_problem = self.update_cells_list(output_cells_list)
        return solved_problem

    def incomplete_full_block_with_rule_element_has_rule_element_len_solve(self):
        output_cells_list = copy.deepcopy(self.cells)
        incomplete_full_blocks = self.identify_incomplete_full_blocks()
        for incomplete_full_block in incomplete_full_blocks:
            block_cells = output_cells_list[incomplete_full_block.initial_index:incomplete_full_block.last_index]
            identified_cells = [cell for cell in block_cells if cell.rule_element_index is not None]
            if len(identified_cells) > 0:
                rule_element_index = identified_cells[0].rule_element_index
                rule_element = self.rule[rule_element_index]
                if incomplete_full_block.block_len == rule_element:
                    if incomplete_full_block.initial_index >0:
                        output_cells_list[incomplete_full_block.initial_index - 1] = Cell(CellState.empty)
                    if incomplete_full_block.last_index < self.length:
                        output_cells_list[incomplete_full_block.last_index] = Cell(CellState.empty)
        solved_problem = self.update_cells_list(output_cells_list)
        return solved_problem

    def fitting_big_rule_element_in_only_available_spot_solve(self):
        output_cells_list = copy.deepcopy(self.cells)
        list_of_not_blocks_without_empty = []
        numerized_list_series = pd.Series(self.numerize_cell_list())
        list_of_non_zero_series = Problem.find_series_without_value(numerized_list_series,0)
        for non_zero_serie in list_of_non_zero_series:
            list_of_not_blocks_without_empty.append(FullBlock(block_len=len(non_zero_serie),initial_index=non_zero_serie.index[0]))
        if len(self.rule) > 0:
            max_rule_element = max(self.rule)
            max_rule_element_indexes = [index for index, rule_element in enumerate(self.rule) if rule_element == max_rule_element]
            if len(max_rule_element_indexes) == 1:
                max_rule_element_index = max_rule_element_indexes[0]
                eligibles_blocks_without_empty = [block for block in list_of_not_blocks_without_empty if (block.block_len >= max_rule_element)]
                if len(eligibles_blocks_without_empty) == 1:
                    eligible_block = eligibles_blocks_without_empty[0]
                    if (eligible_block.block_len < max_rule_element * 2):
                        max_overlap_index = eligible_block.initial_index + max_rule_element - 1
                        min_overlap_index = eligible_block.last_index - max_rule_element
                        for index in range(min_overlap_index,max_overlap_index + 1):
                            output_cells_list[index] = Cell(CellState.full, rule_element_index=max_rule_element_index)
        solved_problem = self.update_cells_list(output_cells_list)
        return solved_problem


    def solve(self):
        base_problem = copy.deepcopy(self)
        output_problem = copy.deepcopy(self)
        output_problem.identify_and_update_rule_element_indexes()
        splitted_problem = output_problem.split_if_possible()
        if len(splitted_problem) == 2:
            first_problem = splitted_problem[0]
            second_problem = splitted_problem[1]
            return first_problem.solve() + second_problem.solve()
        else:
            if output_problem.is_solved():
                return output_problem
            elif output_problem.is_line_fully_defined_by_rule():
                output_problem = output_problem.fully_defined_solve()
            elif output_problem.all_full_cell_found():
                output_problem = output_problem.all_full_cell_found_solve()
            else:
                if output_problem.is_subject_to_overlap_solving():
                    output_problem = output_problem.overlapping_solve()
                #else:
                    
                output_problem = output_problem.complete_full_blocks_with_max_rule_size_solve()
                output_problem = output_problem.extremities_fill_empty_solve()
                output_problem = output_problem.complete_extremities_full_block_solve()
                output_problem = output_problem.complete_gaps_between_full_with_same_rule_element_index_solve()
                output_problem = output_problem.incomplete_full_block_with_rule_element_has_rule_element_len_solve()
                output_problem = output_problem.fitting_big_rule_element_in_only_available_spot_solve()
            if output_problem == base_problem:
                return output_problem
            else:
                return output_problem.solve()


    # def solve_old(self):
    #     base_problem = copy.deepcopy(self)
    #     output_problem = copy.deepcopy(self)
    #     output_problem.identify_and_update_rule_element_indexes()
    #     if output_problem.is_splittable():
    #         splitted_problem = output_problem.split()
    #         first_problem = splitted_problem[0]
    #         second_problem = splitted_problem[1]
    #         return first_problem.solve_old() + second_problem.solve_old()
    #     else:
    #         if output_problem.is_solved():
    #             return output_problem
    #         elif output_problem.is_line_fully_defined_by_rule():
    #             output_problem = output_problem.fully_defined_solve()
    #         elif output_problem.all_full_cell_found():
    #             output_problem = output_problem.all_full_cell_found_solve()
    #         else:
    #             if output_problem.is_subject_to_overlap_solving():
    #                 output_problem = output_problem.overlapping_solve()
    #             output_problem = output_problem.complete_full_blocks_with_max_rule_size_solve()
    #             output_problem = output_problem.extremities_fill_empty_solve()
    #             output_problem = output_problem.complete_extremities_full_block_solve()
    #             output_problem = output_problem.complete_gaps_between_full_with_same_rule_element_index_solve()
    #         if output_problem == base_problem:
    #             return output_problem
    #         else:
    #             return output_problem.solve_old()
    
    def solve_inplace(self):
        base_problem = copy.deepcopy(self)
        output_problem = copy.deepcopy(self)
        output_problem.identify_and_update_rule_element_indexes()
        if output_problem.is_splittable():
            splitted_problem = output_problem.split()
            first_problem = splitted_problem[0]
            second_problem = splitted_problem[1]
            first_problem.solve_inplace()
            second_problem.solve_inplace()
            output_problem =  first_problem + second_problem
        else:
            if output_problem.is_solved():
                return
            elif output_problem.is_line_fully_defined_by_rule():
                output_problem = output_problem.fully_defined_solve()
            elif output_problem.all_full_cell_found():
                output_problem = output_problem.all_full_cell_found_solve()
            else:
                if output_problem.is_subject_to_overlap_solving():
                    output_problem = output_problem.overlapping_solve()
                output_problem = output_problem.complete_full_blocks_with_max_rule_size_solve()
                output_problem = output_problem.extremities_fill_empty_solve()
                output_problem = output_problem.complete_extremities_full_block_solve()
        if output_problem == base_problem:
            return
        else:
            self.update_cells_list_inplace(output_problem.cells[:])
            self.solve_inplace()

class InvalidProblemDict(Exception):
    pass

class InvalidProblemDictAssignment(Exception):
    pass

class ProblemDict:
    
    def __init__(self,problems = None, rule_list = None, problem_size = None):
        if (rule_list is None) or (problem_size is None):
            if problems is None:
                raise InvalidProblemDict("cannot create ProblemDict without either rule_list and problem_size, or problems")
            else:
                self.check_input_problems(problems)
                self.problems = self.generate_problems_from_list(problems)
        else:
            self.problems = self.generate_problems_from_rule_list(rule_list,problem_size)
    
    def check_input_problems(self,problems):
        if len(problems) <= 1:
            return
        initial_len = len(problems[0])
        index = 1
        while index < len(problems):
            if len(problems[index]) != initial_len:
                raise InvalidProblemDict("Unable to create ProblemDict : All problems don't have the same size")
            index += 1

    def generate_problems_from_list(self,problems):
        output_dict = {index : problems[index] for index in range(0,len(problems))}
        return output_dict

    def generate_problems_from_rule_list(self, rule_list, problem_size):
        problems = []
        for rule in rule_list:
            problem = Problem(rule = rule, cells = [Cell()] * problem_size)
            problems.append(problem)
        output_dict = self.generate_problems_from_list(problems)
        return output_dict

    def __getitem__(self, index):
        return self.problems[index]

    def __setitem__(self, index, value:Problem):
        if value.rule != self.problems[index].rule:
            raise InvalidProblemDictAssignment("new problem doesn't have the same rules")
        new_problem_cells = value.cells
        for i, cell in enumerate(new_problem_cells):
            try:
                self.problems[index][i] = cell
            except InvalidProblem:
                raise InvalidProblemDictAssignment("new problem cells incompatible with old ones")
    
    def __len__(self):
        return len(self.problems)
    
    def __eq__(self,other):
        return self.problems == other.problems

    def items(self):
        return self.problems.items()
