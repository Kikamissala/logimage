import copy
from logimage.cell import Cell, CellState
from logimage.rule import Rule, RuleList, RuleSet
from logimage.problem import Problem, FullBlock, InvalidProblem, ProblemAddError, ProblemDict, InvalidProblemDict,\
    InvalidProblemDictAssignment
import pytest
import pandas as pd

def test_problem_raises_if_rule_minimum_len_exceeds_cells_size():
    with pytest.raises(InvalidProblem) as err:
        problem = Problem(rule = Rule([2,1]), cells = [Cell(), Cell(), Cell()])

def test_problem_contains_rule_and_list_of_cells_and_numerized_list_is_None_when_undefined():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell()])
    numerized_cell_list = problem.numerize_cell_list()
    assert(numerized_cell_list == [-1,-1,-1])

def test_problem_numerized_list_is_ones_when_full():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full), Cell(CellState.full), Cell(CellState.full)])
    numerized_cell_list = problem.numerize_cell_list()
    assert(numerized_cell_list == [1,1,1])

def test_problem_numerized_list_is_zeroes_when_full():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.empty), Cell(CellState.empty), Cell(CellState.empty)])
    numerized_cell_list = problem.numerize_cell_list()
    assert(numerized_cell_list == [0,0,0])

def test_getitem_in_problem_returns_cell_at_index():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full,rule_element_index=0), Cell(CellState.empty), Cell(CellState.empty)])
    first_cell = problem[0]
    assert(first_cell == Cell(CellState.full,rule_element_index=0))

def test_setitem_does_nothing_if_cell_is_the_same():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full,rule_element_index=0), Cell(CellState.empty), Cell(CellState.empty)])
    expected_problem = copy.deepcopy(problem)
    problem[0] = Cell(CellState.full,rule_element_index=0)
    assert(problem == expected_problem)

def test_setitem_modifies_state_to_empty_if_undefined():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.undefined), Cell(CellState.undefined), Cell(CellState.undefined)])
    problem[0] = Cell(CellState.empty)
    expected_problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.empty), Cell(CellState.undefined), Cell(CellState.undefined)])
    assert(problem == expected_problem)

def test_setitem_modifies_state_to_full_with_rule_element_index_if_undefined():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.undefined), Cell(CellState.undefined), Cell(CellState.undefined)])
    problem[0] = Cell(CellState.full,rule_element_index=0)
    expected_problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full,rule_element_index=0), Cell(CellState.undefined), Cell(CellState.undefined)])
    assert(problem == expected_problem)

def test_setitem_raises_if_modifying_already_defined_cell():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full,rule_element_index=0), Cell(CellState.empty), Cell(CellState.empty)])
    with pytest.raises(InvalidProblem) as err:
        problem[0] = Cell(CellState.empty)

def test_setitem_updates_rule_element_index_if_full_without_rule_element_index():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full), Cell(CellState.undefined), Cell(CellState.undefined)])
    problem[0] = Cell(CellState.full,rule_element_index=0)
    expected_problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full,rule_element_index=0), Cell(CellState.undefined), Cell(CellState.undefined)])
    assert(problem == expected_problem)

def test_setitem_raises_if_modifying_already_defined_rule_element_index():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full,rule_element_index=0), Cell(CellState.empty), Cell(CellState.empty)])
    with pytest.raises(InvalidProblem) as err:
        problem[0] = Cell(CellState.full, rule_element_index=1)

def test_is_problem_solved_returns_true_when_no_cell_is_undefined():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full), Cell(CellState.empty), Cell(CellState.full)])
    is_problem_solved = problem.is_solved()
    assert(is_problem_solved == True)

def test_problem_with_fully_defined_line_by_rule_returns_true_when_function_called():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell()])
    bool_is_line_fully_defined_by_rule = problem.is_line_fully_defined_by_rule()
    assert(bool_is_line_fully_defined_by_rule == True)

def test_get_different_state_indexes_returns_indexes_of_different_state():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell()])
    problem2 = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full), Cell(), Cell()])
    updated_state_indexes = problem.get_updated_state_indexes(problem2)
    assert(updated_state_indexes == [0])

def test_updating_cells_list_with_new_states_changes_cells_list():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell()])
    new_cells_list = [Cell(CellState.full), Cell(), Cell()]
    updated_problem = problem.update_cells_list(new_cells_list)
    assert(updated_problem.cells == new_cells_list)

def test_updating_cells_list_with_new_rule_element_index_sets_new_rule_element_index():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full), Cell(), Cell()])
    new_cells_list = [Cell(CellState.full,rule_element_index=0), Cell(), Cell()]
    updated_problem = problem.update_cells_list(new_cells_list)
    assert(updated_problem.cells == new_cells_list)

def test_updating_cells_list_with_new_cell_none_keeps_previous_cell():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full), Cell(), Cell()])
    new_cells_list = [Cell(), Cell(), Cell()]
    problem.update_cells_list(new_cells_list)
    expected_cells_list = [Cell(CellState.full), Cell(), Cell()]
    assert(problem.cells == expected_cells_list)

def test_solving_complete_problem_from_all_undefined_returns_fully_defined_list():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell()])
    solved_problem = problem.fully_defined_solve()
    assert((solved_problem.is_solved() == True) & (solved_problem.numerize_cell_list() == [1,0,1]))

def test_solving_complete_problem_with_len_4_and_rule_2_1_returns_numerized_list_with_1_1_0_1():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(), Cell(), Cell(), Cell()])
    solved_problem = problem.fully_defined_solve()
    assert((solved_problem.is_solved() == True) & (solved_problem.numerize_cell_list() == [1,1,0,1]))

def test_compute_number_of_freedom_degrees_of_fully_defined_rule_is_zero():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell()])
    nb_freedom_degrees = problem.compute_number_of_freedom_degrees()
    assert(nb_freedom_degrees == 0)

def test_compute_number_of_freedom_degrees_of_rule_with_1_1_and_len_4_is_one():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell(), Cell()])
    nb_freedom_degrees = problem.compute_number_of_freedom_degrees()
    assert(nb_freedom_degrees == 1)

def test_first_index_of_value_for_list_with_none_value():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    numerized_series = pd.Series([1])
    first_non_none_index = problem.find_first_index_of_value(numerized_series,None)
    assert(first_non_none_index is None)

def test_find_sequences_without_zeroes_in_empty_series_returns_empty_list():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    numerized_series = pd.Series(data=None,dtype='float64')
    list_of_not_zero_series = problem.find_series_without_value(numerized_series,0)
    expected_list_of_series = []
    assert(list_of_not_zero_series == expected_list_of_series)

def test_find_sequences_without_zeroes_in_only_zero_series_returns_empty_list():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    numerized_series = pd.Series([0])
    list_of_not_zero_series = problem.find_series_without_value(numerized_series,0)
    expected_list_of_series = []
    assert(list_of_not_zero_series == expected_list_of_series)

def test_find_sequences_without_zeroes_in_simple_series():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    numerized_series = pd.Series([1,1])
    list_of_not_zero_series = problem.find_series_without_value(numerized_series,0)
    expected_list_of_series = [pd.Series([1,1])]
    assert(all([list_of_not_zero_series[i].equals(expected_list_of_series[i]) for i in range(0,len(list_of_not_zero_series))]))

def test_find_sequences_without_zeroes_in_series_with_one_zero_at_the_end():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([1,1,0])
    list_of_not_zero_series = problem.find_series_without_value(numerized_series,0)
    expected_list_of_series = [pd.Series([1,1])]
    pd.testing.assert_series_equal(list_of_not_zero_series[0], expected_list_of_series[0])

def test_find_sequences_without_zeroes_in_series_with_one_zero_at_the_beginning():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.empty),Cell(CellState.full),Cell(CellState.full)])
    numerized_series = pd.Series([0,1,1])
    list_of_not_zero_series = problem.find_series_without_value(numerized_series,0)
    expected_list_of_series = [pd.Series([1,1],index=[1,2])]
    pd.testing.assert_series_equal(list_of_not_zero_series[0], expected_list_of_series[0])

def test_find_sequences_without_zeroes_in_series_with_one_zero_at_the_beginning_and_end():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([0,1,1,0])
    list_of_not_zero_series = problem.find_series_without_value(numerized_series,0)
    expected_list_of_series = [pd.Series([1,1],index=[1,2])]
    pd.testing.assert_series_equal(list_of_not_zero_series[0], expected_list_of_series[0])

def test_find_sequences_without_zeroes_in_series_with_two_non_zero_sequences():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([1,1,0,1])
    list_of_not_zero_series = problem.find_series_without_value(numerized_series,0)
    expected_list_of_series = [pd.Series([1,1],index=[0,1]),pd.Series([1],index=[3])]
    assert(all([list_of_not_zero_series[i].equals(expected_list_of_series[i]) for i in range(0,len(list_of_not_zero_series))]))

def test_find_sequences_without_zeroes_in_serie_with_none_values():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([1,1,0,-1]).astype("int64",errors ="ignore")
    list_of_not_zero_series = problem.find_series_without_value(numerized_series,0)
    expected_list_of_series = [pd.Series([1,1],index=[0,1]),pd.Series([-1],index=[3])]
    assert(all([list_of_not_zero_series[i].equals(expected_list_of_series[i]) for i in range(0,len(list_of_not_zero_series))]))

def test_find_sequences_without_zeroes_in_serie_with_none_value():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([1,1,0,-1])
    list_of_not_zero_series = problem.find_series_without_value(numerized_series,0)
    expected_list_of_series = [pd.Series([1,1]),pd.Series([-1],index = [3])]
    assert(all([list_of_not_zero_series[i].equals(expected_list_of_series[i]) for i in range(0,len(list_of_not_zero_series))]))

def test_get_series_of_unique_values_with_several_series_of_1():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined)])
    numerized_list_series = pd.Series(problem.numerize_cell_list())
    list_of_consecutive_full_series = Problem.find_series_with_unique_value(numerized_list_series,1)
    expected_second_series = pd.Series([1],index=[6])
    pd.testing.assert_series_equal(list_of_consecutive_full_series[1],expected_second_series)

def test_identify_full_rules_from_problem_cells_with_one_full_cell():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    identified_rules = problem.identify_complete_full_blocks()
    assert(identified_rules == [FullBlock(block_len=1,initial_index=0)])

def test_identify_full_rules_from_problem_cells_with_two_full_cell():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    identified_rules = problem.identify_complete_full_blocks()
    assert(identified_rules == [FullBlock(block_len = 2, initial_index = 0)])

def test_identify_full_rules_from_problem_cells_with_two_full_cells_and_one_empty_cell():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    identified_rules = problem.identify_complete_full_blocks()
    assert(identified_rules == [FullBlock(block_len = 2, initial_index = 0)])

def test_identify_full_rules_from_problem_cells_with_rules_2_1_and_2_full_cells_one_empty_one_full():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full)])
    identified_blocks = problem.identify_complete_full_blocks()
    assert(identified_blocks == [FullBlock(block_len = 2, initial_index = 0),FullBlock(block_len = 1, initial_index = 3)])

def test_identify_full_rules_from_problem_cells_with_rules_2_1_and_2_full_cells_one_empty_one_undefined():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined)])
    identified_blocks = problem.identify_complete_full_blocks()
    assert(identified_blocks == [FullBlock(block_len = 2, initial_index = 0)])

def test_identify_full_rules_from_problem_cells_with_rules_2_1_and_2_full_cells_one_undefined_returns_empty():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined)])
    identified_blocks = problem.identify_complete_full_blocks()
    assert(identified_blocks == [])

def test_problem_object_contains_list_of_value_index_rule_element_index():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined)])
    assert((problem.cells[0].rule_element_index == 0))

def test_get_rule_element_indexes_from_cells_in_problem_returns_none_if_no_cell_full():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.get_rule_element_indexes()
    assert((identified_rule_element_indexes == [None]))

def test_get_rule_element_indexes_from_cells_in_problem_returns_none_if_empty_cell():
    problem = Problem(rule = Rule([]), cells = [Cell(CellState.empty)])
    identified_rule_element_indexes = problem.get_rule_element_indexes()
    assert((identified_rule_element_indexes == [None]))

def test_get_rule_element_indexes_from_cells_in_problem_returns_0_0_none_none():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.full,rule_element_index=0),Cell(CellState.empty),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.get_rule_element_indexes()
    assert((identified_rule_element_indexes == [0,0,None,None]))

def test_identify_rule_element_indexes_for_problem_with_len_one_and_one_full_cell():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [0]))

def test_identify_rule_element_indexes_for_problem_with_len_two_and_two_full_cells():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [0,0]))

def test_identify_rule_element_indexes_for_problem_with_len_one_and_one_undefined_cell_returns_list_with_none():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None]))

def test_identify_rule_element_indexes_for_problem_with_len_5_and_one_rule_element_value_3_gets_identified():
    problem = Problem(rule = Rule([3]), cells = [Cell(CellState.empty),Cell(CellState.full),Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,0,0,0,None]))

def test_identify_rule_element_indexes_for_solved_problem_with_len_5_and_rule_2_1():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full),Cell(CellState.empty)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [0,0,None,1,None]))

def test_identify_rule_element_if_block_is_at_left_extremity_of_line():
    problem = Problem(rule = Rule([2,2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [0,0,None,None,None]))

def test_identify_rule_element_if_block_is_at_right_extremity_of_line():
    problem = Problem(rule = Rule([2,2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full),Cell(CellState.full)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,None,None,1,1]))

def test_identify_rule_elements_if_all_blocks_of_same_size_in_rule_are_complete():
    problem = Problem(rule = Rule([1,2,2]), cells = [Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,None,1,1,None,2,2,None,None]))

def test_first_undefined_cell_index_of_problem_with_undefined_is_0():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])
    first_undefined_cell_index = problem.first_undefined_cell_index()
    assert((first_undefined_cell_index == 0))

def test_first_undefined_cell_index_of_problem_with_full_is_none():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    first_undefined_cell_index = problem.first_undefined_cell_index()
    assert((first_undefined_cell_index == None))

def test_first_undefined_cell_index_of_problem_with_full_then_undefined_is_1():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined)])
    first_undefined_cell_index = problem.first_undefined_cell_index()
    assert((first_undefined_cell_index == 2))

def test_last_undefined_cell_index_of_problem_with_undefined_is_0():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])
    first_undefined_cell_index = problem.last_undefined_cell_index()
    assert((first_undefined_cell_index == 0)) 

def test_last_undefined_cell_index_of_problem_with_full_is_none():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    first_undefined_cell_index = problem.last_undefined_cell_index()
    assert((first_undefined_cell_index == None))

def test_last_undefined_cell_index_of_problem_with_full_then_undefined_is_1():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full)])
    first_undefined_cell_index = problem.last_undefined_cell_index()
    assert((first_undefined_cell_index == 0))

def test_identify_rule_element_if_block_at_left_extremity_with_one_empty_at_left():
    problem = Problem(rule = Rule([2,2]), cells = [Cell(CellState.empty),Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,0,0,None,None]))

def test_identify_rule_element_if_two_full_blocks_at_left_extremity_returns_index_0_and_1_rule_element():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [0,None,1,None,None]))

def test_indentify_rule_element_index_in_solved_problem():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [0,None,1]))

def test_identify_rule_element_if_block_at_right_extremity_with_one_empty_at_right():
    problem = Problem(rule = Rule([2,2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,None,None,1,1,None]))

def test_identify_rule_element_if_two_full_blocks_at_right_extremity_returns_index_0_and_1_rule_element():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,None,0,None,1]))

def test_identify_rule_elemnt_if_only_one_rule_element_updates_all_full_cells_to_rule_element_0():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,None,0,None,None]))

def test_identify_rule_element_if_space_between_start_and_first_empty_with_full_in_between_is_too_little_for_2_first_rule_elements():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,None,0,None,None]))

def test_identify_rule_element_if_space_between_last_empty_and_end_with_full_in_between_is_too_little_for_2_last_rule_elements():
    problem = Problem(rule = Rule([1,2]), cells = [Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,None,1,None,None]))

def test_identify_rule_element_if_two_full_side_to_side_and_one_has_no_rule_element_index():
    problem = Problem(rule = Rule([1,2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full,rule_element_index=1),Cell(CellState.full),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,None,1,1,None]))

def test_identify_when_maximum_rule_element_value_is_unique_and_incomplete_block_size_higher_than_second_highest_rule_element_value():
    problem = Problem(rule = Rule([1,3]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.full),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,None,1,1,None]))

def test_identify_first_rule_element_when_space_between_start_and_first_full_cell_too_short_for_other_rule_element():
    problem = Problem(rule = Rule([1,2]), cells = [Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])
    identified_rule_element_indexes = problem.identify_rule_element_indexes()
    assert((identified_rule_element_indexes == [None,0,None,None,None]))

def test_update_rule_element_indexes_in_problem_changes_value_updates_attribute_and_corresponding_cells():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)])
    new_rule_element_indexes = [0,None,1]
    problem.update_rule_element_indexes(new_rule_element_indexes)
    expected_updated_cells = [Cell(CellState.full, rule_element_index = 0),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)]
    assert((problem.rule_elements_indexes == [0,None,1]) & (problem.cells == expected_updated_cells))

def test_update_rule_element_indexes_changing_already_defined_value_raises():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)])
    new_rule_element_indexes = [0,None,0]
    with pytest.raises(InvalidProblem) as err:
        problem.update_rule_element_indexes(new_rule_element_indexes)

def test_update_rule_element_indexes_in_problem_with_new_value_none_compared_to_already_known_index_keeps_previous_index():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)])
    new_rule_element_indexes = [None,None,None]
    problem.update_rule_element_indexes(new_rule_element_indexes)
    expected_updated_cells = [Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)]
    assert((problem.rule_elements_indexes == [None,None,1]) & (problem.cells == expected_updated_cells))

def test_scan_and_update_rule_element_indexes():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)])
    problem.identify_and_update_rule_element_indexes()
    expected_updated_cells = [Cell(CellState.full, rule_element_index = 0),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)]
    assert((problem.rule_elements_indexes == [0,None,1]) & (problem.cells == expected_updated_cells))

def test_problem_subject_to_overlap_solving_for_rule_2_and_len_3_is_true():
    problem = Problem(rule = Rule([2]), cells = [Cell(),Cell(),Cell()])
    bool_is_subject_to_overlap_solving = problem.is_subject_to_overlap_solving()
    assert(bool_is_subject_to_overlap_solving == True)

def test_problem_subject_to_overlap_solving_for_rule_1_and_len_2_is_false():
    problem = Problem(rule = Rule([1]), cells = [Cell(),Cell()])
    bool_is_subject_to_overlap_solving = problem.is_subject_to_overlap_solving()
    assert(bool_is_subject_to_overlap_solving == False)

def test_problem_subject_to_overlap_solving_for_rule_2_1_and_len_5_is_true():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(),Cell(),Cell(),Cell(),Cell()])
    bool_is_subject_to_overlap_solving = problem.is_subject_to_overlap_solving()
    assert(bool_is_subject_to_overlap_solving == True)

def test_problem_subject_to_overlap_solving_for_rule_2_1_and_len_6_is_true():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(),Cell(),Cell(),Cell(),Cell(),Cell()])
    bool_is_subject_to_overlap_solving = problem.is_subject_to_overlap_solving()
    assert(bool_is_subject_to_overlap_solving == False)

def test_min_starting_index_of_first_element_of_rule_is_0():
    rule = Rule([1])
    min_starting_index_of_rule = rule.compute_min_starting_indexes()
    assert(min_starting_index_of_rule == [0])

def test_min_starting_index_of_rule_1_1_is_0_2():
    rule = Rule([1,1])
    min_starting_index_of_rule = rule.compute_min_starting_indexes()
    assert(min_starting_index_of_rule == [0,2])

def test_min_starting_index_of_rule_1_2_1_is_0_2_5():
    rule = Rule([1,2,1])
    min_starting_index_of_rule = rule.compute_min_starting_indexes()
    assert(min_starting_index_of_rule == [0,2,5])

def test_overlapping_solving_for_rule_2_and_len_3_has_middle_element_full_with_rule_element_index_0():
    problem = Problem(rule = Rule([2]), cells = [Cell(),Cell(),Cell()])
    solved_problem = problem.overlapping_solve()
    expected_cells_list = [Cell(),Cell(CellState.full, rule_element_index=0),Cell()]
    assert(solved_problem.cells == expected_cells_list)

def test_problem_subject_to_overlap_solving_for_rule_2_1_and_len_5_has_second_element_full_and_rule_element_index_0():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(),Cell(),Cell(),Cell(),Cell()])
    solved_problem = problem.overlapping_solve()
    expected_cells_list = [Cell(),Cell(CellState.full, rule_element_index=0),Cell(),Cell(),Cell()]
    assert(solved_problem.cells == expected_cells_list)

def test_problem_subject_to_overlap_solving_for_rule_1_2_and_len_5_has_second_to_last_element_full_and_rule_element_index_0():
    problem = Problem(rule = Rule([1,2]), cells = [Cell(),Cell(),Cell(),Cell(),Cell()])
    solved_problem = problem.overlapping_solve()
    expected_cells_list = [Cell(),Cell(),Cell(),Cell(CellState.full, rule_element_index=1),Cell()]
    assert(solved_problem.cells == expected_cells_list)

def test_problem_all_full_cell_found_of_solved_problem_is_true():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    bool_all_full_cell_found = problem.all_full_cell_found()
    assert(bool_all_full_cell_found == True)

def test_problem_all_full_cell_found_of_problem_with_rule_2_and_two_full_cell_and_one_undefined_is_true():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full), Cell(CellState.full), Cell()])
    bool_all_full_cell_found = problem.all_full_cell_found()
    assert(bool_all_full_cell_found == True)

def test_problem_all_full_cell_found_of_problem_with_rule_2_1_and_two_full_cell_is_false():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full), Cell(CellState.full), Cell(),Cell()])
    bool_all_full_cell_found = problem.all_full_cell_found()
    assert(bool_all_full_cell_found == False)

def test_all_full_cell_found_solve_fills_undefined_cells():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full), Cell(CellState.full), Cell(),Cell()])
    solved_problem = problem.all_full_cell_found_solve()
    expected_cells_list = [Cell(CellState.full,rule_element_index=0), Cell(CellState.full,rule_element_index=0), Cell(CellState.empty),Cell(CellState.empty)]
    assert(solved_problem.cells == expected_cells_list)

def test_all_full_cell_found_solve_fills_undefined_cells_in_2_1_rule_problem():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full), Cell(CellState.full), Cell(),Cell(CellState.full),Cell(CellState.undefined)])
    solved_problem = problem.all_full_cell_found_solve()
    expected_cells_list = [Cell(CellState.full,rule_element_index=0), Cell(CellState.full,rule_element_index=0), Cell(CellState.empty),Cell(CellState.full,rule_element_index=1), Cell(CellState.empty)]
    assert(solved_problem.cells == expected_cells_list)

def test_index_strict_upper_born_to_fill_with_empty_of_problem_with_first_cell_undefined_then_empty_and_first_rule_element_2_is_1():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined), Cell(CellState.empty), Cell(CellState.full),Cell()])
    max_index = problem.index_strict_upper_born_to_fill_with_empty()
    assert(max_index == 1)

def test_index_strict_upper_born_to_fill_with_empty_of_problem_with_first_cell_full_is_0():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full), Cell(CellState.undefined)])
    max_index = problem.index_strict_upper_born_to_fill_with_empty()
    assert(max_index == 0)

def test_index_strict_upper_born_to_fill_with_empty_of_problem_with_first_two_cells_undefined_then_empty_and_rule_element_3_is_2():
    problem = Problem(rule = Rule([3]), cells = [Cell(CellState.undefined), Cell(CellState.undefined), Cell(CellState.empty),Cell(),Cell(),Cell()])
    max_index = problem.index_strict_upper_born_to_fill_with_empty()
    assert(max_index == 2)

def test_index_strict_upper_born_to_fill_with_empty_of_problem_with_first_cell_empty_then_undefined_then_empty_and_rule_element_2_is_2():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.empty), Cell(CellState.undefined), Cell(CellState.empty),Cell(),Cell(),Cell()])
    max_index = problem.index_strict_upper_born_to_fill_with_empty()
    assert(max_index == 2)

def test_index_strict_upper_born_to_fill_with_empty_of_problem_with_first_cell_undefined_then_full_and_first_rule_element_2_is_0():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined), Cell(CellState.full), Cell(),Cell()])
    max_index = problem.index_strict_upper_born_to_fill_with_empty()
    assert(max_index == 0)

def test_index_strict_upper_born_to_fill_with_empty_where_several_series_of_undefined_to_fill_return_index_of_last_series():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined), Cell(CellState.empty), Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full),Cell(CellState.undefined)])
    max_index = problem.index_strict_upper_born_to_fill_with_empty()
    assert(max_index == 3)

def test_index_strict_upper_born_to_fill_with_empty_of_problem_with_first_undefined_after_full_rule_element_2_is_0():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.empty), Cell(CellState.full), Cell(CellState.undefined),Cell(CellState.empty)])
    max_index = problem.index_strict_upper_born_to_fill_with_empty()
    assert(max_index == 0)

def test_index_strict_upper_born_to_fill_with_empty_of_problem_with_first_full_rule_element_index_0_is_first_full_index():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.empty), Cell(CellState.full,rule_element_index=0), Cell(CellState.undefined),Cell(CellState.empty)])
    max_index = problem.index_strict_upper_born_to_fill_with_empty()
    assert(max_index == 3)

def test_head_fill_empty_solve_with_first_cell_undefined_then_empty_and_first_rule_element_2_updates_problem():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined), Cell(CellState.empty), Cell(CellState.full),Cell()])
    solved_problem = problem.head_fill_empty_solve()
    expected_cells_list = [Cell(CellState.empty), Cell(CellState.empty), Cell(CellState.full),Cell()]
    assert(solved_problem.cells == expected_cells_list)

def test_index_strict_lower_born_to_fill_with_empty_of_problem_with_last_cell_undefined_then_empty_and_last_rule_element_2_is_2():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined), Cell(CellState.full), Cell(CellState.empty),Cell(CellState.undefined)])
    min_index = problem.index_strict_lower_born_to_fill_with_empty()
    assert(min_index == 2)

def test_index_strict_lower_born_to_fill_with_empty_of_problem_with_last_cell_undefined_then_empty_and_rule_1_2_is_1():
    problem = Problem(rule = Rule([1,2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined), Cell(CellState.full), Cell(CellState.empty),Cell(CellState.undefined)])
    min_index = problem.index_strict_lower_born_to_fill_with_empty()
    assert(min_index == 4)

def test_index_strict_lower_born_to_fill_with_empty_of_problem_with_last_full_rule_element_index_max_rule_element_index_is_last_full_index():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined), Cell(CellState.full,rule_element_index=0), Cell(CellState.empty),Cell(CellState.undefined)])
    max_index = problem.index_strict_lower_born_to_fill_with_empty()
    assert(max_index == 2)

def test_tail_fill_empty_solve_with_last_cell_undefined_then_empty_and_last_rule_element_2_updates_problem():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined), Cell(CellState.full), Cell(CellState.empty),Cell(CellState.undefined)])
    solved_problem = problem.tail_fill_empty_solve()
    expected_cells_list = [Cell(CellState.undefined), Cell(CellState.full), Cell(CellState.empty),Cell(CellState.empty)]
    assert(solved_problem.cells == expected_cells_list)

def test_tail_fill_empty_solve_with_complex_problem():
    problem = Problem(rule = Rule([14]), cells = [Cell(CellState.undefined)] * 4 + [Cell(CellState.empty)]+ [Cell(CellState.undefined)] + [Cell(CellState.full, rule_element_index=0)] * 8 + [Cell(CellState.undefined)]*6)
    solved_problem = problem.extremities_fill_empty_solve()
    expected_cells_list = [Cell(CellState.empty)] * 5 + [Cell(CellState.undefined)] + [Cell(CellState.full, rule_element_index=0)] * 8 + [Cell(CellState.undefined)]*6
    assert(solved_problem.cells == expected_cells_list)

def test_get_incomplete_full_blocks_from_problem_with_one_full_returns_empty_list():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    incomplete_full_blocks = problem.identify_incomplete_full_blocks()
    assert(incomplete_full_blocks == [])

def test_get_incomplete_full_blocks_from_problem_with_2_full_one_undefined_returns_one_full_block():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.undefined)])
    incomplete_full_blocks = problem.identify_incomplete_full_blocks()
    assert(incomplete_full_blocks == [FullBlock(block_len=2,initial_index=0)])

def test_get_incomplete_full_blocks_from_problem_with_2_full_one_undefined_and_one_full_returns_2_full_block():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.full)])
    incomplete_full_blocks = problem.identify_incomplete_full_blocks()
    assert(incomplete_full_blocks == [FullBlock(block_len=2,initial_index=0),FullBlock(block_len=1,initial_index=3)])

def test_get_incomplete_full_blocks_from_problem_with_2_full_one_empty_returns_empty_list():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    incomplete_full_blocks = problem.identify_incomplete_full_blocks()
    assert(incomplete_full_blocks == [])

def test_get_incomplete_full_blocks_from_problem_with_one_complete_full_block_then_incomplete_returns_second_block():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined)])
    incomplete_full_blocks = problem.identify_incomplete_full_blocks()
    assert(incomplete_full_blocks == [FullBlock(block_len=1,initial_index=3)])

def test_get_incomplete_full_blocks_from_problem_with_2_full_blocks_of_len_1_separated_from_extremity_by_1_undefined_returns_two_full_block_of_len_1():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined)])
    incomplete_full_blocks = problem.identify_incomplete_full_blocks()
    assert(incomplete_full_blocks == [FullBlock(block_len=1,initial_index=1),FullBlock(block_len=1,initial_index=6)])

def test_complete_full_blocks_with_max_rule_size_solve_of_problem_with_rule_1_and_1_full_between_2_undefined():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined)])
    solved_problem = problem.complete_full_blocks_with_max_rule_size_solve()
    expected_cells = [Cell(CellState.empty),Cell(CellState.full),Cell(CellState.empty)]
    assert(solved_problem.cells == expected_cells)

def test_complete_full_blocks_with_max_rule_size_solve_of_problem_with_rule_1_and_1_full_then_1_undefined():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full),Cell(CellState.undefined)])
    solved_problem = problem.complete_full_blocks_with_max_rule_size_solve()
    expected_cells = [Cell(CellState.full),Cell(CellState.empty)]
    assert(solved_problem.cells == expected_cells)

def test_complete_full_blocks_with_max_rule_size_solve_of_problem_with_rule_1_and_1_undefined_then_1_full():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined),Cell(CellState.full)])
    solved_problem = problem.complete_full_blocks_with_max_rule_size_solve()
    expected_cells = [Cell(CellState.empty),Cell(CellState.full)]
    assert(solved_problem.cells == expected_cells)

def test_complete_full_blocks_with_max_rule_size_solve_of_problem_with_rule_2_and_full_block_size_less_no_update():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.full)])
    problem.complete_full_blocks_with_max_rule_size_solve()
    expected_cells = [Cell(CellState.undefined),Cell(CellState.full)]
    assert(problem.cells == expected_cells)

def test_complete_full_blocks_at_extremities_solve_with_rule_1_and_full_then_undefined_complete_first_block():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full),Cell(CellState.undefined)])
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.empty)]
    assert(solved_problem.cells == expected_cells)

def test_complete_full_blocks_at_extremities_solve_with_rule_2_and_full_then_undefined_complete_first_block():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.undefined)])
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.full,rule_element_index=0)]
    assert(solved_problem.cells == expected_cells)

def test_complete_full_blocks_at_extremities_solve_with_rule_2_and_undefined_then_full_complete_first_cell_full():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.full)])
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.full,rule_element_index=0)]
    assert(solved_problem.cells == expected_cells)

def test_complete_full_blocks_at_extremities_solve_with_rule_1_and_ndefined_then_full_complete_first_cell_empty():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined),Cell(CellState.full)])
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.empty),Cell(CellState.full,rule_element_index=0)]
    assert(solved_problem.cells == expected_cells)

def test_complete_full_blocks_at_extremities_solve_with_rule_1_and_full_then_2_undefined_complete_second_block_empty():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined)])
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.empty),Cell(CellState.undefined)]
    assert(solved_problem.cells == expected_cells)

def test_complete_full_blocks_at_extremities_solve_with_rule_1_and_2_undefined_then_full_complete_second_cell_empty():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full)])
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full,rule_element_index=0)]
    assert(solved_problem.cells == expected_cells)

def test_complete_full_blocks_at_extremities_solve_with_rule_1_1_len_3_and_undefined_between_2_full_fills_with_empty():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.full)])
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)]
    assert(solved_problem.cells == expected_cells)

def test_full_block_before_first_possible_end_index_fills_full_until_this_index():
    problem = Problem(rule = Rule([3]), cells = [Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined)])
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.undefined),Cell(CellState.full,rule_element_index=0),Cell(CellState.full,rule_element_index=0),Cell(CellState.undefined)]
    assert(solved_problem.cells == expected_cells)

def test_full_block_after_last_possible_first_index_fills_full_until_this_index():
    problem = Problem(rule = Rule([3]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined)])
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.undefined),Cell(CellState.full,rule_element_index=0),Cell(CellState.full,rule_element_index=0),Cell(CellState.undefined)]
    assert(solved_problem.cells == expected_cells)

def test_both_extremity_can_be_partially_filled_works():
    problem = Problem(rule = Rule([3,3]), cells = [Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined)])
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.undefined),Cell(CellState.full,rule_element_index=0),Cell(CellState.full,rule_element_index=0),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full,rule_element_index=1),Cell(CellState.full,rule_element_index=1),Cell(CellState.undefined)]
    assert(solved_problem.cells == expected_cells)

def test_complete_extremities_full_block_solve_with_complex_problem_works():
    problem = Problem(Rule([6, 1, 4]), cells = [Cell(CellState.undefined)] * 2 + [Cell(CellState.full)] + [Cell(CellState.undefined)] * 17)
    solved_problem = problem.complete_extremities_full_block_solve()
    expected_cells = [Cell(CellState.undefined)] * 2 + [Cell(CellState.full,rule_element_index=0)] * 4 + [Cell(CellState.undefined)] * 14
    assert(solved_problem.cells == expected_cells)

# Rajouter la possibilité de remplir entre 2 cells full avec le même rule_element_index, des fulls avec 
# le même rule_element_index.

def test_complete_gap_between_two_full_with_same_rule_element_index():
    problem = Problem(Rule([3]), cells = [Cell(CellState.undefined)] * 2 + [Cell(CellState.full,rule_element_index=0)] + [Cell(CellState.undefined)] + [Cell(CellState.full,rule_element_index=0)] + [Cell(CellState.undefined)] * 2)
    solved_problem = problem.complete_gaps_between_full_with_same_rule_element_index_solve()
    expected_cells = [Cell(CellState.undefined)] * 2 + [Cell(CellState.full,rule_element_index=0)] * 3 + [Cell(CellState.undefined)] * 2
    assert(solved_problem.cells == expected_cells)

# Rajouter aussi la possibilité de remplir les blocs avec un empty en bout et au moins un rule_element_index connu

# Il faut aussi permettre de compléter les blocs qui avec rule_element_index identifiés qui font la 
# taille de l'indice correspondant.

def test_complete_block_if_rule_element_idex_known_and_block_len_equals_rule_element():
    problem = Problem(Rule([1,2]), cells = [Cell(CellState.undefined)] * 2 + [Cell(CellState.full,rule_element_index=0)] + [Cell(CellState.undefined)] * 5)
    solved_problem = problem.incomplete_full_block_with_rule_element_has_rule_element_len_solve()
    expected_cells = [Cell(CellState.undefined)] + [Cell(CellState.empty)]  + [Cell(CellState.full,rule_element_index=0)] +  [Cell(CellState.empty)] + [Cell(CellState.undefined)] * 4
    assert(solved_problem.cells == expected_cells)
# Potentiellement ajouter la feature qui comble les undefined entre deux blocs vides dont l'écart est inférieur à la rule 
# la plus faible de la ligne


def test_problem_with_big_block_that_can_only_fit_in_one_non_empty_block_and_can_overlap_there():
    problem = Problem(Rule([1,2]), cells = [Cell(CellState.empty)] +[Cell(CellState.undefined)] +  [Cell(CellState.empty)] + [Cell(CellState.undefined)] +  [Cell(CellState.empty)] + [Cell(CellState.undefined)]*2)
    solved_problem = problem.fitting_big_rule_element_in_only_available_spot_solve()
    expected_cells = [Cell(CellState.empty)] +[Cell(CellState.undefined)] +  [Cell(CellState.empty)] + [Cell(CellState.undefined)] +  [Cell(CellState.empty)] + [Cell(CellState.full,rule_element_index=1)]*2
    assert(solved_problem.cells == expected_cells)

# Rajouter la règle selon laquelle si on a dans les blocs avec des règles identifiées, des blocs consécutifs
# On peut ajouter des vides entre les 2 en imaginant la case la plus à droite du premier bloc, et celle là plus à gauche du suivant
def test_problem_with_one_rule_and_identified_cell_fills_empty_outside_of_range():
    problem = Problem(Rule([2]), cells = [Cell(CellState.undefined)]*2 +[Cell(CellState.full,rule_element_index=0)] +  [Cell(CellState.undefined)]* 3)
    solved_problem = problem.fill_empty_between_indentified_blocks_solve()
    print(solved_problem)
    expected_cells = [Cell(CellState.empty)] +[Cell(CellState.undefined)] +  [Cell(CellState.full,rule_element_index=0)] + [Cell(CellState.undefined)] +  [Cell(CellState.empty)] * 2
    assert(solved_problem.cells == expected_cells)

def test_problem_with_full_blocks_with_consecutive_rule_element_indexes_fills_out_of_range_gap_with_empty():
    problem = Problem(Rule([2,2]), cells = [Cell(CellState.undefined)] +[Cell(CellState.full,rule_element_index=0)] +  [Cell(CellState.undefined)]* 3 +[Cell(CellState.full,rule_element_index=1)] + [Cell(CellState.undefined)])
    solved_problem = problem.fill_empty_between_indentified_blocks_solve()
    print(solved_problem)
    expected_cells = [Cell(CellState.undefined)] +[Cell(CellState.full,rule_element_index=0)] +  [Cell(CellState.undefined)] + [Cell(CellState.empty)] + [Cell(CellState.undefined)]  +[Cell(CellState.full,rule_element_index=1)] + [Cell(CellState.undefined)]
    assert(solved_problem.cells == expected_cells)

def test_solving_of_already_solved_problem_returns_problem():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full,rule_element_index=0)])
    problem.solve()
    assert(problem.cells == [Cell(CellState.full,rule_element_index=0)])

def test_solving_of_already_solved_problem_without_rule_element_indexes_returns_indexes():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.full,rule_element_index=0)])

def test_solving_fully_defined_problem_solves_problem_and_updates_rule_element_indexes():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.full,rule_element_index=0)])

def test_solving_all_full_placed_fills_empty_and_updates_rule_element_index():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.empty),Cell(CellState.full,rule_element_index=0),Cell(CellState.empty)])

def test_simple_solving_overlapping_solve_updates_cells_and_rule_element_indexes():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.undefined),Cell(CellState.full,rule_element_index=0),Cell(CellState.undefined)])

def test_complex_solving_overlapping_solve_updates_cells_and_rule_element_indexes():
    problem = Problem(rule = Rule([2,2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.undefined),Cell(CellState.full,rule_element_index=0),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full,rule_element_index=1),Cell(CellState.undefined)])

def test_complete_full_blocks_with_max_rule_size_solve():
    problem = Problem(rule = Rule([1,2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.full),Cell(CellState.undefined)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1),Cell(CellState.full,rule_element_index=1),Cell(CellState.empty)])

def test_complete_full_blocks_with_max_rule_size_solve_with_several_max_size_blocks():
    problem = Problem(rule = Rule([1,2,2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.full),Cell(CellState.undefined)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1),Cell(CellState.full,rule_element_index=1),Cell(CellState.empty),Cell(CellState.full,rule_element_index=2),Cell(CellState.full,rule_element_index=2),Cell(CellState.empty)])

def test_fill_extremities_with_empty_when_both_extremities_can_be_filled():
    problem = Problem(rule = Rule([2,2]), cells = [Cell(CellState.empty), Cell(CellState.undefined), Cell(CellState.empty)] + 7 * [Cell(CellState.undefined)] + [Cell(CellState.empty), Cell(CellState.undefined), Cell(CellState.empty)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.empty), Cell(CellState.empty), Cell(CellState.empty)] + 7 * [Cell(CellState.undefined)] + [Cell(CellState.empty), Cell(CellState.empty), Cell(CellState.empty)])

def test_fill_extremities_with_empty_when_first_full_element_has_rule_element_index_0_fills_everything_before():
    problem = Problem(rule = Rule([2,2]), cells = [Cell(CellState.undefined), Cell(CellState.undefined), Cell(CellState.empty),Cell(CellState.full,rule_element_index=0),Cell(CellState.full,rule_element_index=0),Cell(CellState.empty)] + 4*[Cell(CellState.undefined)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.empty), Cell(CellState.empty), Cell(CellState.empty),Cell(CellState.full,rule_element_index=0),Cell(CellState.full,rule_element_index=0),Cell(CellState.empty)] + 4*[Cell(CellState.undefined)])

def test_fill_extremities_problem_updates_problem_and_rule_element_indexes():
    problem = Problem(rule = Rule([3,1]), cells = [Cell(CellState.undefined), Cell(CellState.full), Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.undefined), Cell(CellState.full,rule_element_index=0), Cell(CellState.full,rule_element_index=0),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])

def test_solving_empty_rule_problem():
    problem = Problem(rule = Rule([]), cells = [Cell(CellState.undefined)])
    solved_problem = problem.solve()
    assert(solved_problem.cells == [Cell(CellState.empty)])

def test_adding_one_empty_problem_len_1_and_problem_rule_1_len_1_creates_problem_with_added_cells_lists_and_rules():
    problem1 = Problem(rule = Rule([]), cells = [Cell(CellState.empty)])
    problem2 = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])
    mixed_problem = problem1 + problem2
    expected_mixed_problem = Problem(rule = Rule([1]), cells = [Cell(CellState.empty),Cell(CellState.undefined)])
    assert(mixed_problem == expected_mixed_problem)

def test_adding_problem_len_2_rule_1_with_undefined_empty_and_problem_len_1_rule_1_returns_problem_len_3_rule_1_1():
    problem1 = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined), Cell(CellState.empty)])
    problem2 = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])
    mixed_problem = problem1 + problem2
    expected_mixed_problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.undefined), Cell(CellState.empty),Cell(CellState.undefined)])
    assert(mixed_problem == expected_mixed_problem)

def test_adding_2_problems_len_1_rule_1_with_full_raises():
    problem1 = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    problem2 = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    with pytest.raises(ProblemAddError) as err:
        mixed_problem = problem1 + problem2
    
def test_adding_2_problems_without_empty_at_junction_raises():
    problem1 = Problem(rule = Rule([1]), cells = [Cell(CellState.full), Cell(CellState.undefined)])
    problem2 = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    with pytest.raises(ProblemAddError) as err:
        mixed_problem = problem1 + problem2

def test_adding_problem_len_2_rule_1_with_undefined_empty_and_problem_len_1_rule_1_with_rule_element_index_increment_rule_element_index():
    problem1 = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined), Cell(CellState.empty)])
    problem2 = Problem(rule = Rule([1]), cells = [Cell(CellState.full,rule_element_index=0)])
    mixed_problem = problem1 + problem2
    expected_mixed_problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.undefined), Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)])
    assert(mixed_problem == expected_mixed_problem)

def test_split_if_possible_with_complete_full_block_with_known_rule_element_index_returns_2_problems():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full,rule_element_index=0), Cell(CellState.empty),Cell(CellState.undefined)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([1]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.empty)]),\
        Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_if_possible_with_empty_at_start_returns_first_part_without_rule_element_and_empty_cell():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.empty), Cell(CellState.undefined)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([]), cells = [Cell(CellState.empty)]),\
        Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_if_possible_with_empty_at_end_returns_first_part_with_rule_and_undefined_cell_and_second_part_no_rule_and_empty_cell():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined), Cell(CellState.empty)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)]),\
        Problem(rule = Rule([]), cells = [Cell(CellState.empty)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_if_possible_for_only_full_cells_returns_original_problem():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [problem]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_if_possible_for_only_empty_cells_returns_original_problem():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.empty),Cell(CellState.empty)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [problem]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_if_possible_for_only_undefined_cells_returns_original_problem():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [problem]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_if_possible_problem_with_empty_at_start():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.empty),Cell(CellState.empty),Cell(CellState.undefined)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([]), cells = [Cell(CellState.empty),Cell(CellState.empty)]),\
        Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_if_possible_with_partially_identified_block_if_empty_before_at_distance_making_impossible_put_other_rule():
    problem = Problem(Rule([1,3]), cells = [Cell(CellState.undefined)] + [Cell(CellState.empty)] + [Cell(CellState.undefined)] + [Cell(CellState.full,rule_element_index=1)] + [Cell(CellState.undefined)] * 2)
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)]),\
        Problem(rule = Rule([3]), cells = [Cell(CellState.empty)] + [Cell(CellState.undefined)] + [Cell(CellState.full,rule_element_index=0)] + [Cell(CellState.undefined)] * 2)]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_if_possible_with_partially_identified_block_if_empty_after_at_distance_making_impossible_put_other_rule():
    problem = Problem(Rule([3,1]), cells = [Cell(CellState.undefined)] * 2 + [Cell(CellState.full,rule_element_index=0)] + [Cell(CellState.undefined)] + [Cell(CellState.empty)] + [Cell(CellState.undefined)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([3]), cells = [Cell(CellState.undefined)] * 2 + [Cell(CellState.full,rule_element_index=0)] + [Cell(CellState.undefined)]),\
        Problem(rule = Rule([1]), cells = [Cell(CellState.empty)] + [Cell(CellState.undefined)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_partially_identified_block_if_empty_at_start_extremity_splits():
    problem = Problem(Rule([3]), cells = [Cell(CellState.undefined)] + [Cell(CellState.empty)] + [Cell(CellState.full,rule_element_index=0)] + [Cell(CellState.undefined)] * 2)
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([]), cells = [Cell(CellState.undefined)]),\
        Problem(rule = Rule([3]), cells = [Cell(CellState.empty)] + [Cell(CellState.full,rule_element_index=0)] + [Cell(CellState.undefined)] * 2)]
    assert(splitted_problem_list == expected_splitted_list)

def test_partially_identified_block_if_empty_at_end_extremity_splits():
    problem = Problem(Rule([3]), cells = [Cell(CellState.undefined)] * 2 + [Cell(CellState.full,rule_element_index=0)] + [Cell(CellState.empty)] + [Cell(CellState.undefined)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([3]), cells = [Cell(CellState.undefined)] * 2 + [Cell(CellState.full,rule_element_index=0)]),\
        Problem(rule = Rule([]), cells =  [Cell(CellState.empty)] + [Cell(CellState.undefined)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_problem_with_empty_at_start():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.empty),Cell(CellState.empty),Cell(CellState.undefined)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([]), cells = [Cell(CellState.empty),Cell(CellState.empty)]),\
        Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_problem_with_empty_at_end():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.empty)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([1]), cells = [Cell(CellState.undefined)]),\
        Problem(rule = Rule([]), cells = [Cell(CellState.empty),Cell(CellState.empty)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_splitting_problem_with_first_complete_full_block_starting_at_index_0():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.empty),Cell(CellState.undefined)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([1]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.empty)]),\
        Problem(rule = Rule([]), cells = [Cell(CellState.undefined)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_splitting_problem_with_first_complete_full_block_starting_at_index_max():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.full,rule_element_index=0)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([]), cells = [Cell(CellState.undefined)]),\
        Problem(rule = Rule([1]), cells = [Cell(CellState.empty),Cell(CellState.full,rule_element_index=0)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_problem_with_rule_element_indexes_splitted_reduces_the_rule_element_index_of_second_part_by_rule_len_of_first_part():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([1]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.empty)]),\
        Problem(rule = Rule([1]), cells = [Cell(CellState.full, rule_element_index=0)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_split_problem_where_first_indexed_complete_full_block_is_in_the_middle_of_problem():
    problem = Problem(rule = Rule([1,1,1]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.undefined), Cell(CellState.empty),Cell(CellState.full,rule_element_index=1),Cell(CellState.empty),Cell(CellState.full,rule_element_index=2)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([1]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.undefined)]),\
        Problem(rule = Rule([1,1]), cells = [Cell(CellState.empty),Cell(CellState.full,rule_element_index=0),Cell(CellState.empty),Cell(CellState.full,rule_element_index=1)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_lol():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.empty), Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])
    splitted_problem_list = problem.split_if_possible()
    expected_splitted_list = [Problem(rule = Rule([]), cells = [Cell(CellState.empty)]),\
        Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])]
    assert(splitted_problem_list == expected_splitted_list)

def test_solve_problem_with_split_needed_with_overlapping_in_second_part():
    problem = Problem(rule = Rule([1,2]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.empty), Cell(CellState.empty), Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])
    solved_problem = problem.solve()
    expected_problem = Problem(rule = Rule([1,2]), cells = [Cell(CellState.full,rule_element_index=0), Cell(CellState.empty), Cell(CellState.empty), Cell(CellState.undefined),Cell(CellState.full,rule_element_index=1),Cell(CellState.undefined)])
    assert(solved_problem == expected_problem)

def test_solve_problem_with_split_needed_with_extremity_full_completing_in_second_part():
    problem = Problem(rule = Rule([2,3]), cells = [Cell(CellState.empty),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])
    solved_problem = problem.solve()
    expected_problem = Problem(rule = Rule([2,3]), cells = [Cell(CellState.empty),Cell(CellState.full,rule_element_index=0),Cell(CellState.full,rule_element_index=0),Cell(CellState.empty),Cell(CellState.undefined),Cell(CellState.full, rule_element_index=1),Cell(CellState.full,rule_element_index=1),Cell(CellState.undefined),Cell(CellState.empty)])
    assert(solved_problem == expected_problem)

def test_solve_with_full_block_size_equal_max_rule_size():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.undefined)] * 2 + [Cell(CellState.full)] + [Cell(CellState.undefined)]*17)
    solved_problem = problem.solve()

def test_solve_inplace_problem_with_split_needed_with_extremity_full_completing_in_second_part():
    problem = Problem(rule = Rule([2,3]), cells = [Cell(CellState.empty),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.empty),Cell(CellState.undefined),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])
    problem.solve_inplace()
    expected_problem = Problem(rule = Rule([2,3]), cells = [Cell(CellState.empty),Cell(CellState.full,rule_element_index=0),Cell(CellState.full,rule_element_index=0),Cell(CellState.empty),Cell(CellState.undefined),Cell(CellState.full, rule_element_index=1),Cell(CellState.full,rule_element_index=1),Cell(CellState.undefined),Cell(CellState.empty)])
    assert(problem == expected_problem)

def test_problem_dict_from_list_of_problems():
    problem = Problem(rule = Rule([1]), cells = [Cell()])
    problem1 = Problem(rule = Rule([1]), cells = [Cell()])
    problem_dict = ProblemDict(problems = [problem, problem1])

def test_problem_dict_attribute_is_dict():
    problem = Problem(rule = Rule([1]), cells = [Cell()])
    problem1 = Problem(rule = Rule([1]), cells = [Cell()])
    problem_dict = ProblemDict(problems = [problem, problem1])
    assert(isinstance(problem_dict.problems,dict))

def test_problem_dict_raises_when_all_problems_not_the_same_size():
    problem = Problem(rule = Rule([1]), cells = [Cell()])
    problem1 = Problem(rule = Rule([1]), cells = [Cell(),Cell()])
    with pytest.raises(InvalidProblemDict) as err:
        problem_dict = ProblemDict(problems = [problem, problem1])

def test_problem_dict_with_only_rule_list_raises():
    rule_list = RuleList([[1],[1]])
    with pytest.raises(InvalidProblemDict) as err:
        problem_dict = ProblemDict(rule_list = rule_list)

def test_generate_problems_from_rules_and_problem_size():
    rule_list = RuleList([[1],[1]])
    problem_size = 1
    problem_dict = ProblemDict(rule_list = rule_list, problem_size = problem_size)
    expected_problem_dict = {0 : Problem(rule = Rule([1]),cells=[Cell()]), 1 : Problem(rule = Rule([1]),cells=[Cell()])}
    assert(problem_dict.problems == expected_problem_dict)

def test_problem_dict_getitem_returns_problem_at_index_i():
    problem = Problem(rule = Rule([1]), cells = [Cell()])
    problem1 = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    problem_dict = ProblemDict(problems = [problem, problem1])
    problem_from_problem_dict = problem_dict[1]
    assert(problem_from_problem_dict == problem1)

def test_assigning_problem_at_index_i():
    problem = Problem(rule = Rule([1]), cells = [Cell()])
    problem1 = Problem(rule = Rule([1]), cells = [Cell()])
    problem_dict = ProblemDict(problems = [problem, problem1])
    problem_to_assign = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    problem_dict[0] = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    assert(problem_dict[0] == problem_to_assign)

def test_assigning_incompatible_problem_raises():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    problem1 = Problem(rule = Rule([1]), cells = [Cell()])
    problem_dict = ProblemDict(problems = [problem, problem1])
    problem_to_assign = Problem(rule = Rule([1]), cells = [Cell()])
    with pytest.raises(InvalidProblemDictAssignment) as err:
        problem_dict[0] = problem_to_assign

def test_assigning_problem_with_different_rule_raise():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    problem1 = Problem(rule = Rule([1]), cells = [Cell()])
    problem_dict = ProblemDict(problems = [problem, problem1])
    problem_to_assign = Problem(rule = Rule([]), cells = [Cell(CellState.full)])
    with pytest.raises(InvalidProblemDictAssignment) as err:
        problem_dict[0] = problem_to_assign

def test_get_problem_dict_len():
    problem = Problem(rule = Rule([1]), cells = [Cell()])
    problem1 = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    problem_dict = ProblemDict(problems = [problem, problem1])
    assert(len(problem_dict) == 2)