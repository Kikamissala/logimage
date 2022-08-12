from logimage.main import Cell, CellState, FullBlock, RuleList, Grid, \
 InvalidGridSet,InvalidCellStateModification, Logimage, InvalidRule,RuleElement, Rule, RuleSet, Problem,\
    Solution,FullBlock, InvalidProblem, CellUpdateError
import numpy as np
import pytest
import pandas as pd

def test_cell_with_same_same_state_are_equal():
    cell = Cell(cell_state=CellState.empty)
    other_cell = Cell(cell_state=CellState.empty)
    assert(cell == other_cell)

def test_cell_with_same_coordinates_and_different_state_are_different():
    cell = Cell(cell_state=CellState.empty)
    other_cell = Cell(cell_state=CellState.undefined)
    assert(cell != other_cell)

def test_modify_cell_state_to_full_from_not_undefined_raises():
    cell = Cell(cell_state=CellState.empty)
    with pytest.raises(InvalidCellStateModification) as err:
        cell.full()

def test_modify_cell_state_to_empty_from_not_undefined_raises():
    cell = Cell(cell_state=CellState.full)
    with pytest.raises(InvalidCellStateModification) as err:
        cell.empty()

def test_modify_cell_state_from_not_undefined_raises():
    cell = Cell(cell_state=CellState.full)
    with pytest.raises(InvalidCellStateModification) as err:
        cell.full()

def test_modify_cell_state_from_undefined_to_full():
    cell = Cell(cell_state=CellState.undefined)
    cell.full()
    assert(cell == Cell(CellState.full))

def test_update_cell_state_from_undefined_to_empty():
    cell = Cell(cell_state=CellState.undefined)
    cell.update_state(CellState.empty)
    assert(cell == Cell(CellState.empty))

def test_update_cell_state_from_undefined_to_full():
    cell = Cell(cell_state=CellState.undefined)
    cell.update_state(CellState.full)
    assert(cell == Cell(CellState.full))

def test_update_cell_state_with_not_cell_state_value_raises():
    cell = Cell(cell_state=CellState.undefined)
    with pytest.raises(InvalidCellStateModification) as err:
        cell.update_state(1)

def test_update_cell_state_with_non_undefined_cell_raises():
    cell = Cell(cell_state=CellState.full)
    with pytest.raises(InvalidCellStateModification) as err:
        cell.update_state(CellState.empty)

def test_numerize_undefined_cell_returns_none():
    cell = Cell()
    numerized_cell = cell.numerize()
    assert(numerized_cell == -1)

def test_numerize_empty_cell_returns_0():
    cell = Cell(CellState.empty)
    numerized_cell = cell.numerize()
    assert(numerized_cell == 0)

def test_numerize_full_cell_returns_one():
    cell = Cell(CellState.full)
    numerized_cell = cell.numerize()
    assert(numerized_cell == 1)

def test_update_cell_rule_element_index_changes_value():
    cell = Cell(CellState.full)
    cell.set_rule_element_index(0)
    assert(cell.rule_element_index == 0)

def test_update_cell_rule_element_for_non_full_cell_raises():
    cell = Cell(CellState.undefined)
    with pytest.raises(CellUpdateError) as err:
        cell.set_rule_element_index(0)

def test_grid_creation_with_one_row_and_one_column_returns_list_of_list_of_one_cell():
    grid = Grid(row_number = 1, column_number = 1)
    assert(grid.cells == np.array([[Cell()]]))

def test_grid_creation_with_one_row_and_two_column_returns_list_of_list_of_one_list_of_two_cells_with_good_coordinates():
    grid = Grid(row_number = 1, column_number = 2)
    assert(np.array_equal(grid.cells, np.array([[Cell(), Cell()]])))

def test_grid_creation_with_two_row_and_one_column_return_list_of_lists_of_two_lists_with_one_cell_each_with_good_coordinates():
    grid = Grid(row_number = 2, column_number = 1)
    assert(np.array_equal(grid.cells, np.array([[Cell()], [Cell()]])))

def test_grid_creation_with_two_rows_and_two_columns_return_right_list_of_lists_with_good_coordinates():
    grid = Grid(row_number = 2, column_number = 2)
    assert(np.array_equal(grid.cells, np.array([[Cell(), Cell()], [Cell(), Cell()]])))

def test_get_item_in_grid_returns_right_cell():
    grid = Grid(row_number = 2, column_number = 2)
    selected_cell = grid[0,0]
    assert(selected_cell == Cell())

def test_get_row_in_grid_returns_right_row():
    grid = Grid(row_number = 2, column_number = 2)
    selected_cell = grid[0,:]
    assert(np.array_equal(selected_cell, np.array([Cell(),Cell()])))

def test_set_cell_in_grid_updates_grid():
    grid = Grid(row_number = 2, column_number = 2)
    grid[0,0] = Cell(CellState.empty)
    assert(grid[0,0] == Cell(CellState.empty))

def test_set_row_values_in_grid_updates_grid():
    grid = Grid(row_number = 2, column_number = 2)
    grid[0,:] = np.array([Cell(CellState.empty),Cell(CellState.empty)])
    assert(np.array_equal(grid[0,:], np.array([Cell(CellState.empty),Cell(CellState.empty)])))

def test_set_value_in_grid_not_cell_returns_error():
    grid = Grid(row_number = 2, column_number = 2)
    with pytest.raises(InvalidGridSet) as err:
        grid[0,0] = 8

def test_set_row_in_grid_not_cell_returns_error():
    grid = Grid(row_number = 2, column_number = 2)
    with pytest.raises(InvalidGridSet) as err:
        grid[0,:] = np.array([8,8])

def test_set_cell_at_coordinates_to_empty():
    grid = Grid(row_number = 2, column_number = 2)
    grid.empty(0,0)
    assert(grid[0,0] == Cell(cell_state=CellState.empty))

def test_set_cell_at_coordinates_to_full():
    grid = Grid(row_number = 2, column_number = 2)
    grid.full(0,0)
    assert(grid[0,0] == Cell(cell_state=CellState.full))

def test_emptying_not_undefined_cell_in_grid_raises():
    grid = Grid(row_number = 2, column_number = 2)
    grid.full(0,0)
    with pytest.raises(InvalidCellStateModification) as err:
        grid.empty(0,0)

def test_translate_rule_element_of_element_1_returns_list_of_len_1_with_1():
    rule_element = RuleElement(1)
    assert(rule_element.translate_to_list() == [Cell(CellState.full)])

def test_translate_rule_element_of_element_2_returns_list_of_len_2_with_1():
    rule_element = RuleElement(2)
    assert(rule_element.translate_to_list() == [Cell(CellState.full),Cell(CellState.full)])

def test_check_if_elements_of_rule_are_rule_elements():
    rule = Rule([1,1])
    list_of_bool = [isinstance(element,RuleElement) for element in rule]
    assert(all(list_of_bool) == True)

def test_compute_rule_minimum_possible_line_len():
    rule = Rule([1,1])
    minimum_possible_line_len = rule.compute_min_possible_len()
    assert(minimum_possible_line_len == 3)

def test_compute_rule_minimum_possible_line_len_for_unique_rule_element():
    rule = Rule([2])
    minimum_possible_line_len = rule.compute_min_possible_len()
    assert(minimum_possible_line_len == 2)

def test_check_if_elements_of_rulelist_are_rules():
    rule_list = RuleList([[1,1],[1,1]])
    list_of_bool = [isinstance(element,Rule) for element in rule_list]
    assert(all(list_of_bool) == True)

def test_compute_maximum_minimum_possible_len_of_rule_list():
    rule_list = RuleList([[1,1],[1,1,1]])
    maximum_minimum_possible_len = rule_list.compute_maximum_minimum_possible_len()
    assert(maximum_minimum_possible_len == 5)

def test_rules_creation_with_row_rules_and_column_rules():
    row_rules = [[1],[1]]
    column_rules = [[1],[1]]
    rules = RuleSet(row_rules = row_rules, column_rules = column_rules)
    assert((rules.row_rules == row_rules) & (rules.column_rules == column_rules))

def test_generate_logimage_needs_grid_dimensions_and_list_of_rules_for_rows_and_columns():
    logimage = Logimage(grid_dimensions = (2,2),rules = RuleSet(row_rules = [[1],[1]],column_rules = [[1],[1]]))

def test_logimage_init_raises_when_a_rule_values_exceeds_size_of_grid():
    with pytest.raises(InvalidRule) as err:
        logimage = Logimage(grid_dimensions = (2,2),rules = RuleSet(row_rules = [[1,1],[1]],column_rules = [[1],[1]]))

def test_logimage_init_raises_when_number_of_row_rules_not_equal_number_of_rows():
    with pytest.raises(InvalidRule) as err:
        logimage = Logimage(grid_dimensions = (2,2),rules = RuleSet(row_rules = [[1],[1],[1]],column_rules = [[1],[1]]))

@pytest.mark.skip()
def test_extracting_problem_from_logimage():
    assert(True == False)

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

def test_is_problem_solved_returns_true_when_no_cell_is_undefined():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full), Cell(CellState.full), Cell(CellState.full)])
    is_problem_solved = problem.is_solved()
    assert(is_problem_solved == True)

def test_problem_with_fully_defined_line_by_rule_returns_true_when_function_called():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell()])
    bool_is_line_fully_defined_by_rule = problem.is_line_fully_defined_by_rule()
    assert(bool_is_line_fully_defined_by_rule == True)

def test_updating_cells_list_with_new_states_changes_cells_list():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell()])
    new_cells_list = [Cell(CellState.full), Cell(), Cell()]
    problem.update_cells_list(new_cells_list)
    assert(problem.cells == new_cells_list)

def test_updating_cells_list_with_new_rule_element_index_sets_new_rule_element_index():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full), Cell(), Cell()])
    new_cells_list = [Cell(CellState.full,rule_element_index=0), Cell(), Cell()]
    problem.update_cells_list(new_cells_list)
    assert(problem.cells == new_cells_list)

def test_updating_cells_list_with_new_cell_none_keeps_previous_cell():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(CellState.full), Cell(), Cell()])
    new_cells_list = [Cell(), Cell(), Cell()]
    problem.update_cells_list(new_cells_list)
    expected_cells_list = [Cell(CellState.full), Cell(), Cell()]
    assert(problem.cells == expected_cells_list)

def test_solving_complete_problem_from_all_undefined_returns_fully_defined_list():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell()])
    problem.fully_defined_solve()
    assert((problem.is_solved() == True) & (problem.numerize_cell_list() == [1,0,1]))

def test_solving_complete_problem_with_len_4_and_rule_2_1_returns_numerized_list_with_1_1_0_1():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(), Cell(), Cell(), Cell()])
    problem.fully_defined_solve()
    assert((problem.is_solved() == True) & (problem.numerize_cell_list() == [1,1,0,1]))

def test_compute_number_of_freedom_degrees_of_fully_defined_rule_is_zero():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell()])
    nb_freedom_degrees = problem.compute_number_of_freedom_degrees()
    assert(nb_freedom_degrees == 0)

def test_compute_number_of_freedom_degrees_of_rule_with_1_1_and_len_4_is_one():
    problem = Problem(rule = Rule([1,1]), cells = [Cell(), Cell(), Cell(), Cell()])
    nb_freedom_degrees = problem.compute_number_of_freedom_degrees()
    assert(nb_freedom_degrees == 1)

# L'idée est d'avoir pour un élément de règle une position, ou alors pour une règle une association d'élement de règle et 
# une position.
# Il faudrait pouvoir dans une situation donnée, évaluer la solution partielle. 
# Il faut pouvoir identifier les parties de règles identifiables. 
# Exemple si on a plusieurs cases noires posées d'affilée tout à gauche (ou tout à droite) et une case blanche ensuite, alors
# il faut identifier la partie de la règle déjà en place.
# De même si on identifie un pattern complet (suite de cases noire entourées de 2 vides)
# et que ce pattern a une taille qui correspond à un élément unique dans la consigne, alors on peut évaluer sa position
# Si on a un pattern en bout de ligne avec une vide, c'est soit le premier, soit le dernier élément de la consigne.
# De plus si la partie de la règle identifiée ne correspond pas à la consigne, on spécifie qu'il y a une erreur sur la ligne
# Cela va aussi servir pour la règle de l'overlap. Puisqu'on va essayer de placer l'ensemble des cases le plus à gauche possible
# et ensuite identifier si pour la même partie de règle (même position dans la rule) on a des cases remplies en commun.
# Il faut donc un objet qui donne pour chaque case posée dans une ligne l'élément de règle identifié (si possible).
# On pourra aussi définir un guess. Une proposition de disposition de cases qui satisfait la consigne
# Il faut aussi définir un moyen de résoudre un sous problème (partie du problème qui est lui même un problème)
# l'étude de l'overlap peut se faire indépendamment des cases déjà en place
# Il faut aussi mettre en place un système qui met à jour les cases de la ligne, en fonction d'une heuristique qui
# aurait déterminé des cases à poser. Toute case déjà définie ne peut être mise à jour. Si la solution trouvée
# par l'heuristique entre en contradiction avec les cases posées préalablement, il doit y avoir une erreur.
# Il faut être capable de déterminer qu'on peut définir un sous problème (si une partie du problème est résolu, en bout de ligne,
# ou alors qu'une partie du problème est totalement identifiée)
# il faut être capable de définir ces sous problèmes, et de tenter des heuristiques dessus indépendemment du problème global, 
# puis remonter les résultats éventuels des recherches dans le problème global, et ce de manière récursive. (il peut y avoir
# un sous problème de sous problème)

def test_first_index_of_value_for_list_with_none_value():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    numerized_series = pd.Series([1])
    first_non_none_index = problem.find_first_index_of_value(numerized_series,None)
    assert(first_non_none_index is None)

def test_find_sequences_without_zeroes_in_empty_series_returns_empty_list():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    numerized_series = pd.Series()
    list_of_not_zero_series = problem.find_series_without_value_zero(numerized_series)
    expected_list_of_series = []
    assert(list_of_not_zero_series == expected_list_of_series)

def test_find_sequences_without_zeroes_in_only_zero_series_returns_empty_list():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    numerized_series = pd.Series([0])
    list_of_not_zero_series = problem.find_series_without_value_zero(numerized_series)
    expected_list_of_series = []
    assert(list_of_not_zero_series == expected_list_of_series)

def test_find_sequences_without_zeroes_in_simple_series():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    numerized_series = pd.Series([1,1])
    list_of_not_zero_series = problem.find_series_without_value_zero(numerized_series)
    expected_list_of_series = [pd.Series([1,1])]
    assert(all([list_of_not_zero_series[i].equals(expected_list_of_series[i]) for i in range(0,len(list_of_not_zero_series))]))

def test_find_sequences_without_zeroes_in_series_with_one_zero_at_the_end():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([1,1,0])
    list_of_not_zero_series = problem.find_series_without_value_zero(numerized_series)
    expected_list_of_series = [pd.Series([1,1])]
    pd.testing.assert_series_equal(list_of_not_zero_series[0], expected_list_of_series[0])

def test_find_sequences_without_zeroes_in_series_with_one_zero_at_the_beginning():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([0,1,1])
    list_of_not_zero_series = problem.find_series_without_value_zero(numerized_series)
    expected_list_of_series = [pd.Series([1,1],index=[1,2])]
    pd.testing.assert_series_equal(list_of_not_zero_series[0], expected_list_of_series[0])

def test_find_sequences_without_zeroes_in_series_with_one_zero_at_the_beginning_and_end():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([0,1,1,0])
    list_of_not_zero_series = problem.find_series_without_value_zero(numerized_series)
    expected_list_of_series = [pd.Series([1,1],index=[1,2])]
    pd.testing.assert_series_equal(list_of_not_zero_series[0], expected_list_of_series[0])

def test_find_sequences_without_zeroes_in_series_with_two_non_zero_sequences():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([1,1,0,1])
    list_of_not_zero_series = problem.find_series_without_value_zero(numerized_series)
    expected_list_of_series = [pd.Series([1,1],index=[0,1]),pd.Series([1],index=[3])]
    assert(all([list_of_not_zero_series[i].equals(expected_list_of_series[i]) for i in range(0,len(list_of_not_zero_series))]))

def test_find_sequences_without_zeroes_in_serie_with_none_values():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([1,1,0,-1]).astype("int64",errors ="ignore")
    list_of_not_zero_series = problem.find_series_without_value_zero(numerized_series)
    expected_list_of_series = [pd.Series([1,1],index=[0,1]),pd.Series([-1],index=[3])]
    assert(all([list_of_not_zero_series[i].equals(expected_list_of_series[i]) for i in range(0,len(list_of_not_zero_series))]))

def test_find_sequences_without_zeroes_in_serie_with_none_value():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    numerized_series = pd.Series([1,1,0,-1])
    list_of_not_zero_series = problem.find_series_without_value_zero(numerized_series)
    expected_list_of_series = [pd.Series([1,1]),pd.Series([-1],index = [3])]
    assert(all([list_of_not_zero_series[i].equals(expected_list_of_series[i]) for i in range(0,len(list_of_not_zero_series))]))

def test_identify_full_rules_from_problem_cells_with_one_full_cell():
    problem = Problem(rule = Rule([1]), cells = [Cell(CellState.full)])
    identified_rules = problem.identify_full_blocks()
    assert(identified_rules == [FullBlock(block_len=1,initial_index=0)])

def test_identify_full_rules_from_problem_cells_with_two_full_cell():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full)])
    identified_rules = problem.identify_full_blocks()
    assert(identified_rules == [FullBlock(block_len = 2, initial_index = 0)])

def test_identify_full_rules_from_problem_cells_with_two_full_cells_and_one_empty_cell():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty)])
    identified_rules = problem.identify_full_blocks()
    assert(identified_rules == [FullBlock(block_len = 2, initial_index = 0)])

def test_identify_full_rules_from_problem_cells_with_rules_2_1_and_2_full_cells_one_empty_one_full():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.full)])
    identified_blocks = problem.identify_full_blocks()
    assert(identified_blocks == [FullBlock(block_len = 2, initial_index = 0),FullBlock(block_len = 1, initial_index = 3)])

def test_identify_full_rules_from_problem_cells_with_rules_2_1_and_2_full_cells_one_empty_one_undefined():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.empty),Cell(CellState.undefined)])
    identified_blocks = problem.identify_full_blocks()
    assert(identified_blocks == [FullBlock(block_len = 2, initial_index = 0)])

def test_identify_full_rules_from_problem_cells_with_rules_2_1_and_2_full_cells_one_undefined_returns_empty():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full),Cell(CellState.full),Cell(CellState.undefined),Cell(CellState.undefined)])
    identified_blocks = problem.identify_full_blocks()
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
    print(problem.last_undefined_cell_index())
    assert((identified_rule_element_indexes == [None,None,None,1,1]))

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
    problem.overlapping_solve()
    expected_cells_list = [Cell(),Cell(CellState.full, rule_element_index=0),Cell()]
    assert(problem.cells == expected_cells_list)

def test_problem_subject_to_overlap_solving_for_rule_2_1_and_len_5_has_second_element_full_and_rule_element_index_0():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(),Cell(),Cell(),Cell(),Cell()])
    problem.overlapping_solve()
    expected_cells_list = [Cell(),Cell(CellState.full, rule_element_index=0),Cell(),Cell(),Cell()]
    assert(problem.cells == expected_cells_list)

def test_problem_subject_to_overlap_solving_for_rule_1_2_and_len_5_has_second_to_last_element_full_and_rule_element_index_0():
    problem = Problem(rule = Rule([1,2]), cells = [Cell(),Cell(),Cell(),Cell(),Cell()])
    problem.overlapping_solve()
    expected_cells_list = [Cell(),Cell(),Cell(),Cell(CellState.full, rule_element_index=1),Cell()]
    assert(problem.cells == expected_cells_list)

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
    problem.all_full_cell_found_solve()
    expected_cells_list = [Cell(CellState.full), Cell(CellState.full), Cell(CellState.empty),Cell(CellState.empty)]
    assert(problem.cells == expected_cells_list)

def test_all_full_cell_found_solve_fills_undefined_cells_in_2_1_rule_problem():
    problem = Problem(rule = Rule([2,1]), cells = [Cell(CellState.full), Cell(CellState.full), Cell(),Cell(CellState.full)])
    problem.all_full_cell_found_solve()
    expected_cells_list = [Cell(CellState.full), Cell(CellState.full), Cell(CellState.empty),Cell(CellState.full)]
    assert(problem.cells == expected_cells_list)

def test_is_left_extremity_fillable_with_empty_of_problem_with_first_cell_undefined_then_empty_and_first_rule_element_2_is_true():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined), Cell(CellState.empty), Cell(CellState.full),Cell()])
    bool_left_extremity_fillable = problem.is_left_extremity_empty_fillable()
    assert(bool_left_extremity_fillable == True)

def test_is_left_extremity_fillable_with_empty_of_problem_with_first_two_cells_undefined_then_empty_and_rule_element_3_is_true():
    problem = Problem(rule = Rule([3]), cells = [Cell(CellState.undefined), Cell(CellState.undefined), Cell(CellState.empty),Cell(),Cell(),Cell()])
    bool_left_extremity_fillable = problem.is_left_extremity_empty_fillable()
    assert(bool_left_extremity_fillable == True)

def test_is_left_extremity_fillable_with_empty_of_problem_with_first_cell_empty_then_undefined_then_empty_and_rule_element_2_is_true():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.empty), Cell(CellState.undefined), Cell(CellState.empty),Cell(),Cell(),Cell()])
    bool_left_extremity_fillable = problem.is_left_extremity_empty_fillable()
    assert(bool_left_extremity_fillable == True)

def test_is_left_extremity_fillable_with_empty_of_problem_with_first_cell_undefined_then_full_and_first_rule_element_2_is_false():
    problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined), Cell(CellState.full), Cell(),Cell()])
    bool_left_extremity_fillable = problem.is_left_extremity_empty_fillable()
    assert(bool_left_extremity_fillable == False)