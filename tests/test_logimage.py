from logimage.cell import Cell, CellState, Grid
from logimage.rule import RuleSet, InvalidRule, RuleList, Rule
from logimage.logimage import InvalidGuessChange, Logimage, LogimageProblems, NoGuessLeft, ProblemCoordinates, Guess, Modification
from logimage.problem import Problem,ProblemDict
import pytest
import numpy as np
import copy

def test_generate_logimage_needs_grid_dimensions_and_list_of_rules_for_rows_and_columns():
    logimage = Logimage(rules = RuleSet(row_rules = [[1],[1]],column_rules = [[1],[1]]))

def test_logimage_init_raises_when_a_rule_values_exceeds_size_of_grid():
    with pytest.raises(InvalidRule) as err:
        logimage = Logimage(rules = RuleSet(row_rules = [[1,1],[1]],column_rules = [[1],[1]]))

def test_create_logimage_problems_from_ruleset_and_grid_dimensions_has_column_problems_and_row_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    assert((isinstance(logimage_problems.row_problems,ProblemDict)) & (isinstance(logimage_problems.column_problems,ProblemDict)) 
        & (len(logimage_problems.row_problems) == 2) & (len(logimage_problems.column_problems) == 2))

def test_getitem_in_logimage_problems_with_index_0_returns_row_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    expected_return_value = ProblemDict(rule_list=RuleList([Rule([1]),Rule([1])]), problem_size=2)
    assert(logimage_problems[0] == expected_return_value)

def test_setitem_changes_problem():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems[0,0] = Problem(Rule([1]),cells = [Cell(CellState.full),Cell()])
    expected_problem = Problem(Rule([1]),cells = [Cell(CellState.full),Cell()])
    assert(logimage_problems[0][0] == expected_problem)

def test_setitem_on_row_impacts_columns_at_changed_indexes():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems[0,0] = Problem(Rule([1]),cells = [Cell(CellState.full),Cell()])
    assert(logimage_problems[1][0][0] == Cell(CellState.full))

def test_setitem_on_column_impacts_rows_at_changed_indexes():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems[1,0] = Problem(Rule([1]),cells = [Cell(CellState.empty),Cell(CellState.full)])
    assert(logimage_problems[0][1][0] == Cell(CellState.full))

def test_setitem_updates_modifications():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems[1,0] = Problem(Rule([1]),cells = [Cell(CellState.full),Cell(CellState.undefined)])
    expected_modifications = [Modification(problem_coordinates=ProblemCoordinates(1,0),cell_index=0,new_cell=Cell(CellState.full))]
    assert(logimage_problems.modifications == expected_modifications)

def test_problem_coordinates_defined_with_dimension_and_problem_index():
    problem_coordinates = ProblemCoordinates(0,0)
    assert((problem_coordinates.dimension == 0) & (problem_coordinates.index == 0))

def test_problem_coordinates_hashing_works():
    problem_coordinates = ProblemCoordinates(0,0)
    other_problem_coordinates = ProblemCoordinates(0,0)
    assert(hash(problem_coordinates) == hash(other_problem_coordinates))

def test_adding_already_existing_index_in_candidate_problems_leaves_it_unchanged():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.add_candidate_problem(0,0)
    logimage_problems.add_candidate_problem(0,0)
    assert(logimage_problems.candidate_problems == {ProblemCoordinates(0,0)})

def test_removing_candidate_from_candidate_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.add_candidate_problem(0,0)
    logimage_problems.remove_candidate_problem(0,0)
    assert(logimage_problems.candidate_problems == set())

def test_solve_problem_function_updates_problem_and_transposes_cell_updates():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.solve_problem(dimension = 0, index = 0)
    assert((logimage_problems[0][0][1].cell_state == CellState.full) & (logimage_problems[1][1][0].cell_state == CellState.full))

def test_solve_problem_function_updates_problem_and_transposes_cell_updates_but_doesnt_set_rule_element_index():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.solve_problem(dimension = 0, index = 0)
    assert((logimage_problems[0][0][1].rule_element_index == 0) & (logimage_problems[1][1][0].rule_element_index == None))


def test_solve_problem_function_updates_candidate_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.solve_problem(dimension = 0, index = 0)
    assert(logimage_problems.candidate_problems == {ProblemCoordinates(1,1)})

def test_value_from_candidate_problems_can_be_used_in_solve_function():
    pass

def test_select_problem_to_solve_from_updated_problems_selects_problem_with_least_undefined_cells():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.add_candidate_problem(0,0)
    logimage_problems.add_candidate_problem(1,0)
    selected_problem = logimage_problems.select_candidate_problem()
    assert(selected_problem == ProblemCoordinates(1,0))

def test_select_problem_when_a_problem_in_candidate_has_0_undefined_left_removes_problem_from_candidate():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems[0,0] = Problem(Rule([2]),cells = [Cell(CellState.full),Cell(CellState.full)])
    logimage_problems.candidate_problems = {ProblemCoordinates(0,0)}
    logimage_problems.select_candidate_problem()
    assert(logimage_problems.candidate_problems == set())

def test_select_problem_when_candidate_problems_empty_returns_none():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    selected_problem = logimage_problems.select_candidate_problem()
    assert(selected_problem is None)

def test_select_problem_when_a_problem_in_candidate_has_0_undefined_left_removes_problem_from_candidate():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems[0,0] = Problem(Rule([2]),cells = [Cell(CellState.full),Cell(CellState.full)])
    logimage_problems.candidate_problems = {ProblemCoordinates(0,0)}
    selected_problem = logimage_problems.select_candidate_problem()
    assert(selected_problem is None)

def test_select_problem_when_problem_has_1_left_undefined_is_selected_and_following_candidates_are_not_analyzed():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems[0,0] = Problem(Rule([2]),cells = [Cell(CellState.full),Cell(CellState.full)])
    logimage_problems.add_candidate_problem(0,0)
    selected_problem = logimage_problems.select_candidate_problem()
    expected_candidate_problems = {ProblemCoordinates(1,0),ProblemCoordinates(1,1),ProblemCoordinates(0,0)}
    assert((selected_problem == ProblemCoordinates(1,0)) & (logimage_problems.candidate_problems == expected_candidate_problems))

def test_run_solve_on_selected_candidate_removes_selected_candidate_from_candidate_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.add_candidate_problem(0,0)
    logimage_problems.solve_problem(dimension = 0, index = 0)
    assert(logimage_problems.candidate_problems == {ProblemCoordinates(1,1)})

def test_get_candidate_and_solve_problem_solves_and_update_candidate_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.add_candidate_problem(0,0)
    logimage_problems.get_candidate_and_solve_problem()
    assert((logimage_problems[0][0][1] == Cell(CellState.full,rule_element_index=0)) & (logimage_problems.candidate_problems == {ProblemCoordinates(1,1)}))

def test_create_modification_and_apply_it_to_logimage_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    modification = Modification(problem_coordinates = ProblemCoordinates(0,0), cell_index = 0, new_cell = Cell(CellState.full, rule_element_index = 1))
    logimage_problems.modify(modification)
    assert((logimage_problems[0][0][0] == Cell(CellState.full,rule_element_index=1)) & (len(logimage_problems.modifications) == 1))

@pytest.mark.skip()
def test_run_solve_appends_resolution_history_with_modified_state_cells():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.add_candidate_problem(0,0)
    logimage_problems.get_candidate_and_solve_problem()
    assert(True == False)


# On veut créer un objet avec la cellule et ses coordonées. Cet objet servira à alimenter l'historique.
# On veut aussi pouvoir faire un guess, un guess contient l'objet logimage problem bloqué, la cellule
# guess avec ses coordonnées et sa valeur, et le nombre de guess faits. 
# On alimente la liste de guess si besoin et si on a testé toutes les options du guess 1 alors on sort 
# une erreur. Dans le cas où un guess futur a été testé sur toutes ses possibilités, on le supprime et
# on revient au précédent dont on change la valeur et on incrément le nombre de trys 

def test_run_solve_doesnt_update_history_if_change_is_only_rule_element_index():
    pass

def test_scan_problems_for_fully_defined_updates_candidate_problems_with_row_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.scan_fully_defined_problems()
    assert(logimage_problems.candidate_problems == {ProblemCoordinates(0,0)})

def test_scan_problems_for_fully_defined_updates_candidate_problems_with_column_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([2])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.scan_fully_defined_problems()
    assert(logimage_problems.candidate_problems == {ProblemCoordinates(1,1)})

def test_scan_problems_for_overlap_updates_candidate_problems_with_row_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.scan_for_overlap_problems()
    assert(logimage_problems.candidate_problems == {ProblemCoordinates(0,0)})

def test_scan_problems_for_overlap_updates_candidate_problems_with_column_problems():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([2])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.scan_for_overlap_problems()
    assert(logimage_problems.candidate_problems == {ProblemCoordinates(1,1)})

def test_is_solved_for_solved_logimage_problems_returns_true():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([2])]),column_rules=RuleList([Rule([2]),Rule([2])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems[0,0] = Problem(Rule([2]),cells = [Cell(CellState.full),Cell(CellState.full)])
    logimage_problems[0,1] = Problem(Rule([2]),cells = [Cell(CellState.full),Cell(CellState.full)])
    assert(logimage_problems.is_solved() == True)

def test_full_solve_solves_problem_with_only_fully_defined_problems_solves_logimage():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([2])]),column_rules=RuleList([Rule([2]),Rule([2])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.solve()
    assert((logimage_problems.finished == True) & (logimage_problems.solved == True))

def test_full_solve_solves_simple_logimage():
    pass

def test_full_solve_on_impossible_problem_has_finished_true_and_solved_false():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    logimage_problems.solve()
    assert((logimage_problems.finished == True) & (logimage_problems.solved == False))

def test_update_grid_from_problems_uses_row_problems_to_define_grid():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage = Logimage(rules = rule_set)
    logimage.logimage_problems[0,0] = Problem(Rule([1]),cells=[Cell(CellState.full),Cell(CellState.empty)])
    logimage.update_grid_from_problems()
    expected_grid = Grid(row_number=2,column_number=2)
    expected_grid[0,:] = np.array([1,0])
    assert(np.array_equal(expected_grid,logimage.grid))

def test_render_grid_displays_grid():
    pass

def test_reconstruct_resolution_shows_evolution_of_resolution_from_history():
    pass

def test_compute_solving_percentage():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set=rule_set)
    logimage_problems[0,0] = Problem(rule = Rule([1]),cells = [Cell(CellState.full),Cell(CellState.undefined)])
    percentage = logimage_problems.compute_solving_percentage()
    assert(percentage == 25)

def test_get_guess_candidate_finds_problem_coordinates_and_cell_index_when_changed_index_is_first_index():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set=rule_set)
    logimage_problems[0,0] = Problem(rule = Rule([1]),cells = [Cell(CellState.full),Cell(CellState.undefined)])
    problem_coordinates, cell_index = logimage_problems.get_guess_candidate()
    assert((problem_coordinates == ProblemCoordinates(0,0)) & (cell_index == 1))

def test_get_guess_candidate_finds_problem_coordinates_and_cell_index_when_changed_index_is_last_index():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set=rule_set)
    logimage_problems[0,0] = Problem(rule = Rule([1]),cells = [Cell(CellState.undefined),Cell(CellState.full)])
    problem_coordinates, cell_index = logimage_problems.get_guess_candidate()
    assert((problem_coordinates == ProblemCoordinates(0,0)) & (cell_index == 0))

def test_get_guess_candidate_finds_problem_coordinates_and_cell_index_when_last_changed_problem_is_full():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set=rule_set)
    logimage_problems[0,0] = Problem(rule = Rule([1]),cells = [Cell(CellState.undefined),Cell(CellState.full)])
    logimage_problems[0,1] = Problem(rule = Rule([1]),cells = [Cell(CellState.empty),Cell(CellState.full)])
    problem_coordinates, cell_index = logimage_problems.get_guess_candidate()
    assert((problem_coordinates == ProblemCoordinates(0,0)) & (cell_index == 0))

def test_get_guess_candidate_finds_problem_coordinates_and_cell_index_no_modifications():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set=rule_set)
    problem_coordinates, cell_index = logimage_problems.get_guess_candidate()
    assert((problem_coordinates == ProblemCoordinates(0,0)) & (cell_index == 0))

def test_find_problem_with_least_undefined():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set=rule_set)
    logimage_problems[0,0] = Problem(rule = Rule([1]),cells = [Cell(CellState.undefined),Cell(CellState.full)])
    candidate_problem_coordinates = logimage_problems.find_problem_coordinates_with_least_undefined()
    assert(candidate_problem_coordinates == ProblemCoordinates(0,0))

def test_guess_candidate_with_strategy_of_choosing_the_problem_with_least_undefined():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set=rule_set)
    problem_coordinates, cell_index = logimage_problems.get_guess_candidate(heuristic="least_undefined")
    assert((problem_coordinates == ProblemCoordinates(0,0)) & (cell_index == 0))

def test_guess_chooses_last_modified_problem_with_undefined_cell_and_chooses_first_cell_close_to_defined_cell():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage = Logimage(rules = rule_set)
    logimage.logimage_problems[0,0] = Problem(rule = Rule([1]),cells = [Cell(CellState.full),Cell(CellState.undefined)])
    guess = logimage.create_guess()
    expected_modification = Modification(problem_coordinates=ProblemCoordinates(0,0),cell_index=1,new_cell=Cell(CellState.full))
    expected_guess = Guess(logimage_problems=logimage.logimage_problems,modification=expected_modification)
    assert(guess == expected_guess)

def test_adding_new_guess_to_logimage_appends_guess_list():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage = Logimage(rules = rule_set)
    logimage.add_new_guess()
    assert((len(logimage.guess_list) == 1) & (isinstance(logimage.guess_list[0],Guess)))

@pytest.mark.skip()
def test_guess_chooses_first_undefined_cell_when_modifications_empty():
    assert(True == False)

def test_guess_changes_cell_state_of_modification_with_change_try():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set=rule_set)
    modification = Modification(ProblemCoordinates(0,0),0,Cell(CellState.full))
    guess = Guess(logimage_problems,modification)
    guess.change_try()
    expected_modification = Modification(ProblemCoordinates(0,0),0,Cell(CellState.empty))
    assert((guess.logimage_problems == logimage_problems) & (guess.modification == expected_modification))

def test_guess_changes_change_try_when_all_options_tried_raises():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage_problems = LogimageProblems(rule_set=rule_set)
    modification = Modification(ProblemCoordinates(0,0),0,Cell(CellState.full))
    guess = Guess(logimage_problems,modification)
    guess.change_try()
    with pytest.raises(InvalidGuessChange) as err:
        guess.change_try()

def test_change_try_last_guess_modifies_last_guess_in_guess_list():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage = Logimage(rules = rule_set)
    logimage.add_new_guess()
    logimage.change_try_last_guess()
    expected_guess = Guess(logimage.logimage_problems,Modification(ProblemCoordinates(0,0),0,Cell(CellState.full)))
    expected_guess.change_try()
    assert((len(logimage.guess_list) == 1) & (logimage.guess_list[0] == expected_guess))

def test_change_try_last_guess_in_guess_list_with_1_guess_already_max_try_raises():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage = Logimage(rules = rule_set)
    logimage.add_new_guess()
    logimage.change_try_last_guess()
    with pytest.raises(NoGuessLeft) as err:
        logimage.change_try_last_guess()

def test_change_try_last_guess_in_guess_list_with_1_guess_already_max_try_raises():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage = Logimage(rules = rule_set)
    logimage.add_new_guess()
    logimage.change_try_last_guess()
    with pytest.raises(NoGuessLeft) as err:
        logimage.change_try_last_guess()

def test_change_try_last_guess_in_guess_list_with_first_guess_changeable_but_not_second_removes_second_and_changes_first():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage = Logimage(rules = rule_set)
    first_guess = Guess(logimage.logimage_problems,Modification(ProblemCoordinates(0,0),0,Cell(CellState.full)))
    second_guess = Guess(logimage.logimage_problems,Modification(ProblemCoordinates(0,0),1,Cell(CellState.full)))
    second_guess.change_try()
    logimage.guess_list = [copy.deepcopy(first_guess),copy.deepcopy(second_guess)]
    logimage.change_try_last_guess()
    expected_guess = copy.deepcopy(first_guess)
    expected_guess.change_try()
    assert((len(logimage.guess_list) == 1) & (logimage.guess_list[0] == expected_guess))

def test_guess_solves_simple_multisolution_logimage():
    rule_set = RuleSet(row_rules=RuleList([Rule([1]),Rule([1])]),column_rules=RuleList([Rule([1]),Rule([1])]))
    logimage = Logimage(rules = rule_set)
    logimage.solve()
    assert(logimage.solved == True)

# Guess appends list of guesses

# Guess tries one solution

# Guess stores stalled logimage_problem, and defines modification to make

# implementing guess replaces logimage LogimageProblems by the guess one, and applies guess modification 

# when guess leads to solved logimage, stops

# When solution leads to error, guess modifies solution

# When two possibilities in guess leads to error and only one guess in list of guesses, grid is impossible

# When two possibilities in guess leads to error and other guess before, guess is deleted and previous guess is tried

@pytest.mark.skip()
def test_guess_takes_logimage_problems_and_last_candidate_problem():
    rule_set = RuleSet(row_rules=RuleList([Rule([2]),Rule([2])]),column_rules=RuleList([Rule([2]),Rule([2])]))
    logimage_problems = LogimageProblems(rule_set = rule_set)
    last_candidate_problem = ProblemCoordinates(0,0)
    guess = Guess(logimage_problems,last_candidate_problem)
    assert(guess.try_number == 0)