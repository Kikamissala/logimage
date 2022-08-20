from logimage.cell import Cell, CellState
from logimage.rule import RuleSet, InvalidRule, RuleList, Rule
from logimage.logimage import Logimage, LogimageProblems, ProblemCoordinates
from logimage.problem import Problem,ProblemDict
import pytest

def test_generate_logimage_needs_grid_dimensions_and_list_of_rules_for_rows_and_columns():
    logimage = Logimage(grid_dimensions = (2,2),rules = RuleSet(row_rules = [[1],[1]],column_rules = [[1],[1]]))

def test_logimage_init_raises_when_a_rule_values_exceeds_size_of_grid():
    with pytest.raises(InvalidRule) as err:
        logimage = Logimage(grid_dimensions = (2,2),rules = RuleSet(row_rules = [[1,1],[1]],column_rules = [[1],[1]]))

def test_logimage_init_raises_when_number_of_row_rules_not_equal_number_of_rows():
    with pytest.raises(InvalidRule) as err:
        logimage = Logimage(grid_dimensions = (2,2),rules = RuleSet(row_rules = [[1],[1],[1]],column_rules = [[1],[1]]))

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
    logimage_problems[1,0] = Problem(Rule([1]),cells = [Cell(CellState.full),Cell(CellState.full)])
    assert(logimage_problems[0][1][0] == Cell(CellState.full))

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

def test_run_solve_appends_resolution_history_with_modified_state_cells():
    pass

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
    pass

def test_render_grid_displays_grid():
    pass

def test_reconstruct_resolution_shows_evolution_of_resolution_from_history():
    pass