from logimage.main import Cell, CellState, FullBlock, RuleList, Grid, \
 InvalidGridSet,InvalidCellStateModification, Logimage, InvalidRule,RuleElement, Rule, RuleSet, Problem,\
    Solution,FullBlock, InvalidProblem, CellUpdateError, ProblemAddError
import numpy as np
import pytest
import pandas as pd

def test_solve_problem_with_split_needed_with_overlapping_in_second_part():
    problem = Problem(rule = Rule([1,2]), cells = [Cell(CellState.full,rule_element_index=0),Cell(CellState.empty), Cell(CellState.empty), Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])
    problem.solve()
    expected_problem = Problem(rule = Rule([1,2]), cells = [Cell(CellState.full,rule_element_index=0), Cell(CellState.empty), Cell(CellState.empty), Cell(CellState.undefined),Cell(CellState.full,rule_element_index=1),Cell(CellState.undefined)])
    assert(problem == expected_problem)

# def test_solve_problem_with_split_needed_with_overlapping_in_second_part():
#     problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.undefined),Cell(CellState.undefined)])
#     problem.solve()
#     expected_problem = Problem(rule = Rule([2]), cells = [Cell(CellState.undefined),Cell(CellState.full,rule_element_index=0),Cell(CellState.undefined)])
#     assert(problem == expected_problem)