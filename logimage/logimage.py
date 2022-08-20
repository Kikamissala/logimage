from logimage.cell import Cell, CellState, Grid
from .problem import Problem, ProblemDict
from logimage.rule import InvalidRule, RuleSet
import copy
import numpy as np
from matplotlib import pyplot


class Logimage:
    
    def __init__(self, rules : RuleSet):
        self.rules = rules
        self.grid = Grid(len(rules.row_rules), len(rules.column_rules))
        self.raise_if_rules_invalid()
        self.logimage_problems = LogimageProblems(rule_set=rules)

    def is_rule_exceeding_grid_size(self):
        maximum_minimum_possible_len_row = self.rules.column_rules.compute_maximum_minimum_possible_len()
        maximum_minimum_possible_len_column = self.rules.row_rules.compute_maximum_minimum_possible_len()
        if maximum_minimum_possible_len_row > self.grid.row_number:
            return True
        elif maximum_minimum_possible_len_column > self.grid.column_number:
            return True
        else:
            return False

    def raise_if_rules_invalid(self):
        if self.is_rule_exceeding_grid_size():
            raise InvalidRule("A rule is exceeding grid size")
    
    def update_grid_from_problems(self):
        for row_index, problem in self.logimage_problems.row_problems.items():
            numerized_cells = problem.numerize_cell_list()
            self.grid[row_index,:] = np.array(numerized_cells)
    
    def solve(self):
        self.logimage_problems.solve()
        self.solved = self.logimage_problems.solved
        self.finished = self.logimage_problems.finished
        self.update_grid_from_problems()
    
    def plot_grid(self):
        pyplot.figure(figsize=(5,5))
        pyplot.imshow(self.grid.cells)
        pyplot.show()

class ProblemCoordinates:

    def __init__(self, dimension, index):
        self.dimension = dimension
        self.index = index

    def __eq__(self, other):
        return isinstance(other, ProblemCoordinates) & (self.dimension == other.dimension) & (self.index == other.index)

    def __hash__(self):
        return hash((self.dimension,self.index))
    
    def __repr__(self):
        return f"ProblemCoordinates : ({self.dimension},{self.index})"

class LogimageProblems:
    
    def __init__(self, rule_set:RuleSet):
        self.row_problems = ProblemDict(rule_list = rule_set.row_rules, problem_size= len(rule_set.column_rules))
        self.column_problems = ProblemDict(rule_list = rule_set.column_rules, problem_size= len(rule_set.row_rules))
        self.candidate_problems = set()
    
    def __getitem__(self,dimension):
        if dimension ==0:
            return self.row_problems
        elif dimension == 1:
            return self.column_problems
        else:
            raise KeyError("LogimageProblems only has 2 elements")
    
    def other_dimension(self,dimension):
        if dimension == 0:
            return 1
        elif dimension == 1:
            return 0

    def __setitem__(self,pos,value:Problem):
        dimension,problem_index = pos
        if dimension not in [0,1]:
            raise KeyError("LogimageProblems only has 2 elements")
        previous_problem = self.__getitem__(dimension)[problem_index]
        modified_indexes = previous_problem.get_updated_state_indexes(value)
        self.__getitem__(dimension)[problem_index] = value
        for index in modified_indexes:
            other_dimension = self.other_dimension(dimension)
            self.__getitem__(other_dimension)[index][problem_index] = Cell(value[index].cell_state)
            self.add_candidate_problem(dimension=other_dimension,index=index)

    def add_candidate_problem(self,dimension, index):
        self.candidate_problems.add(ProblemCoordinates(dimension,index))
    
    def remove_candidate_problem(self,dimension, index):
        self.candidate_problems.remove(ProblemCoordinates(dimension,index))

    def solve_problem(self, dimension, index):
        problem_to_solve = self.__getitem__(dimension)[index]
        solved_problem = problem_to_solve.solve()
        self.__setitem__(pos = (dimension,index),value=solved_problem)
        problem_coordinates = ProblemCoordinates(dimension,index)
        if problem_coordinates in self.candidate_problems:
            self.remove_candidate_problem(dimension,index)
    
    def get_problem_from_coordinates(self, coordinates):
        dimension = coordinates.dimension
        index = coordinates.index
        return self.__getitem__(dimension)[index]

    def select_candidate_problem(self):
        if len(self.candidate_problems) == 0:
            return
        chosen_coordinates = None
        output_number_of_undefined = None
        candidate_problems = copy.deepcopy(self.candidate_problems)
        for problem_coordinates in candidate_problems:
            problem = self.get_problem_from_coordinates(problem_coordinates)   
            number_of_undefined = problem.get_number_of_cell_in_state(CellState.undefined)
            if number_of_undefined == 0:
                self.remove_candidate_problem(problem_coordinates.dimension,problem_coordinates.index)
                continue
            elif number_of_undefined == 1:
                chosen_coordinates = problem_coordinates
                break
            elif output_number_of_undefined is None:
                output_number_of_undefined = number_of_undefined
                chosen_coordinates = problem_coordinates
            elif number_of_undefined < output_number_of_undefined:
                output_number_of_undefined = number_of_undefined
                chosen_coordinates = problem_coordinates
        return chosen_coordinates
    
    def get_candidate_and_solve_problem(self):
        chosen_coordinates = self.select_candidate_problem()
        if chosen_coordinates is None:
            return
        else:
            self.solve_problem(dimension=chosen_coordinates.dimension,index=chosen_coordinates.index)

    def scan_fully_defined_problems(self):
        for problem_index,problem in self.row_problems.items():
            if problem.is_line_fully_defined_by_rule():
                self.add_candidate_problem(dimension=0,index=problem_index)
        for problem_index,problem in self.column_problems.items():
            if problem.is_line_fully_defined_by_rule():
                self.add_candidate_problem(dimension=1,index=problem_index)
    
    def scan_for_overlap_problems(self):
        for problem_index,problem in self.row_problems.items():
            if problem.is_subject_to_overlap_solving():
                self.add_candidate_problem(dimension=0,index=problem_index)
        for problem_index,problem in self.column_problems.items():
            if problem.is_subject_to_overlap_solving():
                self.add_candidate_problem(dimension=1,index=problem_index)

    def solve(self):
        self.scan_fully_defined_problems()
        self.scan_for_overlap_problems()
        while len(self.candidate_problems) > 0:
            self.get_candidate_and_solve_problem()
        self.finished = True
        if self.is_solved():
            self.solved = True
        else:
            self.solved = False
    
    def is_solved(self):
        for problem_index,problem in self.row_problems.items():
            for cell in problem:
                if cell.cell_state == CellState.undefined:
                    return False
        return True