from logimage.cell import Cell, CellState, Grid, InvalidCellStateModification
from .problem import Problem, ProblemDict ,InvalidProblem
from logimage.rule import InvalidRule, RuleSet
import copy
import numpy as np
from matplotlib import pyplot


class NoGuessLeft(Exception):
    pass

class Logimage:
    
    def __init__(self, rules : RuleSet, guessing_heuristic = "last_modifications"):
        self.rules = rules
        self.grid = Grid(len(rules.row_rules), len(rules.column_rules))
        self.raise_if_rules_invalid()
        self.logimage_problems = LogimageProblems(rule_set=rules)
        self.guess_list = []
        self.solved = False
        self.guessing_heuristic = guessing_heuristic

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
    
    def create_guess(self):
        problem_coordinates, modification_cell_index = self.logimage_problems.get_guess_candidate(self.guessing_heuristic)
        modification = Modification(problem_coordinates,modification_cell_index,Cell(CellState.full))
        guess = Guess(self.logimage_problems,modification)
        return guess
    
    def add_new_guess(self):
        new_guess = self.create_guess()
        self.guess_list.append(new_guess)
    
    def change_try_last_guess(self):
        last_guess = self.guess_list[-1]
        try:
            last_guess.change_try()
        except InvalidGuessChange:
            if len(self.guess_list) <= 1:
                raise NoGuessLeft("All guess possibilities tried for first guess, unable to solve") from None
            else:
                self.guess_list = self.guess_list[:-1]
                self.change_try_last_guess()
                return
        self.guess_list = self.guess_list[:-1] + [last_guess]
    
    def solve_old(self):
        self.logimage_problems.solve()
        self.solved = self.logimage_problems.solved
        self.finished = self.logimage_problems.finished
        self.update_grid_from_problems()
    
    def solve(self):
        while self.solved is not True:
            try:
                self.logimage_problems.solve()
            except (InvalidProblem, InvalidCellStateModification):
                try:
                    self.change_try_last_guess()
                except NoGuessLeft:
                    self.solved = False
                    return
                guess_to_try = copy.deepcopy(self.guess_list[-1])
                new_logimage_problems = copy.deepcopy(guess_to_try.logimage_problems)
                new_logimage_problems.modify(guess_to_try.modification)
                self.logimage_problems = copy.deepcopy(new_logimage_problems)
            else:
                if self.logimage_problems.solved != True:
                    print(self.logimage_problems.compute_solving_percentage())
                    self.add_new_guess()
                    guess_to_try = copy.deepcopy(self.guess_list[-1])
                    new_logimage_problems = copy.deepcopy(guess_to_try.logimage_problems)
                    new_logimage_problems.modify(guess_to_try.modification)
                    self.logimage_problems = copy.deepcopy(new_logimage_problems)
                else:                
                    self.solved = self.logimage_problems.solved
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

class Modification:

    def __init__(self,problem_coordinates:ProblemCoordinates, cell_index,new_cell:Cell):
        self.problem_coordinates = problem_coordinates
        self.cell_index = cell_index
        self.new_cell = new_cell
    
    def __eq__(self,other):
        return (isinstance(other,Modification)) & (self.problem_coordinates == other.problem_coordinates) \
        & (self.cell_index == other.cell_index) & (self.new_cell == other.new_cell)
    
    def __repr__(self):
        return f"Modification : {self.problem_coordinates}, {self.cell_index}, {self.new_cell}"

class LogimageProblems:
    
    def __init__(self, rule_set:RuleSet):
        self.row_problems = ProblemDict(rule_list = rule_set.row_rules, problem_size= len(rule_set.column_rules))
        self.column_problems = ProblemDict(rule_list = rule_set.column_rules, problem_size= len(rule_set.row_rules))
        self.candidate_problems = set()
        self.modifications = []
    
    def __getitem__(self,dimension):
        if dimension ==0:
            return self.row_problems
        elif dimension == 1:
            return self.column_problems
        else:
            raise KeyError("LogimageProblems only has 2 elements")
    
    def __eq__(self,other):
        return (isinstance(other, LogimageProblems)) & (self.row_problems == other.row_problems) \
            & (self.column_problems == other.column_problems) 
    
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
        for modified_index in modified_indexes:
            other_dimension = self.other_dimension(dimension)
            self.__getitem__(other_dimension)[modified_index][problem_index] = Cell(value[modified_index].cell_state)
            self.add_candidate_problem(dimension=other_dimension,index=modified_index)
            problem_coordinates=ProblemCoordinates(dimension,problem_index)
            cell_index = modified_index
            new_cell = value[modified_index]
            modification = Modification(problem_coordinates,cell_index,new_cell)
            self.modifications.append(modification)
    
    def modify(self,modification:Modification):
        problem_to_modify = copy.deepcopy(self.get_problem_from_coordinates(modification.problem_coordinates))
        problem_to_modify[modification.cell_index] = modification.new_cell
        dimension = modification.problem_coordinates.dimension
        index = modification.problem_coordinates.index
        self.__setitem__(pos = (dimension,index),value = problem_to_modify)

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

    def is_neighbour_cell_undefined(self, problem_to_scan, modification_cell_index,gap):
        neighbour_cell_index = modification_cell_index + gap
        cell_after_modification = problem_to_scan[neighbour_cell_index]
        if cell_after_modification.cell_state == CellState.undefined:
            return True

    def get_guess_candidate(self,heuristic = "last_modifications"):
        if heuristic == "last_modifications":
            problem_coordinates, undefined_cell_index = self.get_guess_candidate_in_last_modifications()
        elif heuristic == "least_undefined":
            problem_coordinates, undefined_cell_index = self.get_guess_candidate_in_problems_with_least_undefined()
        if problem_coordinates is not None:
            return problem_coordinates, undefined_cell_index
        else:
            for index, problem in enumerate(self.row_problems):
                first_undefined_cell_index = problem.first_undefined_cell_index()
                if first_undefined_cell_index is not None:
                    problem_coordinates = ProblemCoordinates(0,index)
                    return problem_coordinates, first_undefined_cell_index
                else:
                    continue
    
    def get_guess_candidate_in_last_modifications(self):
        modifications_to_scan = self.modifications[::-1]
        problem_coordinates, undefined_cell_index = self.get_guess_candidate_in_modifications(modifications_to_scan)
        return problem_coordinates, undefined_cell_index
    
    def get_guess_candidate_in_modifications(self, modifications):
        index = 0
        while index < len(modifications):
            modification = modifications[index]
            problem_coordinates = modification.problem_coordinates
            modification_cell_index = modification.cell_index
            problem_to_scan = self.__getitem__(problem_coordinates.dimension)[problem_coordinates.index]
            if modification_cell_index == 0:
                if self.is_neighbour_cell_undefined(problem_to_scan,modification_cell_index,gap = 1):
                    return problem_coordinates, modification_cell_index + 1
            elif modification_cell_index == problem_to_scan.length - 1:
                if self.is_neighbour_cell_undefined(problem_to_scan,modification_cell_index,gap = -1):
                    return problem_coordinates, modification_cell_index - 1
            else:
                for gap in [-1,1]:
                    if self.is_neighbour_cell_undefined(problem_to_scan,modification_cell_index,gap):
                        return problem_coordinates, modification_cell_index + gap
            index += 1
        return None, None

    def find_problem_coordinates_with_least_undefined(self):
        output_problem_coordinates = None
        min_number_of_undefined = None
        for problem_index,problem in self.row_problems.items():
            if problem.first_undefined_cell_index() is None:
                continue
            else:
                number_of_undefined = problem.get_number_of_cell_in_state(CellState.undefined)
                if min_number_of_undefined is None:
                    min_number_of_undefined = number_of_undefined
                    output_problem_coordinates = ProblemCoordinates(0,problem_index)
                elif number_of_undefined < min_number_of_undefined:
                    min_number_of_undefined = number_of_undefined
                    output_problem_coordinates = ProblemCoordinates(0,problem_index)
        for problem_index,problem in self.column_problems.items():
            if problem.first_undefined_cell_index() is None:
                continue
            else:
                number_of_undefined = len([cell for cell in problem if cell.cell_state == CellState.undefined])
                if number_of_undefined < min_number_of_undefined:
                    min_number_of_undefined = number_of_undefined
                    output_problem_coordinates = ProblemCoordinates(0,problem_index)
        return output_problem_coordinates
    
    def get_guess_candidate_in_problems_with_least_undefined(self):
        problem_with_least_undefined_coordinates = self.find_problem_coordinates_with_least_undefined()
        modifications_to_scan = [modification for modification in self.modifications if modification.problem_coordinates == problem_with_least_undefined_coordinates]
        problem_coordinates, undefined_cell_index = self.get_guess_candidate_in_modifications(modifications_to_scan)
        return problem_coordinates, undefined_cell_index
    
    def compute_solving_percentage(self):
        number_of_cells_in_logimage = len(self.row_problems) * len(self.column_problems)
        number_of_unsolved_cells = sum([problem.get_number_of_cell_in_state(CellState.undefined) for problem_index,problem in self.row_problems.items()])
        number_of_solved_cells = number_of_cells_in_logimage - number_of_unsolved_cells
        solving_percentage = (number_of_solved_cells / number_of_cells_in_logimage) * 100
        return solving_percentage

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

class InvalidGuessChange(Exception):
    pass

class Guess:

    def __init__(self, logimage_problems:LogimageProblems, modification : Modification):
        self.logimage_problems = logimage_problems
        self.modification = modification
        self.try_number = 0
    
    def __eq__(self,other):
        return isinstance(other, Guess) & \
        (self.logimage_problems == other.logimage_problems) & \
        (self.modification == other.modification) & \
        (self.try_number == other.try_number)
    
    def change_try(self):
        if self.try_number == 0:
            if self.modification.new_cell.cell_state == CellState.full:
                new_cell = Cell(CellState.empty)
                new_modification = Modification(self.modification.problem_coordinates, self.modification.cell_index,new_cell)
            elif self.modification.new_cell.cell_state == CellState.empty:
                new_cell = Cell(CellState.full)
                new_modification = Modification(self.modification.problem_coordinates, self.modification.cell_index,new_cell)                
            else:
                raise InvalidGuessChange("Impossible to modify guess if Cell is neither full nor empty")
            self.modification = new_modification
            self.try_number += 1
        elif self.try_number >= 1:
            raise InvalidGuessChange("all options have been tried, impossible to try another")
