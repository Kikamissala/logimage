from logimage.cell import Grid
from logimage.rule import InvalidRule, RuleSet

class Logimage:
    
    def __init__(self, grid_dimensions, rules : RuleSet):
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