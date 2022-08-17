from scipy import *
import numpy as np
from enum import Enum

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

    # def modify(self,new_cell_state, new_rule_element_index = None):
    #     if isinstance(new_cell_state,CellState) is False:
    #         raise InvalidCellStateModification("new value is not a cell state")
    #     if self.cell_state != CellState.undefined:
    #         raise InvalidCellStateModification("impossible to modify not undefined cell state")
    #     if new_cell_state == CellState.empty:
    #         self.empty()
    #     elif new_cell_state == CellState.full:
    #         self.full()

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
            self.cells[row_index,column_index] = value
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