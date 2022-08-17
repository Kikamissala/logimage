from logimage.cell import Cell, CellState, Grid, \
 InvalidGridSet,InvalidCellStateModification, CellUpdateError
import numpy as np
import pytest

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