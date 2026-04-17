//! Enhanced tree list widget: column headers, multi-column tree, inline editing,
//! row drag-drop, custom row widgets, and alternating row colors.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Row and column identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnId(pub u64);

// ---------------------------------------------------------------------------
// Column definition
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnSortDirection { Ascending, Descending, None }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnAlignment { Left, Center, Right }

#[derive(Debug, Clone)]
pub struct TreeColumn {
    pub id: ColumnId,
    pub header: String,
    pub width: f32,
    pub min_width: f32,
    pub max_width: f32,
    pub resizable: bool,
    pub sortable: bool,
    pub sort_direction: ColumnSortDirection,
    pub alignment: ColumnAlignment,
    pub visible: bool,
    pub editable: bool,
    pub tooltip: String,
    pub stretch_factor: f32,
}

impl TreeColumn {
    pub fn new(id: ColumnId, header: impl Into<String>) -> Self {
        Self {
            id, header: header.into(), width: 150.0, min_width: 50.0, max_width: 500.0,
            resizable: true, sortable: true, sort_direction: ColumnSortDirection::None,
            alignment: ColumnAlignment::Left, visible: true, editable: false,
            tooltip: String::new(), stretch_factor: 1.0,
        }
    }

    pub fn with_width(mut self, width: f32) -> Self { self.width = width; self }
    pub fn with_alignment(mut self, align: ColumnAlignment) -> Self { self.alignment = align; self }
    pub fn with_editable(mut self, editable: bool) -> Self { self.editable = editable; self }
    pub fn with_sortable(mut self, sortable: bool) -> Self { self.sortable = sortable; self }
}

// ---------------------------------------------------------------------------
// Cell value
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum CellValue {
    Text(String),
    Number(f64),
    Bool(bool),
    Icon(String),
    Progress(f32),
    Color([f32; 4]),
    Custom(String),
    Empty,
}

impl CellValue {
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(t) => t.clone(),
            Self::Number(n) => format!("{:.2}", n),
            Self::Bool(b) => b.to_string(),
            Self::Icon(i) => i.clone(),
            Self::Progress(p) => format!("{:.0}%", p * 100.0),
            Self::Color(c) => format!("rgba({},{},{},{})", c[0], c[1], c[2], c[3]),
            Self::Custom(s) => s.clone(),
            Self::Empty => String::new(),
        }
    }

    pub fn sort_key(&self) -> String {
        match self {
            Self::Number(n) => format!("{:020.6}", n + 1e15),
            Self::Bool(b) => if *b { "1".to_string() } else { "0".to_string() },
            other => other.as_text(),
        }
    }
}

impl fmt::Display for CellValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_text())
    }
}

// ---------------------------------------------------------------------------
// Row
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TreeRow {
    pub id: RowId,
    pub cells: HashMap<ColumnId, CellValue>,
    pub parent: Option<RowId>,
    pub children: Vec<RowId>,
    pub expanded: bool,
    pub selected: bool,
    pub visible: bool,
    pub icon: Option<String>,
    pub depth: u32,
    pub draggable: bool,
    pub drop_target: bool,
    pub highlight_color: Option<[f32; 4]>,
    pub tooltip: String,
    pub enabled: bool,
    pub tag: Option<String>,
}

impl TreeRow {
    pub fn new(id: RowId) -> Self {
        Self {
            id, cells: HashMap::new(), parent: None, children: Vec::new(),
            expanded: false, selected: false, visible: true, icon: None,
            depth: 0, draggable: true, drop_target: true, highlight_color: None,
            tooltip: String::new(), enabled: true, tag: None,
        }
    }

    pub fn set_cell(&mut self, column: ColumnId, value: CellValue) {
        self.cells.insert(column, value);
    }

    pub fn get_cell(&self, column: ColumnId) -> Option<&CellValue> {
        self.cells.get(&column)
    }

    pub fn has_children(&self) -> bool { !self.children.is_empty() }
    pub fn is_leaf(&self) -> bool { self.children.is_empty() }
    pub fn is_root(&self) -> bool { self.parent.is_none() }
}

// ---------------------------------------------------------------------------
// Drag drop
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPosition { Before, After, Inside }

#[derive(Debug, Clone)]
pub struct DragState {
    pub dragging: bool,
    pub source_row: Option<RowId>,
    pub target_row: Option<RowId>,
    pub drop_position: DropPosition,
    pub drag_start_y: f32,
    pub current_y: f32,
}

impl DragState {
    pub fn new() -> Self {
        Self {
            dragging: false, source_row: None, target_row: None,
            drop_position: DropPosition::After, drag_start_y: 0.0, current_y: 0.0,
        }
    }

    pub fn start(&mut self, row: RowId, y: f32) {
        self.dragging = true;
        self.source_row = Some(row);
        self.drag_start_y = y;
        self.current_y = y;
    }

    pub fn update(&mut self, y: f32) { self.current_y = y; }

    pub fn end(&mut self) {
        self.dragging = false;
        self.source_row = None;
        self.target_row = None;
    }
}

impl Default for DragState {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum TreeListEvent {
    RowSelected(RowId),
    RowDeselected(RowId),
    RowExpanded(RowId),
    RowCollapsed(RowId),
    RowDoubleClicked(RowId),
    CellEdited(RowId, ColumnId, CellValue),
    RowDragDrop(RowId, RowId, DropPosition),
    ColumnResized(ColumnId, f32),
    ColumnSorted(ColumnId, ColumnSortDirection),
    SelectionChanged,
    ContextMenu(RowId),
}

// ---------------------------------------------------------------------------
// Tree list state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionModeV2 { Single, Multi, Extended }

pub struct TreeListState {
    pub columns: Vec<TreeColumn>,
    pub rows: HashMap<RowId, TreeRow>,
    pub root_rows: Vec<RowId>,
    pub selection_mode: SelectionModeV2,
    pub selected_rows: Vec<RowId>,
    pub events: Vec<TreeListEvent>,
    pub next_row_id: u64,
    pub next_column_id: u64,
    pub drag_state: DragState,
    pub alternating_colors: bool,
    pub color_even: [f32; 4],
    pub color_odd: [f32; 4],
    pub row_height: f32,
    pub indent_size: f32,
    pub show_root_lines: bool,
    pub show_icons: bool,
    pub filter_text: String,
    pub scroll_y: f32,
    pub total_height: f32,
    pub editing_cell: Option<(RowId, ColumnId)>,
    pub header_visible: bool,
    pub header_height: f32,
}

impl TreeListState {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(), rows: HashMap::new(), root_rows: Vec::new(),
            selection_mode: SelectionModeV2::Single, selected_rows: Vec::new(),
            events: Vec::new(), next_row_id: 1, next_column_id: 1,
            drag_state: DragState::new(), alternating_colors: true,
            color_even: [0.18, 0.18, 0.18, 1.0], color_odd: [0.15, 0.15, 0.15, 1.0],
            row_height: 22.0, indent_size: 20.0, show_root_lines: true,
            show_icons: true, filter_text: String::new(), scroll_y: 0.0,
            total_height: 0.0, editing_cell: None, header_visible: true,
            header_height: 28.0,
        }
    }

    pub fn add_column(&mut self, header: impl Into<String>) -> ColumnId {
        let id = ColumnId(self.next_column_id);
        self.next_column_id += 1;
        self.columns.push(TreeColumn::new(id, header));
        id
    }

    pub fn add_row(&mut self, parent: Option<RowId>) -> RowId {
        let id = RowId(self.next_row_id);
        self.next_row_id += 1;
        let depth = parent.map(|p| self.rows.get(&p).map(|r| r.depth + 1).unwrap_or(0)).unwrap_or(0);
        let mut row = TreeRow::new(id);
        row.parent = parent;
        row.depth = depth;

        if let Some(parent_id) = parent {
            if let Some(parent_row) = self.rows.get_mut(&parent_id) {
                parent_row.children.push(id);
            }
        } else {
            self.root_rows.push(id);
        }

        self.rows.insert(id, row);
        id
    }

    pub fn remove_row(&mut self, id: RowId) -> bool {
        // Recursively remove children.
        if let Some(row) = self.rows.get(&id) {
            let children: Vec<RowId> = row.children.clone();
            for child in children { self.remove_row(child); }
        }

        // Remove from parent.
        if let Some(row) = self.rows.get(&id) {
            if let Some(parent_id) = row.parent {
                if let Some(parent) = self.rows.get_mut(&parent_id) {
                    parent.children.retain(|&c| c != id);
                }
            }
        }

        self.root_rows.retain(|&r| r != id);
        self.selected_rows.retain(|&r| r != id);
        self.rows.remove(&id).is_some()
    }

    pub fn set_cell(&mut self, row: RowId, column: ColumnId, value: CellValue) {
        if let Some(r) = self.rows.get_mut(&row) { r.set_cell(column, value); }
    }

    pub fn select_row(&mut self, id: RowId) {
        match self.selection_mode {
            SelectionModeV2::Single => {
                for row in self.rows.values_mut() { row.selected = false; }
                self.selected_rows.clear();
            }
            _ => {}
        }
        if let Some(row) = self.rows.get_mut(&id) {
            row.selected = true;
            if !self.selected_rows.contains(&id) { self.selected_rows.push(id); }
            self.events.push(TreeListEvent::RowSelected(id));
        }
    }

    pub fn deselect_all(&mut self) {
        for row in self.rows.values_mut() { row.selected = false; }
        self.selected_rows.clear();
        self.events.push(TreeListEvent::SelectionChanged);
    }

    pub fn toggle_expand(&mut self, id: RowId) {
        if let Some(row) = self.rows.get_mut(&id) {
            row.expanded = !row.expanded;
            if row.expanded {
                self.events.push(TreeListEvent::RowExpanded(id));
            } else {
                self.events.push(TreeListEvent::RowCollapsed(id));
            }
        }
    }

    pub fn expand_all(&mut self) {
        for row in self.rows.values_mut() {
            if !row.children.is_empty() { row.expanded = true; }
        }
    }

    pub fn collapse_all(&mut self) {
        for row in self.rows.values_mut() { row.expanded = false; }
    }

    pub fn sort_by_column(&mut self, column: ColumnId, direction: ColumnSortDirection) {
        // Update column sort state.
        for col in &mut self.columns {
            col.sort_direction = if col.id == column { direction } else { ColumnSortDirection::None };
        }

        let ascending = matches!(direction, ColumnSortDirection::Ascending);

        // Sort root rows.
        let rows_ref = &self.rows;
        self.root_rows.sort_by(|a, b| {
            let va = rows_ref.get(a).and_then(|r| r.get_cell(column)).map(|v| v.sort_key()).unwrap_or_default();
            let vb = rows_ref.get(b).and_then(|r| r.get_cell(column)).map(|v| v.sort_key()).unwrap_or_default();
            if ascending { va.cmp(&vb) } else { vb.cmp(&va) }
        });

        self.events.push(TreeListEvent::ColumnSorted(column, direction));
    }

    pub fn set_filter(&mut self, text: impl Into<String>) {
        self.filter_text = text.into();
        self.apply_filter();
    }

    fn apply_filter(&mut self) {
        if self.filter_text.is_empty() {
            for row in self.rows.values_mut() { row.visible = true; }
            return;
        }
        let filter = self.filter_text.to_lowercase();
        for row in self.rows.values_mut() {
            row.visible = row.cells.values().any(|v| v.as_text().to_lowercase().contains(&filter));
        }
    }

    pub fn flattened_visible_rows(&self) -> Vec<RowId> {
        let mut result = Vec::new();
        fn collect(state: &TreeListState, rows: &[RowId], result: &mut Vec<RowId>) {
            for &id in rows {
                if let Some(row) = state.rows.get(&id) {
                    if !row.visible { continue; }
                    result.push(id);
                    if row.expanded && !row.children.is_empty() {
                        collect(state, &row.children, result);
                    }
                }
            }
        }
        collect(self, &self.root_rows.clone(), &mut result);
        result
    }

    pub fn row_count(&self) -> usize { self.rows.len() }
    pub fn column_count(&self) -> usize { self.columns.len() }
    pub fn visible_row_count(&self) -> usize { self.flattened_visible_rows().len() }

    pub fn drain_events(&mut self) -> Vec<TreeListEvent> { std::mem::take(&mut self.events) }

    pub fn begin_edit(&mut self, row: RowId, column: ColumnId) {
        self.editing_cell = Some((row, column));
    }

    pub fn commit_edit(&mut self, value: CellValue) {
        if let Some((row, col)) = self.editing_cell.take() {
            self.set_cell(row, col, value.clone());
            self.events.push(TreeListEvent::CellEdited(row, col, value));
        }
    }

    pub fn cancel_edit(&mut self) { self.editing_cell = None; }

    pub fn move_row(&mut self, source: RowId, target: RowId, position: DropPosition) {
        // Detach source from its parent.
        if let Some(row) = self.rows.get(&source) {
            if let Some(parent) = row.parent {
                if let Some(p) = self.rows.get_mut(&parent) {
                    p.children.retain(|&c| c != source);
                }
            }
        }
        self.root_rows.retain(|&r| r != source);

        match position {
            DropPosition::Inside => {
                if let Some(target_row) = self.rows.get_mut(&target) {
                    target_row.children.push(source);
                    target_row.expanded = true;
                }
                if let Some(source_row) = self.rows.get_mut(&source) {
                    source_row.parent = Some(target);
                    source_row.depth = self.rows.get(&target).map(|r| r.depth + 1).unwrap_or(0);
                }
            }
            DropPosition::Before | DropPosition::After => {
                let parent = self.rows.get(&target).and_then(|r| r.parent);
                if let Some(parent_id) = parent {
                    if let Some(parent_row) = self.rows.get_mut(&parent_id) {
                        let idx = parent_row.children.iter().position(|&c| c == target).unwrap_or(0);
                        let insert_idx = if matches!(position, DropPosition::After) { idx + 1 } else { idx };
                        parent_row.children.insert(insert_idx, source);
                    }
                } else {
                    let idx = self.root_rows.iter().position(|&r| r == target).unwrap_or(0);
                    let insert_idx = if matches!(position, DropPosition::After) { idx + 1 } else { idx };
                    self.root_rows.insert(insert_idx, source);
                }
                if let Some(source_row) = self.rows.get_mut(&source) {
                    source_row.parent = parent;
                    source_row.depth = parent.and_then(|p| self.rows.get(&p).map(|r| r.depth + 1)).unwrap_or(0);
                }
            }
        }

        self.events.push(TreeListEvent::RowDragDrop(source, target, position));
    }
}

impl Default for TreeListState {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_list_basic() {
        let mut tree = TreeListState::new();
        let col = tree.add_column("Name");
        let r1 = tree.add_row(None);
        let r2 = tree.add_row(Some(r1));
        tree.set_cell(r1, col, CellValue::Text("Parent".to_string()));
        tree.set_cell(r2, col, CellValue::Text("Child".to_string()));
        assert_eq!(tree.row_count(), 2);

        tree.toggle_expand(r1);
        let visible = tree.flattened_visible_rows();
        assert_eq!(visible.len(), 2);
    }

    #[test]
    fn sort_column() {
        let mut tree = TreeListState::new();
        let col = tree.add_column("Value");
        let r1 = tree.add_row(None);
        let r2 = tree.add_row(None);
        tree.set_cell(r1, col, CellValue::Number(2.0));
        tree.set_cell(r2, col, CellValue::Number(1.0));
        tree.sort_by_column(col, ColumnSortDirection::Ascending);
        assert_eq!(tree.root_rows[0], r2);
    }
}
