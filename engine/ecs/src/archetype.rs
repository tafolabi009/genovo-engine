//! Archetype-based component storage for the Genovo ECS.
//!
//! An archetype represents a unique set of component types. Entities sharing the
//! same set of components live together in the same archetype, with components
//! stored in contiguous, column-major arrays (`ComponentColumn`). This layout is
//! cache-friendly and enables fast linear iteration in queries.
//!
//! # Architecture
//!
//! ```text
//! Archetype { Position, Velocity }
//! ┌─────────┬──────────────────┬──────────────────┐
//! │ entities │  column:Position │  column:Velocity │
//! ├─────────┼──────────────────┼──────────────────┤
//! │  e0     │  Pos(1,2)        │  Vel(3,4)        │
//! │  e1     │  Pos(5,6)        │  Vel(7,8)        │
//! │  e2     │  Pos(9,0)        │  Vel(1,2)        │
//! └─────────┴──────────────────┴──────────────────┘
//! ```
//!
//! When a component is added or removed, the entity *moves* from its current
//! archetype to one that has the new set of components (swap-remove in the
//! source, push in the target).

use std::alloc::{self, Layout};
use std::any::TypeId;
use std::collections::HashMap;
use std::ptr;

use crate::component::ComponentId;
use crate::entity::Entity;

// ---------------------------------------------------------------------------
// ArchetypeId
// ---------------------------------------------------------------------------

/// Opaque handle identifying an archetype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArchetypeId(pub(crate) u32);

impl ArchetypeId {
    /// The "empty" archetype that entities with no components belong to.
    pub const EMPTY: Self = Self(0);

    /// Raw index suitable for indexing into a `Vec<Archetype>`.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

// ---------------------------------------------------------------------------
// ComponentInfo — metadata for a single component type
// ---------------------------------------------------------------------------

/// Static metadata about a component type. Captured once when the component is
/// first registered and reused for all subsequent operations.
#[derive(Debug, Clone, Copy)]
pub struct ComponentInfo {
    /// The component type's [`ComponentId`] (wraps `TypeId`).
    pub id: ComponentId,
    /// `std::mem::size_of::<T>()`
    pub size: usize,
    /// `std::mem::align_of::<T>()`
    pub align: usize,
    /// Pointer to `drop_in_place::<T>` — used when removing components.
    pub drop_fn: unsafe fn(*mut u8),
    /// The underlying `TypeId`, duplicated from `ComponentId` for convenience.
    pub type_id: TypeId,
}

impl ComponentInfo {
    /// Build `ComponentInfo` for a concrete component type `T`.
    pub fn of<T: 'static>() -> Self {
        unsafe fn drop_ptr<T>(ptr: *mut u8) {
            unsafe { ptr::drop_in_place(ptr as *mut T); }
        }

        Self {
            id: ComponentId::of_raw(TypeId::of::<T>()),
            size: std::mem::size_of::<T>(),
            align: std::mem::align_of::<T>(),
            drop_fn: drop_ptr::<T>,
            type_id: TypeId::of::<T>(),
        }
    }
}

impl PartialEq for ComponentInfo {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id
    }
}
impl Eq for ComponentInfo {}

impl std::hash::Hash for ComponentInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
    }
}

// ---------------------------------------------------------------------------
// ComponentColumn — type-erased column storage
// ---------------------------------------------------------------------------

/// Type-erased, contiguous column storage for a single component type.
///
/// Internally uses a raw byte buffer (`Vec<u8>`) with manual layout management.
/// This enables cache-friendly linear iteration across all entities in an
/// archetype without per-entity hash lookups.
///
/// # Safety
///
/// All raw pointer methods (`push_raw`, `get_raw`, etc.) require that the caller
/// provides data of the correct type, matching the `item_size` and `item_align`
/// used to construct the column. The type-safe wrappers (`push`, `get`,
/// `get_mut`) enforce this via generics.
pub struct ComponentColumn {
    /// Raw byte storage. Length is always `len * item_size`, but the allocation
    /// is rounded up to `capacity * item_size` where capacity is managed via
    /// the standard growth strategy.
    data: *mut u8,
    /// Number of items currently stored.
    len: usize,
    /// Number of items the current allocation can hold.
    capacity: usize,
    /// Size of a single item in bytes.
    item_size: usize,
    /// Alignment of a single item.
    item_align: usize,
    /// Drop function for the component type.
    drop_fn: unsafe fn(*mut u8),
}

// SAFETY: ComponentColumn is Send + Sync because we only store Send + Sync
// component data (enforced by the Component trait bound at the API boundary).
unsafe impl Send for ComponentColumn {}
unsafe impl Sync for ComponentColumn {}

impl ComponentColumn {
    // -- Construction -------------------------------------------------------

    /// Create a new, empty column for a component described by `info`.
    pub fn new(info: &ComponentInfo) -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
            capacity: 0,
            item_size: info.size,
            item_align: info.align,
            drop_fn: info.drop_fn,
        }
    }

    /// Create a column with pre-allocated capacity.
    pub fn with_capacity(info: &ComponentInfo, capacity: usize) -> Self {
        let mut col = Self::new(info);
        if capacity > 0 && info.size > 0 {
            col.grow(capacity);
        }
        col
    }

    // -- Accessors ----------------------------------------------------------

    /// Number of items stored in this column.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the column is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size of each item in bytes.
    #[inline]
    pub fn item_size(&self) -> usize {
        self.item_size
    }

    /// Alignment of each item.
    #[inline]
    pub fn item_align(&self) -> usize {
        self.item_align
    }

    // -- Raw (unsafe) API ---------------------------------------------------

    /// Append raw bytes as a new item at the end of the column.
    ///
    /// # Safety
    ///
    /// - `data` must point to a valid, initialized value of the correct type.
    /// - The pointed-to value is memcpy'd (not moved through Drop), so the
    ///   caller must `forget` the source if it has a destructor.
    pub unsafe fn push_raw(&mut self, data: *const u8) {
        if self.len == self.capacity {
            self.grow(if self.capacity == 0 {
                4
            } else {
                self.capacity * 2
            });
        }
        if self.item_size > 0 {
            unsafe {
                let dst = self.data.add(self.len * self.item_size);
                ptr::copy_nonoverlapping(data, dst, self.item_size);
            }
        }
        self.len += 1;
    }

    /// Get a raw pointer to the item at `index`.
    ///
    /// # Safety
    ///
    /// `index` must be in bounds (`< self.len`).
    #[inline]
    pub unsafe fn get_raw(&self, index: usize) -> *const u8 {
        debug_assert!(index < self.len, "column index out of bounds");
        unsafe { self.data.add(index * self.item_size) }
    }

    /// Get a raw mutable pointer to the item at `index`.
    ///
    /// # Safety
    ///
    /// `index` must be in bounds.
    #[inline]
    pub unsafe fn get_raw_mut(&mut self, index: usize) -> *mut u8 {
        debug_assert!(index < self.len, "column index out of bounds");
        unsafe { self.data.add(index * self.item_size) }
    }

    /// Swap-remove the item at `index`. The last item in the column takes its
    /// place. The removed item's bytes are **not** returned — the caller must
    /// read the data before calling this if they need it.
    ///
    /// # Safety
    ///
    /// `index` must be in bounds.
    pub unsafe fn swap_remove_and_drop(&mut self, index: usize) {
        debug_assert!(index < self.len, "swap_remove index out of bounds");

        unsafe {
            let ptr = self.data.add(index * self.item_size);
            // Drop the element being removed.
            (self.drop_fn)(ptr);

            self.len -= 1;
            if index != self.len && self.item_size > 0 {
                let last = self.data.add(self.len * self.item_size);
                ptr::copy_nonoverlapping(last, ptr, self.item_size);
            }
        }
    }

    /// Swap-remove the item at `index` **without** dropping it. The raw bytes
    /// of the removed element are written into `out_buf` (which must be at
    /// least `item_size` bytes). The last item moves into the vacated slot.
    ///
    /// # Safety
    ///
    /// - `index` must be in bounds.
    /// - `out_buf` must point to writable memory of at least `item_size` bytes.
    pub unsafe fn swap_remove_raw(&mut self, index: usize, out_buf: *mut u8) {
        debug_assert!(index < self.len, "swap_remove_raw index out of bounds");

        if self.item_size > 0 {
            unsafe {
                let ptr = self.data.add(index * self.item_size);
                // Copy the removed element out.
                ptr::copy_nonoverlapping(ptr, out_buf, self.item_size);

                self.len -= 1;
                if index != self.len {
                    let last = self.data.add(self.len * self.item_size);
                    ptr::copy_nonoverlapping(last, ptr, self.item_size);
                }
            }
        } else {
            self.len -= 1;
        }
    }

    /// Swap-remove the item at `index` and push its bytes into `target` column.
    ///
    /// # Safety
    ///
    /// - `index` must be in bounds.
    /// - `target` must have the same `item_size` and `item_align`.
    pub unsafe fn swap_remove_to(&mut self, index: usize, target: &mut ComponentColumn) {
        debug_assert!(index < self.len, "swap_remove_to index out of bounds");
        debug_assert_eq!(self.item_size, target.item_size);

        if self.item_size > 0 {
            unsafe {
                let ptr = self.data.add(index * self.item_size);
                target.push_raw(ptr);

                self.len -= 1;
                if index != self.len {
                    let last = self.data.add(self.len * self.item_size);
                    ptr::copy_nonoverlapping(last, ptr, self.item_size);
                }
            }
        } else {
            unsafe { target.push_raw(ptr::null()); }
            self.len -= 1;
        }
    }

    // -- Type-safe wrappers -------------------------------------------------

    /// Push a typed value into the column.
    ///
    /// # Panics
    ///
    /// Panics if `size_of::<T>() != self.item_size`.
    pub fn push<T: 'static>(&mut self, value: T) {
        assert_eq!(
            std::mem::size_of::<T>(),
            self.item_size,
            "ComponentColumn::push: type size mismatch"
        );
        unsafe {
            self.push_raw(&value as *const T as *const u8);
        }
        std::mem::forget(value);
    }

    /// Get an immutable reference to the item at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `size_of::<T>() != self.item_size` or `index >= self.len`.
    #[inline]
    pub fn get<T: 'static>(&self, index: usize) -> &T {
        assert_eq!(std::mem::size_of::<T>(), self.item_size);
        assert!(index < self.len, "column get index out of bounds");
        unsafe { &*(self.get_raw(index) as *const T) }
    }

    /// Get a mutable reference to the item at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `size_of::<T>() != self.item_size` or `index >= self.len`.
    #[inline]
    pub fn get_mut<T: 'static>(&mut self, index: usize) -> &mut T {
        assert_eq!(std::mem::size_of::<T>(), self.item_size);
        assert!(index < self.len, "column get_mut index out of bounds");
        unsafe { &mut *(self.get_raw_mut(index) as *mut T) }
    }

    /// Get a raw slice to all items as `[T]`.
    ///
    /// # Safety
    ///
    /// The column must actually contain items of type `T`.
    pub unsafe fn as_slice<T: 'static>(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.data as *const T, self.len) }
    }

    /// Get a raw mutable slice to all items as `[T]`.
    ///
    /// # Safety
    ///
    /// The column must actually contain items of type `T`.
    pub unsafe fn as_slice_mut<T: 'static>(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.data as *mut T, self.len) }
    }

    /// Clear all items, dropping each one.
    pub fn clear(&mut self) {
        if self.item_size > 0 {
            for i in 0..self.len {
                unsafe {
                    let ptr = self.data.add(i * self.item_size);
                    (self.drop_fn)(ptr);
                }
            }
        }
        self.len = 0;
    }

    // -- Internal allocation ------------------------------------------------

    fn layout_for_capacity(&self, cap: usize) -> Option<Layout> {
        if self.item_size == 0 || cap == 0 {
            return None;
        }
        let size = self.item_size * cap;
        Layout::from_size_align(size, self.item_align).ok()
    }

    fn grow(&mut self, new_capacity: usize) {
        assert!(new_capacity > self.capacity);

        if self.item_size == 0 {
            self.capacity = new_capacity;
            return;
        }

        let new_layout = self
            .layout_for_capacity(new_capacity)
            .expect("invalid layout");

        let new_data = if self.data.is_null() {
            unsafe { alloc::alloc(new_layout) }
        } else {
            let old_layout = self
                .layout_for_capacity(self.capacity)
                .expect("invalid old layout");
            unsafe { alloc::realloc(self.data, old_layout, new_layout.size()) }
        };

        if new_data.is_null() {
            alloc::handle_alloc_error(new_layout);
        }

        self.data = new_data;
        self.capacity = new_capacity;
    }
}

impl Drop for ComponentColumn {
    fn drop(&mut self) {
        // Drop all stored items.
        self.clear();

        // Free the allocation.
        if !self.data.is_null() {
            if let Some(layout) = self.layout_for_capacity(self.capacity) {
                unsafe {
                    alloc::dealloc(self.data, layout);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ArchetypeEdge — cached archetype transitions
// ---------------------------------------------------------------------------

/// Pre-computed archetype transition edges.
///
/// When adding or removing a component, the entity moves from one archetype to
/// another. These edges are cached so we do not need to re-compute the target
/// archetype on every structural change.
#[derive(Debug, Clone, Default)]
pub struct ArchetypeEdges {
    /// `add_component[C] = target_archetype` — adding component C leads here.
    pub add: HashMap<ComponentId, ArchetypeId>,
    /// `remove_component[C] = target_archetype` — removing component C leads here.
    pub remove: HashMap<ComponentId, ArchetypeId>,
}

// ---------------------------------------------------------------------------
// Archetype
// ---------------------------------------------------------------------------

/// An archetype owns the component data for all entities that share the exact
/// same set of component types.
///
/// Components are stored column-major: one `ComponentColumn` per component type.
/// Entities form a dense array; the row index of an entity in the `entities`
/// vec is also its row index in each column.
pub struct Archetype {
    /// Unique identifier for this archetype.
    id: ArchetypeId,
    /// Sorted list of component types present in this archetype.
    component_types: Vec<ComponentId>,
    /// Metadata for each component type (parallel to `component_types`).
    component_infos: Vec<ComponentInfo>,
    /// Dense array of entities currently stored in this archetype.
    pub(crate) entities: Vec<Entity>,
    /// Column storage for each component type (parallel to `component_types`).
    pub(crate) columns: Vec<ComponentColumn>,
    /// Map from entity to its row index within this archetype.
    pub(crate) entity_to_row: HashMap<Entity, usize>,
    /// Cached archetype transition edges.
    edges: ArchetypeEdges,
}

impl Archetype {
    /// Create a new, empty archetype with the given id and sorted component
    /// set. Component infos must be parallel to `component_types` (same order).
    pub fn new(
        id: ArchetypeId,
        mut component_types: Vec<ComponentId>,
        mut component_infos: Vec<ComponentInfo>,
    ) -> Self {
        // Sort by TypeId so binary search works.
        let mut indices: Vec<usize> = (0..component_types.len()).collect();
        indices.sort_by_key(|&i| component_types[i].type_id());

        let sorted_types: Vec<ComponentId> = indices.iter().map(|&i| component_types[i]).collect();
        let sorted_infos: Vec<ComponentInfo> = indices.iter().map(|&i| component_infos[i]).collect();

        component_types = sorted_types;
        component_infos = sorted_infos;

        // Dedup (shouldn't happen, but defensive).
        let mut deduped_types = Vec::with_capacity(component_types.len());
        let mut deduped_infos = Vec::with_capacity(component_infos.len());
        for i in 0..component_types.len() {
            if i == 0 || component_types[i].type_id() != component_types[i - 1].type_id() {
                deduped_types.push(component_types[i]);
                deduped_infos.push(component_infos[i]);
            }
        }

        let columns = deduped_infos.iter().map(|info| ComponentColumn::new(info)).collect();

        Self {
            id,
            component_types: deduped_types,
            component_infos: deduped_infos,
            entities: Vec::new(),
            columns,
            entity_to_row: HashMap::new(),
            edges: ArchetypeEdges::default(),
        }
    }

    /// Create the empty archetype (no component types).
    pub fn empty(id: ArchetypeId) -> Self {
        Self {
            id,
            component_types: Vec::new(),
            component_infos: Vec::new(),
            entities: Vec::new(),
            columns: Vec::new(),
            entity_to_row: HashMap::new(),
            edges: ArchetypeEdges::default(),
        }
    }

    // -- Accessors ----------------------------------------------------------

    /// The archetype's unique id.
    #[inline]
    pub fn id(&self) -> ArchetypeId {
        self.id
    }

    /// The sorted set of component types.
    #[inline]
    pub fn component_types(&self) -> &[ComponentId] {
        &self.component_types
    }

    /// Component metadata, parallel to `component_types`.
    #[inline]
    pub fn component_infos(&self) -> &[ComponentInfo] {
        &self.component_infos
    }

    /// Dense entity array.
    #[inline]
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Number of entities stored in this archetype.
    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether the archetype is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Returns `true` if this archetype contains the given component type.
    pub fn has_component(&self, id: ComponentId) -> bool {
        self.column_index(id).is_some()
    }

    /// Find the column index for a component type via binary search.
    #[inline]
    pub fn column_index(&self, id: ComponentId) -> Option<usize> {
        self.component_types
            .binary_search_by_key(&id.type_id(), |c| c.type_id())
            .ok()
    }

    /// Get a reference to a column by its index.
    #[inline]
    pub fn column(&self, col_idx: usize) -> &ComponentColumn {
        &self.columns[col_idx]
    }

    /// Get a mutable reference to a column by its index.
    #[inline]
    pub fn column_mut(&mut self, col_idx: usize) -> &mut ComponentColumn {
        &mut self.columns[col_idx]
    }

    /// Get the column for a specific component type.
    pub fn column_for(&self, id: ComponentId) -> Option<&ComponentColumn> {
        self.column_index(id).map(|idx| &self.columns[idx])
    }

    /// Get a mutable column for a specific component type.
    pub fn column_for_mut(&mut self, id: ComponentId) -> Option<&mut ComponentColumn> {
        self.column_index(id).map(|idx| &mut self.columns[idx])
    }

    /// Get the row index for an entity.
    #[inline]
    pub fn entity_row(&self, entity: Entity) -> Option<usize> {
        self.entity_to_row.get(&entity).copied()
    }

    /// Mutable access to the edges cache.
    #[inline]
    pub fn edges_mut(&mut self) -> &mut ArchetypeEdges {
        &mut self.edges
    }

    /// Immutable access to the edges cache.
    #[inline]
    pub fn edges(&self) -> &ArchetypeEdges {
        &self.edges
    }

    // -- Structural operations ----------------------------------------------

    /// Add an entity to this archetype. The caller must supply the raw
    /// component data in the same order as `component_types()`.
    ///
    /// # Safety
    ///
    /// Each pointer in `component_data` must point to valid, initialized data
    /// of the type corresponding to the same index in `component_types`. The
    /// pointed-to values are memcpy'd — the caller must `forget` them.
    pub unsafe fn add_entity_raw(
        &mut self,
        entity: Entity,
        component_data: &[*const u8],
    ) {
        debug_assert_eq!(component_data.len(), self.columns.len());

        let row = self.entities.len();
        self.entities.push(entity);
        self.entity_to_row.insert(entity, row);

        for (col, &data_ptr) in self.columns.iter_mut().zip(component_data.iter()) {
            unsafe { col.push_raw(data_ptr); }
        }
    }

    /// Add a single entity with a single typed component. Used by the World for
    /// the simple case of spawning into a single-component archetype.
    pub fn add_entity_single<T: 'static>(&mut self, entity: Entity, component: T) {
        debug_assert_eq!(self.columns.len(), 1);

        let row = self.entities.len();
        self.entities.push(entity);
        self.entity_to_row.insert(entity, row);
        self.columns[0].push(component);
    }

    /// Remove an entity from this archetype using swap-remove semantics.
    /// The removed entity's component data is dropped.
    ///
    /// Returns the entity that was swapped into the removed slot (if any), so
    /// the caller can update the entity-to-archetype mapping. Returns `None` if
    /// the removed entity was the last one.
    pub fn remove_entity(&mut self, entity: Entity) -> Option<Entity> {
        let row = match self.entity_to_row.remove(&entity) {
            Some(r) => r,
            None => return None,
        };

        let was_last = row == self.entities.len() - 1;

        // Swap-remove in entity array.
        self.entities.swap_remove(row);

        // Swap-remove in each column (drops the removed component).
        for col in &mut self.columns {
            unsafe { col.swap_remove_and_drop(row); }
        }

        if was_last {
            None
        } else {
            // The entity that was at the end is now at `row`.
            let swapped = self.entities[row];
            self.entity_to_row.insert(swapped, row);
            Some(swapped)
        }
    }

    /// Swap-remove an entity from this archetype, writing the removed
    /// component data into raw byte buffers. Does NOT drop the component data.
    ///
    /// `out_buffers` must have one entry per column, each pointing to at least
    /// `item_size` bytes of writable memory.
    ///
    /// Returns the entity that was swapped into the removed slot, if any.
    ///
    /// # Safety
    ///
    /// - `out_buffers` must be correctly sized.
    /// - The caller takes ownership of the raw bytes and must either drop them
    ///   manually or move them into another archetype.
    pub unsafe fn remove_entity_raw(
        &mut self,
        entity: Entity,
        out_buffers: &mut [*mut u8],
    ) -> Option<Entity> {
        let row = match self.entity_to_row.remove(&entity) {
            Some(r) => r,
            None => return None,
        };

        let was_last = row == self.entities.len() - 1;
        self.entities.swap_remove(row);

        debug_assert_eq!(out_buffers.len(), self.columns.len());
        for (col, out) in self.columns.iter_mut().zip(out_buffers.iter()) {
            unsafe { col.swap_remove_raw(row, *out); }
        }

        if was_last {
            None
        } else {
            let swapped = self.entities[row];
            self.entity_to_row.insert(swapped, row);
            Some(swapped)
        }
    }

    /// Move an entity from this archetype to another. For each component type
    /// present in both archetypes, the data is moved (memcpy, no drop).
    ///
    /// Returns the entity that was swapped into the removed slot in `self`
    /// (if any) and the row index in the target archetype.
    ///
    /// Additional component data (for a newly added component) must be pushed
    /// into the target archetype by the caller after this returns.
    ///
    /// # Safety
    ///
    /// Both archetypes must be valid and the entity must exist in `self`.
    pub unsafe fn move_entity_to(
        &mut self,
        entity: Entity,
        target: &mut Archetype,
    ) -> (Option<Entity>, usize) {
        let row = self
            .entity_to_row
            .remove(&entity)
            .expect("entity not in source archetype");

        let was_last = row == self.entities.len() - 1;
        self.entities.swap_remove(row);

        // The target row.
        let target_row = target.entities.len();
        target.entities.push(entity);
        target.entity_to_row.insert(entity, target_row);

        // For each column in the target archetype, find the matching source
        // column and move data.
        for (target_col_idx, target_comp_id) in
            target.component_types.iter().enumerate()
        {
            if let Some(src_col_idx) = self.column_index(*target_comp_id) {
                // This component exists in both source and target — move it.
                unsafe {
                    self.columns[src_col_idx]
                        .swap_remove_to(row, &mut target.columns[target_col_idx]);
                }
            }
            // If the component only exists in the target, the caller must push
            // it manually (e.g., for add_component).
        }

        // Drop columns in source that don't exist in target (they were removed).
        for (src_col_idx, src_comp_id) in self.component_types.iter().enumerate()
        {
            if target.column_index(*src_comp_id).is_none() {
                unsafe {
                    // This component was removed — drop it.
                    let ptr = self.columns[src_col_idx].get_raw(row);
                    (self.columns[src_col_idx].drop_fn)(ptr as *mut u8);
                    // Now swap-remove the raw entry.
                    self.columns[src_col_idx].len -= 1;
                    if row != self.columns[src_col_idx].len && self.columns[src_col_idx].item_size > 0
                    {
                        let last_ptr = self.columns[src_col_idx]
                            .data
                            .add(self.columns[src_col_idx].len * self.columns[src_col_idx].item_size);
                        let dst_ptr = self.columns[src_col_idx]
                            .data
                            .add(row * self.columns[src_col_idx].item_size);
                        ptr::copy_nonoverlapping(
                            last_ptr,
                            dst_ptr,
                            self.columns[src_col_idx].item_size,
                        );
                    }
                }
            }
        }

        let swapped = if was_last {
            None
        } else {
            let swapped = self.entities[row];
            self.entity_to_row.insert(swapped, row);
            Some(swapped)
        };

        (swapped, target_row)
    }

    /// Get a typed component for an entity in this archetype.
    pub fn get_component<T: 'static>(&self, entity: Entity) -> Option<&T> {
        let row = self.entity_to_row.get(&entity)?;
        let comp_id = ComponentId::of_raw(TypeId::of::<T>());
        let col_idx = self.column_index(comp_id)?;
        Some(self.columns[col_idx].get::<T>(*row))
    }

    /// Get a mutable typed component for an entity in this archetype.
    pub fn get_component_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        let row = *self.entity_to_row.get(&entity)?;
        let comp_id = ComponentId::of_raw(TypeId::of::<T>());
        let col_idx = self.column_index(comp_id)?;
        Some(self.columns[col_idx].get_mut::<T>(row))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Health(f32);

    fn pos_info() -> ComponentInfo {
        ComponentInfo::of::<Position>()
    }
    fn vel_info() -> ComponentInfo {
        ComponentInfo::of::<Velocity>()
    }
    fn hp_info() -> ComponentInfo {
        ComponentInfo::of::<Health>()
    }

    fn make_archetype(id: u32, infos: &[ComponentInfo]) -> Archetype {
        let types: Vec<ComponentId> = infos.iter().map(|i| i.id).collect();
        let info_vec: Vec<ComponentInfo> = infos.to_vec();
        Archetype::new(ArchetypeId(id), types, info_vec)
    }

    #[test]
    fn column_push_and_get() {
        let info = pos_info();
        let mut col = ComponentColumn::new(&info);
        col.push(Position { x: 1.0, y: 2.0 });
        col.push(Position { x: 3.0, y: 4.0 });
        assert_eq!(col.len(), 2);
        assert_eq!(col.get::<Position>(0), &Position { x: 1.0, y: 2.0 });
        assert_eq!(col.get::<Position>(1), &Position { x: 3.0, y: 4.0 });
    }

    #[test]
    fn column_get_mut() {
        let info = pos_info();
        let mut col = ComponentColumn::new(&info);
        col.push(Position { x: 1.0, y: 2.0 });
        col.get_mut::<Position>(0).x = 99.0;
        assert_eq!(col.get::<Position>(0).x, 99.0);
    }

    #[test]
    fn column_swap_remove_and_drop() {
        let info = pos_info();
        let mut col = ComponentColumn::new(&info);
        col.push(Position { x: 1.0, y: 0.0 });
        col.push(Position { x: 2.0, y: 0.0 });
        col.push(Position { x: 3.0, y: 0.0 });

        unsafe {
            col.swap_remove_and_drop(0);
        }
        assert_eq!(col.len(), 2);
        // The last element (x=3) should now be at index 0.
        assert_eq!(col.get::<Position>(0).x, 3.0);
        assert_eq!(col.get::<Position>(1).x, 2.0);
    }

    #[test]
    fn column_swap_remove_raw() {
        let info = pos_info();
        let mut col = ComponentColumn::new(&info);
        col.push(Position { x: 10.0, y: 20.0 });
        col.push(Position { x: 30.0, y: 40.0 });

        let mut out = std::mem::MaybeUninit::<Position>::uninit();
        unsafe {
            col.swap_remove_raw(0, out.as_mut_ptr() as *mut u8);
        }
        let removed = unsafe { out.assume_init() };
        assert_eq!(removed, Position { x: 10.0, y: 20.0 });
        assert_eq!(col.len(), 1);
        assert_eq!(col.get::<Position>(0), &Position { x: 30.0, y: 40.0 });
    }

    #[test]
    fn column_with_capacity() {
        let info = pos_info();
        let mut col = ComponentColumn::with_capacity(&info, 100);
        assert_eq!(col.len(), 0);
        col.push(Position { x: 1.0, y: 2.0 });
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn column_clear() {
        let info = pos_info();
        let mut col = ComponentColumn::new(&info);
        col.push(Position { x: 1.0, y: 0.0 });
        col.push(Position { x: 2.0, y: 0.0 });
        col.clear();
        assert_eq!(col.len(), 0);
        assert!(col.is_empty());
    }

    #[test]
    fn column_as_slice() {
        let info = pos_info();
        let mut col = ComponentColumn::new(&info);
        col.push(Position { x: 1.0, y: 2.0 });
        col.push(Position { x: 3.0, y: 4.0 });
        let slice = unsafe { col.as_slice::<Position>() };
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], Position { x: 1.0, y: 2.0 });
        assert_eq!(slice[1], Position { x: 3.0, y: 4.0 });
    }

    #[test]
    fn archetype_has_component() {
        let arch = make_archetype(1, &[pos_info(), vel_info()]);
        assert!(arch.has_component(pos_info().id));
        assert!(arch.has_component(vel_info().id));
        assert!(!arch.has_component(hp_info().id));
    }

    #[test]
    fn archetype_add_and_get_entity() {
        let mut arch = make_archetype(1, &[pos_info(), vel_info()]);
        let entity = Entity::new(0, 0);

        let pos = Position { x: 1.0, y: 2.0 };
        let vel = Velocity { dx: 3.0, dy: 4.0 };

        let pos_col = arch.column_index(pos_info().id).unwrap();
        let vel_col = arch.column_index(vel_info().id).unwrap();

        // We need to build the data pointers in the archetype's column order.
        let mut data_ptrs = vec![ptr::null(); 2];
        data_ptrs[pos_col] = &pos as *const Position as *const u8;
        data_ptrs[vel_col] = &vel as *const Velocity as *const u8;

        unsafe {
            arch.add_entity_raw(entity, &data_ptrs);
        }
        std::mem::forget(pos);
        std::mem::forget(vel);

        assert_eq!(arch.len(), 1);
        assert_eq!(
            arch.get_component::<Position>(entity),
            Some(&Position { x: 1.0, y: 2.0 })
        );
        assert_eq!(
            arch.get_component::<Velocity>(entity),
            Some(&Velocity { dx: 3.0, dy: 4.0 })
        );
    }

    #[test]
    fn archetype_remove_entity() {
        let mut arch = make_archetype(1, &[pos_info()]);
        let e0 = Entity::new(0, 0);
        let e1 = Entity::new(1, 0);
        let e2 = Entity::new(2, 0);

        arch.columns[0].push(Position { x: 0.0, y: 0.0 });
        arch.entities.push(e0);
        arch.entity_to_row.insert(e0, 0);

        arch.columns[0].push(Position { x: 1.0, y: 0.0 });
        arch.entities.push(e1);
        arch.entity_to_row.insert(e1, 1);

        arch.columns[0].push(Position { x: 2.0, y: 0.0 });
        arch.entities.push(e2);
        arch.entity_to_row.insert(e2, 2);

        // Remove e0 — e2 should swap into its slot.
        let swapped = arch.remove_entity(e0);
        assert_eq!(swapped, Some(e2));
        assert_eq!(arch.len(), 2);
        assert_eq!(arch.entity_row(e2), Some(0));
        assert_eq!(arch.entity_row(e1), Some(1));
        assert_eq!(
            arch.get_component::<Position>(e2),
            Some(&Position { x: 2.0, y: 0.0 })
        );
    }

    #[test]
    fn archetype_remove_last_entity() {
        let mut arch = make_archetype(1, &[pos_info()]);
        let e0 = Entity::new(0, 0);
        arch.columns[0].push(Position { x: 5.0, y: 5.0 });
        arch.entities.push(e0);
        arch.entity_to_row.insert(e0, 0);

        let swapped = arch.remove_entity(e0);
        assert_eq!(swapped, None);
        assert_eq!(arch.len(), 0);
    }

    #[test]
    fn archetype_empty_has_no_components() {
        let arch = Archetype::empty(ArchetypeId::EMPTY);
        assert_eq!(arch.component_types().len(), 0);
        assert_eq!(arch.len(), 0);
    }

    #[test]
    fn archetype_sorts_and_dedups_types() {
        let info_a = pos_info();
        let info_b = vel_info();
        // Pass in reverse order — should be sorted.
        let arch = Archetype::new(
            ArchetypeId(1),
            vec![info_b.id, info_a.id],
            vec![info_b, info_a],
        );
        assert_eq!(arch.component_types().len(), 2);
        // Should be sorted by TypeId.
        assert!(
            arch.component_types()[0].type_id() < arch.component_types()[1].type_id()
                || arch.component_types()[0].type_id() > arch.component_types()[1].type_id()
                || arch.component_types()[0] == arch.component_types()[1]
        );
    }

    #[test]
    fn column_zst_component() {
        // Zero-sized types should work.
        #[derive(Debug, Clone, PartialEq)]
        struct Marker;
        let info = ComponentInfo::of::<Marker>();
        assert_eq!(info.size, 0);
        let mut col = ComponentColumn::new(&info);
        col.push(Marker);
        col.push(Marker);
        assert_eq!(col.len(), 2);
        unsafe {
            col.swap_remove_and_drop(0);
        }
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn column_large_growth() {
        let info = pos_info();
        let mut col = ComponentColumn::new(&info);
        for i in 0..1000 {
            col.push(Position {
                x: i as f32,
                y: 0.0,
            });
        }
        assert_eq!(col.len(), 1000);
        assert_eq!(col.get::<Position>(500).x, 500.0);
        assert_eq!(col.get::<Position>(999).x, 999.0);
    }

    #[test]
    fn archetype_edges_default() {
        let arch = Archetype::empty(ArchetypeId(0));
        assert!(arch.edges().add.is_empty());
        assert!(arch.edges().remove.is_empty());
    }

    #[test]
    fn archetype_get_component_mut() {
        let mut arch = make_archetype(1, &[pos_info()]);
        let e = Entity::new(0, 0);
        arch.columns[0].push(Position { x: 1.0, y: 2.0 });
        arch.entities.push(e);
        arch.entity_to_row.insert(e, 0);

        arch.get_component_mut::<Position>(e).unwrap().x = 99.0;
        assert_eq!(arch.get_component::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    fn column_swap_remove_to() {
        let info = pos_info();
        let mut src = ComponentColumn::new(&info);
        let mut dst = ComponentColumn::new(&info);

        src.push(Position { x: 1.0, y: 0.0 });
        src.push(Position { x: 2.0, y: 0.0 });
        src.push(Position { x: 3.0, y: 0.0 });

        unsafe {
            src.swap_remove_to(0, &mut dst);
        }
        assert_eq!(src.len(), 2);
        assert_eq!(dst.len(), 1);
        assert_eq!(dst.get::<Position>(0), &Position { x: 1.0, y: 0.0 });
        // The last element (x=3) should now be at index 0 in src.
        assert_eq!(src.get::<Position>(0), &Position { x: 3.0, y: 0.0 });
    }
}
