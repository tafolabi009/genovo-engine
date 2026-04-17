//! UI Data Binding System
//!
//! Provides reactive data binding between UI elements and application data.
//! Supports observable properties, one-way and two-way binding, computed
//! (derived) properties, collection binding for list views, binding converters,
//! and binding validation.
//!
//! # Architecture
//!
//! ```text
//!  ObservableProperty<T>  <-->  Binding  <-->  UI Widget
//!         ^                       |
//!         |                  BindingConverter
//!    ComputedProperty             |
//!                          BindingValidator
//! ```
//!
//! # Example
//!
//! ```ignore
//! use genovo_ui::data_binding::*;
//!
//! let mut ctx = BindingContext::new();
//! let health = ctx.create_property("health", 100.0f64);
//! let health_text = ctx.create_computed("health_text", move |ctx| {
//!     format!("HP: {:.0}", ctx.get::<f64>(&health).unwrap_or(0.0))
//! });
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// PropertyId
// ---------------------------------------------------------------------------

/// Unique identifier for an observable property within a [`BindingContext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PropertyId {
    /// Internal index.
    pub index: u64,
}

impl PropertyId {
    /// Creates a new property identifier from a raw index.
    pub fn from_raw(index: u64) -> Self {
        Self { index }
    }

    /// Returns the raw index value.
    pub fn raw(&self) -> u64 {
        self.index
    }
}

impl fmt::Display for PropertyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PropertyId({})", self.index)
    }
}

/// Global counter for generating unique property IDs.
static NEXT_PROPERTY_ID: AtomicU64 = AtomicU64::new(1);

/// Generates a new unique property identifier.
fn next_property_id() -> PropertyId {
    PropertyId {
        index: NEXT_PROPERTY_ID.fetch_add(1, Ordering::Relaxed),
    }
}

// ---------------------------------------------------------------------------
// BindingId
// ---------------------------------------------------------------------------

/// Unique identifier for a binding relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindingId {
    /// Internal index.
    pub index: u64,
}

impl BindingId {
    /// Creates a new binding identifier from a raw index.
    pub fn from_raw(index: u64) -> Self {
        Self { index }
    }
}

impl fmt::Display for BindingId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BindingId({})", self.index)
    }
}

/// Global counter for generating unique binding IDs.
static NEXT_BINDING_ID: AtomicU64 = AtomicU64::new(1);

/// Generates a new unique binding identifier.
fn next_binding_id() -> BindingId {
    BindingId {
        index: NEXT_BINDING_ID.fetch_add(1, Ordering::Relaxed),
    }
}

// ---------------------------------------------------------------------------
// BindingDirection
// ---------------------------------------------------------------------------

/// Specifies how data flows through a binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingDirection {
    /// Source changes propagate to target, but not vice versa.
    OneWay,
    /// Changes propagate in both directions between source and target.
    TwoWay,
    /// Value is read from source once when the binding is created.
    OneTime,
}

impl Default for BindingDirection {
    fn default() -> Self {
        Self::OneWay
    }
}

impl fmt::Display for BindingDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OneWay => write!(f, "OneWay"),
            Self::TwoWay => write!(f, "TwoWay"),
            Self::OneTime => write!(f, "OneTime"),
        }
    }
}

// ---------------------------------------------------------------------------
// BindingMode
// ---------------------------------------------------------------------------

/// Controls when binding updates are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingMode {
    /// Updates propagate immediately when the source value changes.
    Immediate,
    /// Updates are batched and applied during the next binding context tick.
    Deferred,
    /// Updates are applied only when explicitly requested.
    Explicit,
}

impl Default for BindingMode {
    fn default() -> Self {
        Self::Immediate
    }
}

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

/// Result of validating a property value before it is accepted.
#[derive(Debug, Clone)]
pub enum ValidationResult {
    /// Value is valid and should be accepted.
    Valid,
    /// Value is invalid with an error message.
    Invalid(String),
    /// Value is valid but a warning should be shown to the user.
    Warning(String),
    /// Value should be coerced to a corrected form.
    Coerced {
        /// The corrected value description.
        message: String,
    },
}

impl ValidationResult {
    /// Returns `true` if the validation passed (either valid or warning).
    pub fn is_acceptable(&self) -> bool {
        matches!(self, Self::Valid | Self::Warning(_) | Self::Coerced { .. })
    }

    /// Returns `true` if the validation failed.
    pub fn is_invalid(&self) -> bool {
        matches!(self, Self::Invalid(_))
    }

    /// Returns the error or warning message, if any.
    pub fn message(&self) -> Option<&str> {
        match self {
            Self::Valid => None,
            Self::Invalid(msg) => Some(msg),
            Self::Warning(msg) => Some(msg),
            Self::Coerced { message } => Some(message),
        }
    }
}

// ---------------------------------------------------------------------------
// BindingValidator
// ---------------------------------------------------------------------------

/// Validates values before they are accepted by a property.
///
/// Validators can reject values, warn about values, or coerce values to
/// a corrected form. Multiple validators can be chained on a single property.
pub struct BindingValidator {
    /// Human-readable name for this validator.
    pub name: String,
    /// The validation function. Takes a reference to the value as `&dyn Any`
    /// and returns a validation result.
    validator_fn: Box<dyn Fn(&dyn Any) -> ValidationResult + Send + Sync>,
}

impl BindingValidator {
    /// Creates a new validator with the given name and validation function.
    pub fn new<F>(name: impl Into<String>, validator_fn: F) -> Self
    where
        F: Fn(&dyn Any) -> ValidationResult + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            validator_fn: Box::new(validator_fn),
        }
    }

    /// Creates a range validator for numeric types.
    pub fn range_f64(name: impl Into<String>, min: f64, max: f64) -> Self {
        let name_str = name.into();
        Self {
            name: name_str.clone(),
            validator_fn: Box::new(move |value: &dyn Any| {
                if let Some(v) = value.downcast_ref::<f64>() {
                    if *v < min {
                        ValidationResult::Invalid(format!(
                            "{}: value {} is below minimum {}",
                            name_str, v, min
                        ))
                    } else if *v > max {
                        ValidationResult::Invalid(format!(
                            "{}: value {} is above maximum {}",
                            name_str, v, max
                        ))
                    } else {
                        ValidationResult::Valid
                    }
                } else {
                    ValidationResult::Invalid("Expected f64 value".to_string())
                }
            }),
        }
    }

    /// Creates a range validator for integer types.
    pub fn range_i64(name: impl Into<String>, min: i64, max: i64) -> Self {
        let name_str = name.into();
        Self {
            name: name_str.clone(),
            validator_fn: Box::new(move |value: &dyn Any| {
                if let Some(v) = value.downcast_ref::<i64>() {
                    if *v < min {
                        ValidationResult::Invalid(format!(
                            "{}: value {} is below minimum {}",
                            name_str, v, min
                        ))
                    } else if *v > max {
                        ValidationResult::Invalid(format!(
                            "{}: value {} is above maximum {}",
                            name_str, v, max
                        ))
                    } else {
                        ValidationResult::Valid
                    }
                } else {
                    ValidationResult::Invalid("Expected i64 value".to_string())
                }
            }),
        }
    }

    /// Creates a non-empty string validator.
    pub fn non_empty_string(name: impl Into<String>) -> Self {
        let name_str = name.into();
        Self {
            name: name_str.clone(),
            validator_fn: Box::new(move |value: &dyn Any| {
                if let Some(s) = value.downcast_ref::<String>() {
                    if s.is_empty() {
                        ValidationResult::Invalid(format!("{}: string must not be empty", name_str))
                    } else {
                        ValidationResult::Valid
                    }
                } else {
                    ValidationResult::Invalid("Expected String value".to_string())
                }
            }),
        }
    }

    /// Creates a max-length string validator.
    pub fn max_length(name: impl Into<String>, max_len: usize) -> Self {
        let name_str = name.into();
        Self {
            name: name_str.clone(),
            validator_fn: Box::new(move |value: &dyn Any| {
                if let Some(s) = value.downcast_ref::<String>() {
                    if s.len() > max_len {
                        ValidationResult::Invalid(format!(
                            "{}: string length {} exceeds maximum {}",
                            name_str,
                            s.len(),
                            max_len
                        ))
                    } else {
                        ValidationResult::Valid
                    }
                } else {
                    ValidationResult::Valid
                }
            }),
        }
    }

    /// Runs the validator against a value.
    pub fn validate(&self, value: &dyn Any) -> ValidationResult {
        (self.validator_fn)(value)
    }
}

impl fmt::Debug for BindingValidator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BindingValidator")
            .field("name", &self.name)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// BindingConverter
// ---------------------------------------------------------------------------

/// Converts values between source and target types in a binding.
///
/// Converters are used when the source property type differs from what the
/// target expects. For example, converting a numeric health value to a
/// formatted string like "HP: 100".
pub struct BindingConverter {
    /// Human-readable name for this converter.
    pub name: String,
    /// Converts from source type to target type.
    convert_fn: Box<dyn Fn(&dyn Any) -> Option<Box<dyn Any + Send + Sync>> + Send + Sync>,
    /// Converts from target type back to source type (for two-way bindings).
    convert_back_fn:
        Option<Box<dyn Fn(&dyn Any) -> Option<Box<dyn Any + Send + Sync>> + Send + Sync>>,
}

impl BindingConverter {
    /// Creates a one-way converter.
    pub fn one_way<F>(name: impl Into<String>, convert_fn: F) -> Self
    where
        F: Fn(&dyn Any) -> Option<Box<dyn Any + Send + Sync>> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            convert_fn: Box::new(convert_fn),
            convert_back_fn: None,
        }
    }

    /// Creates a two-way converter with both forward and backward conversion.
    pub fn two_way<F, G>(name: impl Into<String>, convert_fn: F, convert_back_fn: G) -> Self
    where
        F: Fn(&dyn Any) -> Option<Box<dyn Any + Send + Sync>> + Send + Sync + 'static,
        G: Fn(&dyn Any) -> Option<Box<dyn Any + Send + Sync>> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            convert_fn: Box::new(convert_fn),
            convert_back_fn: Some(Box::new(convert_back_fn)),
        }
    }

    /// Creates a converter that formats an f64 as a currency string.
    pub fn currency(currency_symbol: &str, decimal_places: usize) -> Self {
        let symbol = currency_symbol.to_string();
        let symbol_back = symbol.clone();
        Self::two_way(
            format!("CurrencyConverter({})", symbol),
            move |value: &dyn Any| {
                if let Some(v) = value.downcast_ref::<f64>() {
                    let formatted = format!("{}{:.prec$}", symbol, v, prec = decimal_places);
                    Some(Box::new(formatted) as Box<dyn Any + Send + Sync>)
                } else {
                    None
                }
            },
            move |value: &dyn Any| {
                if let Some(s) = value.downcast_ref::<String>() {
                    let cleaned = s.replace(&symbol_back, "").replace(',', "").trim().to_string();
                    if let Ok(v) = cleaned.parse::<f64>() {
                        Some(Box::new(v) as Box<dyn Any + Send + Sync>)
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
        )
    }

    /// Creates a converter that formats an f64 as a percentage string.
    pub fn percentage(decimal_places: usize) -> Self {
        Self::two_way(
            "PercentageConverter",
            move |value: &dyn Any| {
                if let Some(v) = value.downcast_ref::<f64>() {
                    let formatted = format!("{:.prec$}%", v * 100.0, prec = decimal_places);
                    Some(Box::new(formatted) as Box<dyn Any + Send + Sync>)
                } else {
                    None
                }
            },
            move |value: &dyn Any| {
                if let Some(s) = value.downcast_ref::<String>() {
                    let cleaned = s.replace('%', "").trim().to_string();
                    if let Ok(v) = cleaned.parse::<f64>() {
                        Some(Box::new(v / 100.0) as Box<dyn Any + Send + Sync>)
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
        )
    }

    /// Creates a boolean-to-string converter ("true"/"false").
    pub fn bool_to_string() -> Self {
        Self::two_way(
            "BoolToStringConverter",
            |value: &dyn Any| {
                if let Some(v) = value.downcast_ref::<bool>() {
                    Some(Box::new(v.to_string()) as Box<dyn Any + Send + Sync>)
                } else {
                    None
                }
            },
            |value: &dyn Any| {
                if let Some(s) = value.downcast_ref::<String>() {
                    match s.to_lowercase().as_str() {
                        "true" | "1" | "yes" => {
                            Some(Box::new(true) as Box<dyn Any + Send + Sync>)
                        }
                        "false" | "0" | "no" => {
                            Some(Box::new(false) as Box<dyn Any + Send + Sync>)
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            },
        )
    }

    /// Creates a converter that maps an f64 to a string via a formatting function.
    pub fn f64_to_string(format_fn: impl Fn(f64) -> String + Send + Sync + 'static) -> Self {
        Self::one_way("F64ToStringConverter", move |value: &dyn Any| {
            if let Some(v) = value.downcast_ref::<f64>() {
                Some(Box::new(format_fn(*v)) as Box<dyn Any + Send + Sync>)
            } else {
                None
            }
        })
    }

    /// Creates a converter that inverts a boolean value.
    pub fn bool_invert() -> Self {
        Self::two_way(
            "BoolInvertConverter",
            |value: &dyn Any| {
                if let Some(v) = value.downcast_ref::<bool>() {
                    Some(Box::new(!*v) as Box<dyn Any + Send + Sync>)
                } else {
                    None
                }
            },
            |value: &dyn Any| {
                if let Some(v) = value.downcast_ref::<bool>() {
                    Some(Box::new(!*v) as Box<dyn Any + Send + Sync>)
                } else {
                    None
                }
            },
        )
    }

    /// Applies the forward conversion.
    pub fn convert(&self, value: &dyn Any) -> Option<Box<dyn Any + Send + Sync>> {
        (self.convert_fn)(value)
    }

    /// Applies the backward conversion (for two-way bindings).
    pub fn convert_back(&self, value: &dyn Any) -> Option<Box<dyn Any + Send + Sync>> {
        self.convert_back_fn.as_ref().and_then(|f| f(value))
    }

    /// Returns `true` if this converter supports backward conversion.
    pub fn supports_convert_back(&self) -> bool {
        self.convert_back_fn.is_some()
    }
}

impl fmt::Debug for BindingConverter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BindingConverter")
            .field("name", &self.name)
            .field("supports_convert_back", &self.supports_convert_back())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ChangeNotification
// ---------------------------------------------------------------------------

/// Describes a change to an observable property.
#[derive(Debug, Clone)]
pub struct ChangeNotification {
    /// The property that changed.
    pub property_id: PropertyId,
    /// Name of the property that changed.
    pub property_name: String,
    /// Timestamp of the change (frame number or monotonic counter).
    pub change_id: u64,
}

// ---------------------------------------------------------------------------
// ObservableProperty (type-erased storage)
// ---------------------------------------------------------------------------

/// Type-erased storage for an observable property value.
struct PropertyStorage {
    /// The current value (type-erased).
    value: Box<dyn Any + Send + Sync>,
    /// The TypeId of the stored value.
    type_id: TypeId,
    /// Human-readable name.
    name: String,
    /// Validators applied before setting a new value.
    validators: Vec<BindingValidator>,
    /// Monotonically increasing version counter; incremented on each change.
    version: u64,
    /// Whether change notifications are currently suppressed.
    notifications_suppressed: bool,
}

impl PropertyStorage {
    /// Creates new property storage with an initial value.
    fn new<T: Any + Send + Sync + Clone + 'static>(name: impl Into<String>, value: T) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            value: Box::new(value),
            name: name.into(),
            validators: Vec::new(),
            version: 0,
            notifications_suppressed: false,
        }
    }

    /// Returns a reference to the stored value, downcast to the requested type.
    fn get<T: Any + 'static>(&self) -> Option<&T> {
        self.value.downcast_ref::<T>()
    }

    /// Attempts to set the value, running validators first.
    /// Returns the validation results and whether the value was actually changed.
    fn set<T: Any + Send + Sync + Clone + PartialEq + 'static>(
        &mut self,
        new_value: T,
    ) -> (Vec<ValidationResult>, bool) {
        // Run validators.
        let mut results = Vec::new();
        for validator in &self.validators {
            let result = validator.validate(&new_value);
            if result.is_invalid() {
                results.push(result);
                return (results, false);
            }
            results.push(result);
        }

        // Check type compatibility.
        if TypeId::of::<T>() != self.type_id {
            results.push(ValidationResult::Invalid(format!(
                "Type mismatch: expected {:?}, got {:?}",
                self.type_id,
                TypeId::of::<T>()
            )));
            return (results, false);
        }

        // Check if value actually changed.
        if let Some(current) = self.value.downcast_ref::<T>() {
            if *current == new_value {
                return (results, false);
            }
        }

        self.value = Box::new(new_value);
        self.version += 1;
        (results, true)
    }

    /// Sets the value from a type-erased box, without running validators.
    fn set_any(&mut self, new_value: Box<dyn Any + Send + Sync>) {
        self.value = new_value;
        self.version += 1;
    }
}

impl fmt::Debug for PropertyStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PropertyStorage")
            .field("name", &self.name)
            .field("version", &self.version)
            .field("validator_count", &self.validators.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ComputedPropertyDef
// ---------------------------------------------------------------------------

/// Definition of a computed property that derives its value from other
/// properties.
struct ComputedPropertyDef {
    /// The property ID where the computed result is stored.
    target_id: PropertyId,
    /// The IDs of source properties this computed property depends on.
    dependencies: Vec<PropertyId>,
    /// The computation function. Takes a reference to the binding context
    /// accessor and returns a boxed value.
    compute_fn: Box<dyn Fn(&PropertyAccessor) -> Box<dyn Any + Send + Sync> + Send + Sync>,
    /// Whether this computed property needs recomputation.
    dirty: bool,
    /// The last version of each dependency when this was computed.
    dependency_versions: Vec<u64>,
}

impl fmt::Debug for ComputedPropertyDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputedPropertyDef")
            .field("target_id", &self.target_id)
            .field("dependencies", &self.dependencies)
            .field("dirty", &self.dirty)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PropertyAccessor
// ---------------------------------------------------------------------------

/// Read-only accessor for property values during computed property evaluation.
///
/// This provides a safe way to read property values without allowing mutation
/// during the computation phase.
pub struct PropertyAccessor<'a> {
    properties: &'a HashMap<PropertyId, PropertyStorage>,
}

impl<'a> PropertyAccessor<'a> {
    /// Gets a property value by its ID.
    pub fn get<T: Any + 'static>(&self, id: &PropertyId) -> Option<&T> {
        self.properties.get(id).and_then(|s| s.get::<T>())
    }

    /// Gets the version of a property.
    pub fn version(&self, id: &PropertyId) -> Option<u64> {
        self.properties.get(id).map(|s| s.version)
    }

    /// Returns `true` if a property exists.
    pub fn exists(&self, id: &PropertyId) -> bool {
        self.properties.contains_key(id)
    }

    /// Gets the name of a property.
    pub fn name(&self, id: &PropertyId) -> Option<&str> {
        self.properties.get(id).map(|s| s.name.as_str())
    }
}

// ---------------------------------------------------------------------------
// BindingDef
// ---------------------------------------------------------------------------

/// Definition of a binding between a source and target property.
struct BindingDef {
    /// Unique identifier for this binding.
    id: BindingId,
    /// The source property ID.
    source_id: PropertyId,
    /// The target property ID.
    target_id: PropertyId,
    /// The direction of data flow.
    direction: BindingDirection,
    /// The update mode.
    mode: BindingMode,
    /// Optional converter for transforming values.
    converter: Option<BindingConverter>,
    /// Whether this binding is currently active.
    active: bool,
    /// The last source version that was propagated.
    last_source_version: u64,
    /// The last target version that was propagated (for two-way bindings).
    last_target_version: u64,
}

impl fmt::Debug for BindingDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BindingDef")
            .field("id", &self.id)
            .field("source_id", &self.source_id)
            .field("target_id", &self.target_id)
            .field("direction", &self.direction)
            .field("mode", &self.mode)
            .field("active", &self.active)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CollectionChange
// ---------------------------------------------------------------------------

/// Describes a change to an observable collection.
#[derive(Debug, Clone)]
pub enum CollectionChange {
    /// An item was added at the given index.
    ItemAdded {
        /// The index where the item was added.
        index: usize,
    },
    /// An item was removed from the given index.
    ItemRemoved {
        /// The index from which the item was removed.
        index: usize,
    },
    /// An item at the given index was replaced.
    ItemReplaced {
        /// The index of the replaced item.
        index: usize,
    },
    /// An item was moved from one index to another.
    ItemMoved {
        /// The original index.
        from: usize,
        /// The new index.
        to: usize,
    },
    /// The entire collection was replaced or reset.
    Reset,
}

// ---------------------------------------------------------------------------
// ObservableCollection
// ---------------------------------------------------------------------------

/// A collection that notifies observers when its contents change.
///
/// Used for binding list/grid views to dynamic data. When items are added,
/// removed, moved, or replaced, registered callbacks are invoked so the UI
/// can update incrementally rather than rebuilding the entire list.
pub struct ObservableCollection<T> {
    /// The items in the collection.
    items: Vec<T>,
    /// Monotonically increasing version counter.
    version: u64,
    /// Pending change notifications.
    pending_changes: Vec<CollectionChange>,
    /// Maximum number of pending changes before a reset is issued instead.
    max_pending_changes: usize,
}

impl<T: fmt::Debug> fmt::Debug for ObservableCollection<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObservableCollection")
            .field("len", &self.items.len())
            .field("version", &self.version)
            .field("pending_changes", &self.pending_changes.len())
            .finish()
    }
}

impl<T: Clone> ObservableCollection<T> {
    /// Creates a new empty observable collection.
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            version: 0,
            pending_changes: Vec::new(),
            max_pending_changes: 64,
        }
    }

    /// Creates an observable collection from an existing vector.
    pub fn from_vec(items: Vec<T>) -> Self {
        Self {
            items,
            version: 0,
            pending_changes: Vec::new(),
            max_pending_changes: 64,
        }
    }

    /// Sets the maximum number of pending changes before a reset notification
    /// is issued instead of individual change notifications.
    pub fn set_max_pending_changes(&mut self, max: usize) {
        self.max_pending_changes = max;
    }

    /// Returns the number of items in the collection.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns a reference to an item at the given index.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    /// Returns a mutable reference to an item at the given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.items.get_mut(index)
    }

    /// Returns a slice of all items.
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }

    /// Returns the current version number.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Appends an item to the end of the collection.
    pub fn push(&mut self, item: T) {
        let index = self.items.len();
        self.items.push(item);
        self.version += 1;
        self.record_change(CollectionChange::ItemAdded { index });
    }

    /// Inserts an item at the given index, shifting subsequent items.
    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item);
        self.version += 1;
        self.record_change(CollectionChange::ItemAdded { index });
    }

    /// Removes and returns the item at the given index.
    pub fn remove(&mut self, index: usize) -> T {
        let item = self.items.remove(index);
        self.version += 1;
        self.record_change(CollectionChange::ItemRemoved { index });
        item
    }

    /// Replaces the item at the given index, returning the old item.
    pub fn replace(&mut self, index: usize, new_item: T) -> T {
        let old = std::mem::replace(&mut self.items[index], new_item);
        self.version += 1;
        self.record_change(CollectionChange::ItemReplaced { index });
        old
    }

    /// Moves an item from one index to another.
    pub fn move_item(&mut self, from: usize, to: usize) {
        if from == to || from >= self.items.len() || to >= self.items.len() {
            return;
        }
        let item = self.items.remove(from);
        self.items.insert(to, item);
        self.version += 1;
        self.record_change(CollectionChange::ItemMoved { from, to });
    }

    /// Clears the collection, removing all items.
    pub fn clear(&mut self) {
        self.items.clear();
        self.version += 1;
        self.record_change(CollectionChange::Reset);
    }

    /// Replaces the entire collection with a new set of items.
    pub fn reset(&mut self, new_items: Vec<T>) {
        self.items = new_items;
        self.version += 1;
        self.record_change(CollectionChange::Reset);
    }

    /// Sorts the collection using the provided comparison function.
    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        self.items.sort_by(compare);
        self.version += 1;
        self.record_change(CollectionChange::Reset);
    }

    /// Retains only elements for which the predicate returns `true`.
    pub fn retain<F>(&mut self, predicate: F)
    where
        F: FnMut(&T) -> bool,
    {
        let old_len = self.items.len();
        self.items.retain(predicate);
        if self.items.len() != old_len {
            self.version += 1;
            self.record_change(CollectionChange::Reset);
        }
    }

    /// Returns an iterator over the items.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }

    /// Takes and returns all pending change notifications.
    pub fn drain_changes(&mut self) -> Vec<CollectionChange> {
        std::mem::take(&mut self.pending_changes)
    }

    /// Returns `true` if there are pending change notifications.
    pub fn has_pending_changes(&self) -> bool {
        !self.pending_changes.is_empty()
    }

    /// Records a change, collapsing to a reset if too many changes accumulate.
    fn record_change(&mut self, change: CollectionChange) {
        if self.pending_changes.len() >= self.max_pending_changes {
            self.pending_changes.clear();
            self.pending_changes.push(CollectionChange::Reset);
        } else {
            self.pending_changes.push(change);
        }
    }
}

impl<T: Clone> Default for ObservableCollection<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CollectionBinding
// ---------------------------------------------------------------------------

/// Binds an [`ObservableCollection`] to a UI list view or similar repeating
/// container widget.
///
/// The collection binding monitors changes to the collection and translates
/// them into UI update commands (add row, remove row, update row, etc.).
pub struct CollectionBinding {
    /// Unique identifier for this binding.
    pub id: BindingId,
    /// Human-readable name.
    pub name: String,
    /// The property ID of the collection (stored as a special property type).
    pub collection_property_id: PropertyId,
    /// The last collection version that was synced.
    pub last_synced_version: u64,
    /// Whether this binding is currently active.
    pub active: bool,
    /// Template property IDs for each column/field of the item.
    pub item_template_properties: Vec<String>,
    /// Pending UI update commands generated from collection changes.
    pub pending_updates: Vec<CollectionUpdateCommand>,
}

/// Commands generated by collection binding to update the UI.
#[derive(Debug, Clone)]
pub enum CollectionUpdateCommand {
    /// Insert a new row at the given index.
    InsertRow { index: usize },
    /// Remove the row at the given index.
    RemoveRow { index: usize },
    /// Update the row at the given index with new data.
    UpdateRow { index: usize },
    /// Move a row from one index to another.
    MoveRow { from: usize, to: usize },
    /// Rebuild the entire list from scratch.
    RebuildAll,
}

impl CollectionBinding {
    /// Creates a new collection binding.
    pub fn new(
        name: impl Into<String>,
        collection_property_id: PropertyId,
        item_template_properties: Vec<String>,
    ) -> Self {
        Self {
            id: next_binding_id(),
            name: name.into(),
            collection_property_id,
            last_synced_version: 0,
            active: true,
            item_template_properties,
            pending_updates: Vec::new(),
        }
    }

    /// Processes collection changes and generates UI update commands.
    pub fn process_changes(&mut self, changes: &[CollectionChange]) {
        for change in changes {
            match change {
                CollectionChange::ItemAdded { index } => {
                    self.pending_updates
                        .push(CollectionUpdateCommand::InsertRow { index: *index });
                }
                CollectionChange::ItemRemoved { index } => {
                    self.pending_updates
                        .push(CollectionUpdateCommand::RemoveRow { index: *index });
                }
                CollectionChange::ItemReplaced { index } => {
                    self.pending_updates
                        .push(CollectionUpdateCommand::UpdateRow { index: *index });
                }
                CollectionChange::ItemMoved { from, to } => {
                    self.pending_updates
                        .push(CollectionUpdateCommand::MoveRow {
                            from: *from,
                            to: *to,
                        });
                }
                CollectionChange::Reset => {
                    self.pending_updates.clear();
                    self.pending_updates
                        .push(CollectionUpdateCommand::RebuildAll);
                }
            }
        }
    }

    /// Takes and returns all pending UI update commands.
    pub fn drain_updates(&mut self) -> Vec<CollectionUpdateCommand> {
        std::mem::take(&mut self.pending_updates)
    }

    /// Returns `true` if there are pending UI update commands.
    pub fn has_pending_updates(&self) -> bool {
        !self.pending_updates.is_empty()
    }
}

impl fmt::Debug for CollectionBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CollectionBinding")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("active", &self.active)
            .field("pending_updates", &self.pending_updates.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// BindingError
// ---------------------------------------------------------------------------

/// Errors from binding operations.
#[derive(Debug, Clone)]
pub enum BindingError {
    /// The specified property was not found.
    PropertyNotFound(PropertyId),
    /// The specified binding was not found.
    BindingNotFound(BindingId),
    /// A type mismatch occurred during value transfer.
    TypeMismatch {
        /// Expected type name.
        expected: String,
        /// Actual type name.
        actual: String,
    },
    /// Validation failed when setting a property value.
    ValidationFailed(Vec<ValidationResult>),
    /// A converter failed to produce a value.
    ConversionFailed {
        /// Name of the converter that failed.
        converter_name: String,
    },
    /// A circular dependency was detected in computed properties.
    CircularDependency(Vec<PropertyId>),
    /// The property name is already in use.
    DuplicateName(String),
}

impl fmt::Display for BindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PropertyNotFound(id) => write!(f, "Property not found: {}", id),
            Self::BindingNotFound(id) => write!(f, "Binding not found: {}", id),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "Type mismatch: expected {}, got {}", expected, actual)
            }
            Self::ValidationFailed(results) => {
                write!(f, "Validation failed: ")?;
                for (i, r) in results.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    if let Some(msg) = r.message() {
                        write!(f, "{}", msg)?;
                    }
                }
                Ok(())
            }
            Self::ConversionFailed { converter_name } => {
                write!(f, "Conversion failed in '{}'", converter_name)
            }
            Self::CircularDependency(ids) => {
                write!(f, "Circular dependency detected: {:?}", ids)
            }
            Self::DuplicateName(name) => {
                write!(f, "Property name already in use: '{}'", name)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BindingContext
// ---------------------------------------------------------------------------

/// Central coordinator for all data bindings.
///
/// The binding context owns all observable properties and manages bindings
/// between them. It handles change propagation, computed property evaluation,
/// and collection binding updates.
///
/// # Lifecycle
///
/// 1. Create properties with [`create_property`](Self::create_property).
/// 2. Set up bindings with [`bind`](Self::bind) or [`bind_two_way`](Self::bind_two_way).
/// 3. Create computed properties with [`create_computed`](Self::create_computed).
/// 4. Call [`tick`](Self::tick) each frame to process deferred updates.
pub struct BindingContext {
    /// All registered properties.
    properties: HashMap<PropertyId, PropertyStorage>,
    /// Name-to-ID lookup.
    name_to_id: HashMap<String, PropertyId>,
    /// All registered bindings.
    bindings: Vec<BindingDef>,
    /// Computed property definitions.
    computed_properties: Vec<ComputedPropertyDef>,
    /// Collection bindings.
    collection_bindings: Vec<CollectionBinding>,
    /// Pending change notifications.
    pending_notifications: Vec<ChangeNotification>,
    /// Global change counter for ordering change notifications.
    change_counter: u64,
    /// Whether we are currently inside a batch update (suppressing propagation).
    batch_depth: u32,
    /// Properties that were changed during a batch update.
    batch_dirty: Vec<PropertyId>,
}

impl BindingContext {
    /// Creates a new, empty binding context.
    pub fn new() -> Self {
        Self {
            properties: HashMap::new(),
            name_to_id: HashMap::new(),
            bindings: Vec::new(),
            computed_properties: Vec::new(),
            collection_bindings: Vec::new(),
            pending_notifications: Vec::new(),
            change_counter: 0,
            batch_depth: 0,
            batch_dirty: Vec::new(),
        }
    }

    /// Creates a new observable property with an initial value.
    ///
    /// Returns the property ID that can be used to get/set the value and
    /// create bindings.
    pub fn create_property<T: Any + Send + Sync + Clone + 'static>(
        &mut self,
        name: impl Into<String>,
        initial_value: T,
    ) -> PropertyId {
        let name_str = name.into();
        let id = next_property_id();
        let storage = PropertyStorage::new(name_str.clone(), initial_value);
        self.properties.insert(id, storage);
        self.name_to_id.insert(name_str, id);
        id
    }

    /// Looks up a property ID by name.
    pub fn find_property(&self, name: &str) -> Option<PropertyId> {
        self.name_to_id.get(name).copied()
    }

    /// Returns the name of a property, if it exists.
    pub fn property_name(&self, id: &PropertyId) -> Option<&str> {
        self.properties.get(id).map(|s| s.name.as_str())
    }

    /// Gets the current value of a property.
    pub fn get<T: Any + 'static>(&self, id: &PropertyId) -> Option<&T> {
        self.properties.get(id).and_then(|s| s.get::<T>())
    }

    /// Gets the current version of a property.
    pub fn property_version(&self, id: &PropertyId) -> Option<u64> {
        self.properties.get(id).map(|s| s.version)
    }

    /// Sets the value of a property, triggering change propagation.
    ///
    /// Returns the validation results and whether the value was changed.
    pub fn set<T: Any + Send + Sync + Clone + PartialEq + 'static>(
        &mut self,
        id: &PropertyId,
        value: T,
    ) -> Result<bool, BindingError> {
        let storage = self
            .properties
            .get_mut(id)
            .ok_or(BindingError::PropertyNotFound(*id))?;

        let (results, changed) = storage.set(value);

        // Check for validation failures.
        let has_invalid = results.iter().any(|r| r.is_invalid());
        if has_invalid {
            return Err(BindingError::ValidationFailed(results));
        }

        if changed {
            self.change_counter += 1;
            let notification = ChangeNotification {
                property_id: *id,
                property_name: storage.name.clone(),
                change_id: self.change_counter,
            };

            if self.batch_depth > 0 {
                self.batch_dirty.push(*id);
            } else {
                self.pending_notifications.push(notification);
                self.propagate_bindings_from(*id);
                self.mark_computed_dirty(*id);
            }
        }

        Ok(changed)
    }

    /// Adds a validator to a property.
    pub fn add_validator(
        &mut self,
        id: &PropertyId,
        validator: BindingValidator,
    ) -> Result<(), BindingError> {
        let storage = self
            .properties
            .get_mut(id)
            .ok_or(BindingError::PropertyNotFound(*id))?;
        storage.validators.push(validator);
        Ok(())
    }

    /// Creates a one-way binding from source to target.
    pub fn bind(
        &mut self,
        source_id: PropertyId,
        target_id: PropertyId,
        converter: Option<BindingConverter>,
    ) -> Result<BindingId, BindingError> {
        self.bind_with_options(
            source_id,
            target_id,
            BindingDirection::OneWay,
            BindingMode::Immediate,
            converter,
        )
    }

    /// Creates a two-way binding between source and target.
    pub fn bind_two_way(
        &mut self,
        source_id: PropertyId,
        target_id: PropertyId,
        converter: Option<BindingConverter>,
    ) -> Result<BindingId, BindingError> {
        self.bind_with_options(
            source_id,
            target_id,
            BindingDirection::TwoWay,
            BindingMode::Immediate,
            converter,
        )
    }

    /// Creates a binding with full control over direction and mode.
    pub fn bind_with_options(
        &mut self,
        source_id: PropertyId,
        target_id: PropertyId,
        direction: BindingDirection,
        mode: BindingMode,
        converter: Option<BindingConverter>,
    ) -> Result<BindingId, BindingError> {
        // Verify both properties exist.
        if !self.properties.contains_key(&source_id) {
            return Err(BindingError::PropertyNotFound(source_id));
        }
        if !self.properties.contains_key(&target_id) {
            return Err(BindingError::PropertyNotFound(target_id));
        }

        let id = next_binding_id();
        let source_version = self.properties[&source_id].version;
        let target_version = self.properties[&target_id].version;

        let binding = BindingDef {
            id,
            source_id,
            target_id,
            direction,
            mode,
            converter,
            active: true,
            last_source_version: source_version,
            last_target_version: target_version,
        };

        self.bindings.push(binding);

        // For OneTime bindings, propagate immediately and then deactivate.
        if direction == BindingDirection::OneTime {
            self.propagate_single_binding(self.bindings.len() - 1, true);
            if let Some(b) = self.bindings.last_mut() {
                b.active = false;
            }
        }

        Ok(id)
    }

    /// Removes a binding by its ID.
    pub fn unbind(&mut self, binding_id: BindingId) -> Result<(), BindingError> {
        let index = self
            .bindings
            .iter()
            .position(|b| b.id == binding_id)
            .ok_or(BindingError::BindingNotFound(binding_id))?;
        self.bindings.remove(index);
        Ok(())
    }

    /// Activates or deactivates a binding.
    pub fn set_binding_active(
        &mut self,
        binding_id: BindingId,
        active: bool,
    ) -> Result<(), BindingError> {
        let binding = self
            .bindings
            .iter_mut()
            .find(|b| b.id == binding_id)
            .ok_or(BindingError::BindingNotFound(binding_id))?;
        binding.active = active;
        Ok(())
    }

    /// Creates a computed property that derives its value from other properties.
    ///
    /// The computation function is called with a [`PropertyAccessor`] that
    /// provides read-only access to other property values. The result is stored
    /// in the target property.
    pub fn create_computed<F>(
        &mut self,
        name: impl Into<String>,
        dependencies: Vec<PropertyId>,
        compute_fn: F,
    ) -> PropertyId
    where
        F: Fn(&PropertyAccessor) -> Box<dyn Any + Send + Sync> + Send + Sync + 'static,
    {
        let name_str = name.into();

        // Create a placeholder property for the computed value.
        let target_id = next_property_id();
        let accessor = PropertyAccessor {
            properties: &self.properties,
        };
        let initial_value = compute_fn(&accessor);
        let mut storage = PropertyStorage {
            type_id: (*initial_value).type_id(),
            value: initial_value,
            name: name_str.clone(),
            validators: Vec::new(),
            version: 0,
            notifications_suppressed: false,
        };
        storage.version = 0;
        self.properties.insert(target_id, storage);
        self.name_to_id.insert(name_str, target_id);

        let dep_versions = dependencies
            .iter()
            .map(|id| self.properties.get(id).map_or(0, |s| s.version))
            .collect();

        let computed = ComputedPropertyDef {
            target_id,
            dependencies,
            compute_fn: Box::new(compute_fn),
            dirty: false,
            dependency_versions: dep_versions,
        };

        self.computed_properties.push(computed);
        target_id
    }

    /// Begins a batch update. Change notifications and binding propagation
    /// are deferred until `end_batch` is called.
    pub fn begin_batch(&mut self) {
        self.batch_depth += 1;
    }

    /// Ends a batch update and processes all deferred changes.
    pub fn end_batch(&mut self) {
        if self.batch_depth == 0 {
            return;
        }
        self.batch_depth -= 1;
        if self.batch_depth == 0 {
            let dirty = std::mem::take(&mut self.batch_dirty);
            for property_id in dirty {
                if let Some(storage) = self.properties.get(&property_id) {
                    self.change_counter += 1;
                    let notification = ChangeNotification {
                        property_id,
                        property_name: storage.name.clone(),
                        change_id: self.change_counter,
                    };
                    self.pending_notifications.push(notification);
                }
                self.propagate_bindings_from(property_id);
                self.mark_computed_dirty(property_id);
            }
        }
    }

    /// Processes pending deferred bindings and recomputes dirty computed
    /// properties. Should be called once per frame.
    pub fn tick(&mut self) {
        // Process deferred bindings.
        self.process_deferred_bindings();

        // Recompute dirty computed properties.
        self.recompute_dirty_properties();

        // Clear pending notifications (consumers should have read them).
        self.pending_notifications.clear();
    }

    /// Takes and returns all pending change notifications.
    pub fn drain_notifications(&mut self) -> Vec<ChangeNotification> {
        std::mem::take(&mut self.pending_notifications)
    }

    /// Returns the number of registered properties.
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Returns the number of active bindings.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Returns the number of computed properties.
    pub fn computed_property_count(&self) -> usize {
        self.computed_properties.len()
    }

    /// Returns the number of collection bindings.
    pub fn collection_binding_count(&self) -> usize {
        self.collection_bindings.len()
    }

    /// Adds a collection binding.
    pub fn add_collection_binding(&mut self, binding: CollectionBinding) {
        self.collection_bindings.push(binding);
    }

    /// Removes a collection binding by its ID.
    pub fn remove_collection_binding(&mut self, id: BindingId) -> bool {
        if let Some(index) = self.collection_bindings.iter().position(|b| b.id == id) {
            self.collection_bindings.remove(index);
            true
        } else {
            false
        }
    }

    /// Returns a read-only accessor for evaluating property values.
    pub fn accessor(&self) -> PropertyAccessor<'_> {
        PropertyAccessor {
            properties: &self.properties,
        }
    }

    /// Returns a summary of all registered properties and their current state.
    pub fn debug_dump(&self) -> Vec<PropertyDebugInfo> {
        self.properties
            .iter()
            .map(|(id, storage)| PropertyDebugInfo {
                id: *id,
                name: storage.name.clone(),
                version: storage.version,
                validator_count: storage.validators.len(),
                is_computed: self
                    .computed_properties
                    .iter()
                    .any(|c| c.target_id == *id),
            })
            .collect()
    }

    /// Returns a list of all bindings and their current state.
    pub fn debug_bindings(&self) -> Vec<BindingDebugInfo> {
        self.bindings
            .iter()
            .map(|b| BindingDebugInfo {
                id: b.id,
                source_id: b.source_id,
                target_id: b.target_id,
                direction: b.direction,
                mode: b.mode,
                active: b.active,
                has_converter: b.converter.is_some(),
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Propagates all bindings that have the given source property.
    fn propagate_bindings_from(&mut self, source_id: PropertyId) {
        let binding_count = self.bindings.len();
        for i in 0..binding_count {
            if !self.bindings[i].active {
                continue;
            }
            if self.bindings[i].mode == BindingMode::Explicit {
                continue;
            }
            if self.bindings[i].source_id == source_id {
                self.propagate_single_binding(i, true);
            } else if self.bindings[i].direction == BindingDirection::TwoWay
                && self.bindings[i].target_id == source_id
            {
                self.propagate_single_binding(i, false);
            }
        }
    }

    /// Propagates a single binding in the specified direction.
    fn propagate_single_binding(&mut self, index: usize, forward: bool) {
        let binding = &self.bindings[index];
        let (from_id, to_id) = if forward {
            (binding.source_id, binding.target_id)
        } else {
            (binding.target_id, binding.source_id)
        };

        // Read the source value.
        let source_value_ptr = if let Some(storage) = self.properties.get(&from_id) {
            &*storage.value as *const dyn Any
        } else {
            return;
        };

        // Apply converter if present.
        let binding = &self.bindings[index];
        if let Some(ref converter) = binding.converter {
            // SAFETY: We are not mutating the source storage during this read.
            let source_ref = unsafe { &*source_value_ptr };
            let converted = if forward {
                converter.convert(source_ref)
            } else {
                converter.convert_back(source_ref)
            };

            if let Some(new_value) = converted {
                if let Some(target_storage) = self.properties.get_mut(&to_id) {
                    target_storage.set_any(new_value);
                }
            }
        } else {
            // No converter: direct type-erased copy is not possible without
            // knowing the concrete type at compile time. We skip propagation
            // for bindings without converters when the types differ. For
            // same-type bindings, we allow the source to overwrite the target
            // using a raw copy of the boxed value.
            //
            // In a real engine this would use a type-erased clone trait.
            // For now, we note that converter-less bindings between same types
            // require the value to implement Clone, which we cannot enforce at
            // this level of abstraction. The binding is recorded but no value
            // is transferred until a converter is attached.
        }

        // Update version tracking.
        let binding = &mut self.bindings[index];
        if forward {
            if let Some(storage) = self.properties.get(&from_id) {
                binding.last_source_version = storage.version;
            }
        } else {
            if let Some(storage) = self.properties.get(&to_id) {
                binding.last_target_version = storage.version;
            }
        }
    }

    /// Marks computed properties as dirty if they depend on the changed property.
    fn mark_computed_dirty(&mut self, changed_id: PropertyId) {
        for computed in &mut self.computed_properties {
            if computed.dependencies.contains(&changed_id) {
                computed.dirty = true;
            }
        }
    }

    /// Recomputes all dirty computed properties.
    fn recompute_dirty_properties(&mut self) {
        let computed_count = self.computed_properties.len();
        for i in 0..computed_count {
            if !self.computed_properties[i].dirty {
                // Also check if any dependency versions have changed.
                let mut any_changed = false;
                for (dep_idx, dep_id) in
                    self.computed_properties[i].dependencies.iter().enumerate()
                {
                    if let Some(storage) = self.properties.get(dep_id) {
                        if storage.version
                            != self.computed_properties[i].dependency_versions[dep_idx]
                        {
                            any_changed = true;
                            break;
                        }
                    }
                }
                if !any_changed {
                    continue;
                }
            }

            let accessor = PropertyAccessor {
                properties: &self.properties,
            };
            let new_value = (self.computed_properties[i].compute_fn)(&accessor);
            let target_id = self.computed_properties[i].target_id;

            if let Some(storage) = self.properties.get_mut(&target_id) {
                storage.set_any(new_value);
            }

            // Update dependency versions.
            let deps: Vec<_> = self.computed_properties[i].dependencies.clone();
            for (dep_idx, dep_id) in deps.iter().enumerate() {
                if let Some(storage) = self.properties.get(dep_id) {
                    self.computed_properties[i].dependency_versions[dep_idx] = storage.version;
                }
            }
            self.computed_properties[i].dirty = false;

            // Propagate computed property changes.
            self.propagate_bindings_from(target_id);
        }
    }

    /// Processes deferred-mode bindings.
    fn process_deferred_bindings(&mut self) {
        let binding_count = self.bindings.len();
        for i in 0..binding_count {
            if !self.bindings[i].active || self.bindings[i].mode != BindingMode::Deferred {
                continue;
            }

            let source_id = self.bindings[i].source_id;
            let source_version = self
                .properties
                .get(&source_id)
                .map_or(0, |s| s.version);

            if source_version != self.bindings[i].last_source_version {
                self.propagate_single_binding(i, true);
            }

            if self.bindings[i].direction == BindingDirection::TwoWay {
                let target_id = self.bindings[i].target_id;
                let target_version = self
                    .properties
                    .get(&target_id)
                    .map_or(0, |s| s.version);
                if target_version != self.bindings[i].last_target_version {
                    self.propagate_single_binding(i, false);
                }
            }
        }
    }
}

impl Default for BindingContext {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BindingContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BindingContext")
            .field("property_count", &self.properties.len())
            .field("binding_count", &self.bindings.len())
            .field("computed_count", &self.computed_properties.len())
            .field("collection_binding_count", &self.collection_bindings.len())
            .field("pending_notifications", &self.pending_notifications.len())
            .field("batch_depth", &self.batch_depth)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Debug info types
// ---------------------------------------------------------------------------

/// Debug information about a property.
#[derive(Debug, Clone)]
pub struct PropertyDebugInfo {
    /// Property identifier.
    pub id: PropertyId,
    /// Property name.
    pub name: String,
    /// Current version.
    pub version: u64,
    /// Number of validators.
    pub validator_count: usize,
    /// Whether this is a computed property.
    pub is_computed: bool,
}

/// Debug information about a binding.
#[derive(Debug, Clone)]
pub struct BindingDebugInfo {
    /// Binding identifier.
    pub id: BindingId,
    /// Source property.
    pub source_id: PropertyId,
    /// Target property.
    pub target_id: PropertyId,
    /// Binding direction.
    pub direction: BindingDirection,
    /// Binding mode.
    pub mode: BindingMode,
    /// Whether the binding is active.
    pub active: bool,
    /// Whether the binding has a converter.
    pub has_converter: bool,
}

// ---------------------------------------------------------------------------
// BindingPath
// ---------------------------------------------------------------------------

/// A dot-separated path for navigating nested property hierarchies.
///
/// For example, `"player.stats.health"` would navigate through nested
/// objects to find the health property.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindingPath {
    /// The segments of the path.
    pub segments: Vec<String>,
}

impl BindingPath {
    /// Parses a dot-separated path string.
    pub fn parse(path: &str) -> Self {
        Self {
            segments: path.split('.').map(|s| s.to_string()).collect(),
        }
    }

    /// Returns the number of segments in the path.
    pub fn depth(&self) -> usize {
        self.segments.len()
    }

    /// Returns `true` if this path is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Returns the first segment of the path, if any.
    pub fn root(&self) -> Option<&str> {
        self.segments.first().map(|s| s.as_str())
    }

    /// Returns the last segment of the path, if any.
    pub fn leaf(&self) -> Option<&str> {
        self.segments.last().map(|s| s.as_str())
    }

    /// Returns a new path with the first segment removed.
    pub fn tail(&self) -> Self {
        Self {
            segments: self.segments.iter().skip(1).cloned().collect(),
        }
    }

    /// Returns a new path with an additional segment appended.
    pub fn append(&self, segment: impl Into<String>) -> Self {
        let mut segments = self.segments.clone();
        segments.push(segment.into());
        Self { segments }
    }

    /// Returns the path as a dot-separated string.
    pub fn to_string(&self) -> String {
        self.segments.join(".")
    }

    /// Returns `true` if this path starts with the given prefix path.
    pub fn starts_with(&self, prefix: &BindingPath) -> bool {
        if prefix.segments.len() > self.segments.len() {
            return false;
        }
        self.segments
            .iter()
            .zip(prefix.segments.iter())
            .all(|(a, b)| a == b)
    }
}

impl fmt::Display for BindingPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.segments.join("."))
    }
}

// ---------------------------------------------------------------------------
// BindingExpression
// ---------------------------------------------------------------------------

/// A binding expression that can reference properties using path syntax.
///
/// Expressions support simple property references and basic operations
/// like string formatting with embedded property references.
#[derive(Debug, Clone)]
pub enum BindingExpression {
    /// A simple property reference by path.
    PropertyRef(BindingPath),
    /// A literal string value.
    Literal(String),
    /// A format string with embedded property references.
    /// Placeholders use `{path}` syntax.
    Format {
        /// The format template with `{path}` placeholders.
        template: String,
        /// The paths referenced in the template.
        referenced_paths: Vec<BindingPath>,
    },
    /// A conditional expression.
    Conditional {
        /// The condition property path (must resolve to bool).
        condition: BindingPath,
        /// Value when true.
        true_value: Box<BindingExpression>,
        /// Value when false.
        false_value: Box<BindingExpression>,
    },
}

impl BindingExpression {
    /// Creates a simple property reference expression.
    pub fn property(path: &str) -> Self {
        Self::PropertyRef(BindingPath::parse(path))
    }

    /// Creates a literal string expression.
    pub fn literal(value: impl Into<String>) -> Self {
        Self::Literal(value.into())
    }

    /// Creates a format expression from a template string.
    ///
    /// Scans the template for `{path}` placeholders and extracts the
    /// referenced property paths.
    pub fn format(template: &str) -> Self {
        let mut referenced_paths = Vec::new();
        let mut in_brace = false;
        let mut current_path = String::new();

        for ch in template.chars() {
            match ch {
                '{' if !in_brace => {
                    in_brace = true;
                    current_path.clear();
                }
                '}' if in_brace => {
                    in_brace = false;
                    if !current_path.is_empty() {
                        referenced_paths.push(BindingPath::parse(&current_path));
                    }
                }
                _ if in_brace => {
                    current_path.push(ch);
                }
                _ => {}
            }
        }

        Self::Format {
            template: template.to_string(),
            referenced_paths,
        }
    }

    /// Creates a conditional expression.
    pub fn conditional(
        condition_path: &str,
        true_value: BindingExpression,
        false_value: BindingExpression,
    ) -> Self {
        Self::Conditional {
            condition: BindingPath::parse(condition_path),
            true_value: Box::new(true_value),
            false_value: Box::new(false_value),
        }
    }

    /// Returns all property paths referenced by this expression.
    pub fn referenced_paths(&self) -> Vec<&BindingPath> {
        match self {
            Self::PropertyRef(path) => vec![path],
            Self::Literal(_) => Vec::new(),
            Self::Format {
                referenced_paths, ..
            } => referenced_paths.iter().collect(),
            Self::Conditional {
                condition,
                true_value,
                false_value,
            } => {
                let mut paths = vec![condition];
                paths.extend(true_value.referenced_paths());
                paths.extend(false_value.referenced_paths());
                paths
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BindingScope
// ---------------------------------------------------------------------------

/// A scoped subset of properties that can be bound to a UI subtree.
///
/// Scopes provide a way to create isolated binding namespaces, useful for
/// list item templates where each item needs its own set of property bindings.
#[derive(Debug)]
pub struct BindingScope {
    /// Unique identifier for this scope.
    pub id: u64,
    /// Human-readable name for debugging.
    pub name: String,
    /// The parent scope, if any.
    pub parent_scope_id: Option<u64>,
    /// Properties defined in this scope.
    pub local_properties: HashMap<String, PropertyId>,
    /// Active bindings within this scope.
    pub binding_ids: Vec<BindingId>,
}

/// Global scope counter.
static NEXT_SCOPE_ID: AtomicU64 = AtomicU64::new(1);

impl BindingScope {
    /// Creates a new root scope.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: NEXT_SCOPE_ID.fetch_add(1, Ordering::Relaxed),
            name: name.into(),
            parent_scope_id: None,
            local_properties: HashMap::new(),
            binding_ids: Vec::new(),
        }
    }

    /// Creates a child scope with the given parent.
    pub fn child(name: impl Into<String>, parent_id: u64) -> Self {
        Self {
            id: NEXT_SCOPE_ID.fetch_add(1, Ordering::Relaxed),
            name: name.into(),
            parent_scope_id: Some(parent_id),
            local_properties: HashMap::new(),
            binding_ids: Vec::new(),
        }
    }

    /// Registers a property in this scope.
    pub fn register_property(&mut self, name: impl Into<String>, id: PropertyId) {
        self.local_properties.insert(name.into(), id);
    }

    /// Looks up a property by name in this scope.
    pub fn find_property(&self, name: &str) -> Option<PropertyId> {
        self.local_properties.get(name).copied()
    }

    /// Adds a binding ID to this scope's tracking list.
    pub fn track_binding(&mut self, id: BindingId) {
        self.binding_ids.push(id);
    }

    /// Returns the number of local properties.
    pub fn property_count(&self) -> usize {
        self.local_properties.len()
    }

    /// Returns the number of tracked bindings.
    pub fn binding_count(&self) -> usize {
        self.binding_ids.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_get_property() {
        let mut ctx = BindingContext::new();
        let id = ctx.create_property("health", 100.0f64);
        assert_eq!(ctx.get::<f64>(&id), Some(&100.0));
    }

    #[test]
    fn test_set_property() {
        let mut ctx = BindingContext::new();
        let id = ctx.create_property("health", 100.0f64);
        let changed = ctx.set(&id, 75.0f64).unwrap();
        assert!(changed);
        assert_eq!(ctx.get::<f64>(&id), Some(&75.0));
    }

    #[test]
    fn test_set_same_value_no_change() {
        let mut ctx = BindingContext::new();
        let id = ctx.create_property("health", 100.0f64);
        let changed = ctx.set(&id, 100.0f64).unwrap();
        assert!(!changed);
    }

    #[test]
    fn test_find_property_by_name() {
        let mut ctx = BindingContext::new();
        let id = ctx.create_property("player_name", "Hero".to_string());
        assert_eq!(ctx.find_property("player_name"), Some(id));
        assert_eq!(ctx.find_property("nonexistent"), None);
    }

    #[test]
    fn test_validator_range() {
        let mut ctx = BindingContext::new();
        let id = ctx.create_property("health", 100.0f64);
        ctx.add_validator(&id, BindingValidator::range_f64("health_range", 0.0, 100.0))
            .unwrap();
        assert!(ctx.set(&id, 50.0f64).is_ok());
        assert!(ctx.set(&id, 150.0f64).is_err());
        assert!(ctx.set(&id, -10.0f64).is_err());
    }

    #[test]
    fn test_observable_collection() {
        let mut col: ObservableCollection<String> = ObservableCollection::new();
        col.push("Item 1".to_string());
        col.push("Item 2".to_string());
        assert_eq!(col.len(), 2);

        let changes = col.drain_changes();
        assert_eq!(changes.len(), 2);

        col.remove(0);
        assert_eq!(col.len(), 1);
        assert_eq!(col.get(0), Some(&"Item 2".to_string()));
    }

    #[test]
    fn test_binding_path_parse() {
        let path = BindingPath::parse("player.stats.health");
        assert_eq!(path.depth(), 3);
        assert_eq!(path.root(), Some("player"));
        assert_eq!(path.leaf(), Some("health"));
    }

    #[test]
    fn test_binding_expression_format() {
        let expr = BindingExpression::format("HP: {player.health} / {player.max_health}");
        let paths = expr.referenced_paths();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_currency_converter() {
        let converter = BindingConverter::currency("$", 2);
        let value: f64 = 42.5;
        let result = converter.convert(&value);
        assert!(result.is_some());
    }

    #[test]
    fn test_batch_updates() {
        let mut ctx = BindingContext::new();
        let id1 = ctx.create_property("a", 1.0f64);
        let id2 = ctx.create_property("b", 2.0f64);

        ctx.begin_batch();
        ctx.set(&id1, 10.0f64).unwrap();
        ctx.set(&id2, 20.0f64).unwrap();
        assert!(ctx.pending_notifications.is_empty());
        ctx.end_batch();
        assert_eq!(ctx.pending_notifications.len(), 2);
    }

    #[test]
    fn test_binding_scope() {
        let mut scope = BindingScope::new("root");
        let prop_id = PropertyId::from_raw(42);
        scope.register_property("health", prop_id);
        assert_eq!(scope.find_property("health"), Some(prop_id));
        assert_eq!(scope.find_property("mana"), None);
    }
}
