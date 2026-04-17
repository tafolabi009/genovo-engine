//! Scene templates (prefabs): template definition, instantiation, override
//! tracking, template inheritance, and template variables.
//!
//! Templates (also known as prefabs) are reusable, parameterized scene
//! sub-graphs. They enable a workflow where:
//!
//! - Artists/designers create a template once.
//! - Multiple instances can be placed in scenes.
//! - Per-instance overrides are tracked against the template.
//! - Templates can inherit from other templates (single inheritance).
//! - Templates can expose variables for per-instance customization.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Template ID
// ---------------------------------------------------------------------------

/// Unique identifier for a template definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemplateId(pub u64);

impl TemplateId {
    pub const INVALID: Self = Self(u64::MAX);

    pub fn from_name(name: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut hasher);
        Self(hasher.finish())
    }

    pub fn is_valid(&self) -> bool {
        self.0 != u64::MAX
    }
}

impl fmt::Display for TemplateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Template({})", self.0)
    }
}

/// Unique identifier for a template instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemplateInstanceId(pub u64);

impl TemplateInstanceId {
    pub const INVALID: Self = Self(u64::MAX);
}

// ---------------------------------------------------------------------------
// Template variable
// ---------------------------------------------------------------------------

/// The type of a template variable.
#[derive(Debug, Clone, PartialEq)]
pub enum VariableType {
    Bool,
    Int,
    Float,
    String,
    Vec3,
    Color,
    AssetRef,
}

/// A template variable value.
#[derive(Debug, Clone, PartialEq)]
pub enum VariableValue {
    Bool(bool),
    Int(i32),
    Float(f32),
    String(String),
    Vec3([f32; 3]),
    Color([f32; 4]),
    AssetRef(String),
}

impl VariableValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::String(_) => "string",
            Self::Vec3(_) => "vec3",
            Self::Color(_) => "color",
            Self::AssetRef(_) => "asset_ref",
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i32> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_vec3(&self) -> Option<[f32; 3]> {
        match self {
            Self::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_color(&self) -> Option<[f32; 4]> {
        match self {
            Self::Color(v) => Some(*v),
            _ => None,
        }
    }
}

impl fmt::Display for VariableValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{}", v),
            Self::Int(v) => write!(f, "{}", v),
            Self::Float(v) => write!(f, "{:.4}", v),
            Self::String(v) => write!(f, "\"{}\"", v),
            Self::Vec3(v) => write!(f, "({}, {}, {})", v[0], v[1], v[2]),
            Self::Color(v) => write!(f, "rgba({}, {}, {}, {})", v[0], v[1], v[2], v[3]),
            Self::AssetRef(v) => write!(f, "@{}", v),
        }
    }
}

/// Definition of a template variable.
#[derive(Debug, Clone)]
pub struct TemplateVariable {
    /// Variable name.
    pub name: String,
    /// Display name for the editor.
    pub display_name: String,
    /// Variable type.
    pub var_type: VariableType,
    /// Default value.
    pub default_value: VariableValue,
    /// Optional tooltip.
    pub tooltip: String,
    /// Optional category for grouping in editor.
    pub category: Option<String>,
    /// Optional minimum (for numeric types).
    pub min: Option<f32>,
    /// Optional maximum (for numeric types).
    pub max: Option<f32>,
    /// Component and field path this variable maps to.
    pub binding: Option<VariableBinding>,
}

impl TemplateVariable {
    /// Create a new variable.
    pub fn new(
        name: impl Into<String>,
        var_type: VariableType,
        default: VariableValue,
    ) -> Self {
        let name = name.into();
        Self {
            display_name: name.clone(),
            name,
            var_type,
            default_value: default,
            tooltip: String::new(),
            category: None,
            min: None,
            max: None,
            binding: None,
        }
    }

    /// Set display name.
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = name.into();
        self
    }

    /// Set tooltip.
    pub fn with_tooltip(mut self, tooltip: impl Into<String>) -> Self {
        self.tooltip = tooltip.into();
        self
    }

    /// Set range.
    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    /// Set binding.
    pub fn with_binding(mut self, binding: VariableBinding) -> Self {
        self.binding = Some(binding);
        self
    }
}

/// Describes where a variable value is applied in the template.
#[derive(Debug, Clone)]
pub struct VariableBinding {
    /// Node path within the template (e.g., "Root/Mesh").
    pub node_path: String,
    /// Component type name.
    pub component: String,
    /// Field path within the component.
    pub field: String,
}

impl VariableBinding {
    pub fn new(
        node_path: impl Into<String>,
        component: impl Into<String>,
        field: impl Into<String>,
    ) -> Self {
        Self {
            node_path: node_path.into(),
            component: component.into(),
            field: field.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Template node
// ---------------------------------------------------------------------------

/// A node within a template definition.
#[derive(Debug, Clone)]
pub struct TemplateNode {
    /// Node name.
    pub name: String,
    /// Local transform (position, rotation, scale).
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    /// Components attached to this node (type_name -> serialized data).
    pub components: HashMap<String, HashMap<String, String>>,
    /// Child node indices.
    pub children: Vec<usize>,
    /// Tags.
    pub tags: Vec<String>,
    /// Whether this node is visible by default.
    pub visible: bool,
}

impl TemplateNode {
    /// Create a new template node.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            components: HashMap::new(),
            children: Vec::new(),
            tags: Vec::new(),
            visible: true,
        }
    }

    /// Set position.
    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z];
        self
    }

    /// Add a component.
    pub fn with_component(
        mut self,
        type_name: impl Into<String>,
        data: HashMap<String, String>,
    ) -> Self {
        self.components.insert(type_name.into(), data);
        self
    }
}

// ---------------------------------------------------------------------------
// Template definition
// ---------------------------------------------------------------------------

/// A template (prefab) definition.
#[derive(Debug, Clone)]
pub struct TemplateDefinition {
    /// Unique identifier.
    pub id: TemplateId,
    /// Human-readable name.
    pub name: String,
    /// Optional parent template (for inheritance).
    pub parent: Option<TemplateId>,
    /// Nodes in this template.
    pub nodes: Vec<TemplateNode>,
    /// Exposed variables.
    pub variables: Vec<TemplateVariable>,
    /// Asset path.
    pub asset_path: String,
    /// Description.
    pub description: String,
    /// Category for editor grouping.
    pub category: String,
    /// Thumbnail path.
    pub thumbnail: Option<String>,
    /// Version number.
    pub version: u32,
    /// Tags.
    pub tags: Vec<String>,
}

impl TemplateDefinition {
    /// Create a new template.
    pub fn new(id: TemplateId, name: impl Into<String>) -> Self {
        let mut nodes = Vec::new();
        nodes.push(TemplateNode::new("Root"));

        Self {
            id,
            name: name.into(),
            parent: None,
            nodes,
            variables: Vec::new(),
            asset_path: String::new(),
            description: String::new(),
            category: "General".to_string(),
            thumbnail: None,
            version: 1,
            tags: Vec::new(),
        }
    }

    /// Set parent template for inheritance.
    pub fn with_parent(mut self, parent: TemplateId) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Add a node.
    pub fn add_node(&mut self, parent_index: usize, node: TemplateNode) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        if parent_index < self.nodes.len() {
            self.nodes[parent_index].children.push(index);
        }
        index
    }

    /// Add a variable.
    pub fn add_variable(&mut self, variable: TemplateVariable) {
        self.variables.push(variable);
    }

    /// Get a variable by name.
    pub fn get_variable(&self, name: &str) -> Option<&TemplateVariable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// Get the root node.
    pub fn root_node(&self) -> &TemplateNode {
        &self.nodes[0]
    }

    /// Get a node by index.
    pub fn get_node(&self, index: usize) -> Option<&TemplateNode> {
        self.nodes.get(index)
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// Override tracking
// ---------------------------------------------------------------------------

/// An override applied to a template instance.
#[derive(Debug, Clone)]
pub struct InstanceOverride {
    /// Which node in the template (by path).
    pub node_path: String,
    /// Which component type.
    pub component: String,
    /// Which field.
    pub field: String,
    /// The overridden value (as a string for simplicity).
    pub value: String,
    /// Whether this override is active.
    pub active: bool,
}

impl InstanceOverride {
    pub fn new(
        node_path: impl Into<String>,
        component: impl Into<String>,
        field: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            node_path: node_path.into(),
            component: component.into(),
            field: field.into(),
            value: value.into(),
            active: true,
        }
    }

    /// Create a unique key for this override.
    pub fn key(&self) -> String {
        format!("{}/{}/{}", self.node_path, self.component, self.field)
    }
}

// ---------------------------------------------------------------------------
// Template instance
// ---------------------------------------------------------------------------

/// A placed instance of a template in a scene.
#[derive(Debug, Clone)]
pub struct TemplateInstance {
    /// Instance identifier.
    pub id: TemplateInstanceId,
    /// The template this is an instance of.
    pub template_id: TemplateId,
    /// World position of the instance root.
    pub position: [f32; 3],
    /// World rotation of the instance root.
    pub rotation: [f32; 4],
    /// World scale of the instance root.
    pub scale: [f32; 3],
    /// Per-instance variable values.
    pub variable_values: HashMap<String, VariableValue>,
    /// Per-instance overrides.
    pub overrides: Vec<InstanceOverride>,
    /// Whether this instance is visible.
    pub visible: bool,
    /// Optional instance name.
    pub name: Option<String>,
}

impl TemplateInstance {
    /// Create a new instance of a template.
    pub fn new(
        id: TemplateInstanceId,
        template_id: TemplateId,
    ) -> Self {
        Self {
            id,
            template_id,
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            variable_values: HashMap::new(),
            overrides: Vec::new(),
            visible: true,
            name: None,
        }
    }

    /// Set a variable value.
    pub fn set_variable(
        &mut self,
        name: impl Into<String>,
        value: VariableValue,
    ) {
        self.variable_values.insert(name.into(), value);
    }

    /// Get a variable value.
    pub fn get_variable(&self, name: &str) -> Option<&VariableValue> {
        self.variable_values.get(name)
    }

    /// Add an override.
    pub fn add_override(&mut self, over: InstanceOverride) {
        // Replace existing override for same path.
        let key = over.key();
        if let Some(existing) = self.overrides.iter_mut().find(|o| o.key() == key) {
            *existing = over;
        } else {
            self.overrides.push(over);
        }
    }

    /// Remove an override by key.
    pub fn remove_override(&mut self, key: &str) -> bool {
        let len = self.overrides.len();
        self.overrides.retain(|o| o.key() != key);
        self.overrides.len() < len
    }

    /// Revert all overrides.
    pub fn revert_all(&mut self) {
        self.overrides.clear();
        self.variable_values.clear();
    }

    /// Get the number of overrides.
    pub fn override_count(&self) -> usize {
        self.overrides.iter().filter(|o| o.active).count()
    }

    /// Check if any overrides exist.
    pub fn has_overrides(&self) -> bool {
        self.overrides.iter().any(|o| o.active)
    }

    /// Set position.
    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z];
        self
    }
}

// ---------------------------------------------------------------------------
// Template registry
// ---------------------------------------------------------------------------

/// Central registry for all template definitions and instances.
pub struct TemplateRegistry {
    /// Template definitions.
    templates: HashMap<TemplateId, TemplateDefinition>,
    /// Template instances.
    instances: HashMap<TemplateInstanceId, TemplateInstance>,
    /// Lookup from name to ID.
    name_to_id: HashMap<String, TemplateId>,
    /// Next instance ID.
    next_instance_id: u64,
    /// Registration order.
    registration_order: Vec<TemplateId>,
}

impl TemplateRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            instances: HashMap::new(),
            name_to_id: HashMap::new(),
            next_instance_id: 1,
            registration_order: Vec::new(),
        }
    }

    /// Register a template definition.
    pub fn register(&mut self, template: TemplateDefinition) {
        let id = template.id;
        self.name_to_id
            .insert(template.name.clone(), id);
        self.registration_order.push(id);
        self.templates.insert(id, template);
    }

    /// Get a template by ID.
    pub fn get_template(&self, id: TemplateId) -> Option<&TemplateDefinition> {
        self.templates.get(&id)
    }

    /// Get a template by name.
    pub fn get_template_by_name(&self, name: &str) -> Option<&TemplateDefinition> {
        let id = self.name_to_id.get(name)?;
        self.templates.get(id)
    }

    /// Instantiate a template.
    pub fn instantiate(
        &mut self,
        template_id: TemplateId,
    ) -> Option<TemplateInstanceId> {
        if !self.templates.contains_key(&template_id) {
            return None;
        }

        let instance_id = TemplateInstanceId(self.next_instance_id);
        self.next_instance_id += 1;

        let mut instance = TemplateInstance::new(instance_id, template_id);

        // Set default variable values from template.
        if let Some(template) = self.templates.get(&template_id) {
            for var in &template.variables {
                instance
                    .variable_values
                    .insert(var.name.clone(), var.default_value.clone());
            }
        }

        self.instances.insert(instance_id, instance);
        Some(instance_id)
    }

    /// Get an instance.
    pub fn get_instance(&self, id: TemplateInstanceId) -> Option<&TemplateInstance> {
        self.instances.get(&id)
    }

    /// Get a mutable instance.
    pub fn get_instance_mut(
        &mut self,
        id: TemplateInstanceId,
    ) -> Option<&mut TemplateInstance> {
        self.instances.get_mut(&id)
    }

    /// Remove an instance.
    pub fn remove_instance(&mut self, id: TemplateInstanceId) -> bool {
        self.instances.remove(&id).is_some()
    }

    /// Get all instances of a template.
    pub fn instances_of(&self, template_id: TemplateId) -> Vec<TemplateInstanceId> {
        self.instances
            .iter()
            .filter(|(_, inst)| inst.template_id == template_id)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Resolve template inheritance: get the full node list including
    /// inherited nodes from parent templates.
    pub fn resolve_inheritance(
        &self,
        template_id: TemplateId,
    ) -> Vec<TemplateNode> {
        let mut chain = Vec::new();
        let mut current = Some(template_id);

        while let Some(id) = current {
            if let Some(template) = self.templates.get(&id) {
                chain.push(id);
                current = template.parent;
            } else {
                break;
            }
        }

        // Apply from base to derived.
        chain.reverse();
        let mut result_nodes = Vec::new();

        for id in chain {
            if let Some(template) = self.templates.get(&id) {
                if result_nodes.is_empty() {
                    result_nodes = template.nodes.clone();
                } else {
                    // Merge: add new nodes, override existing by name.
                    for node in &template.nodes {
                        if let Some(existing) = result_nodes
                            .iter_mut()
                            .find(|n| n.name == node.name)
                        {
                            // Override properties.
                            existing.position = node.position;
                            existing.rotation = node.rotation;
                            existing.scale = node.scale;
                            existing.visible = node.visible;
                            for (comp_name, data) in &node.components {
                                existing.components.insert(comp_name.clone(), data.clone());
                            }
                        } else {
                            result_nodes.push(node.clone());
                        }
                    }
                }
            }
        }

        result_nodes
    }

    /// Get all template names.
    pub fn template_names(&self) -> Vec<&str> {
        self.registration_order
            .iter()
            .filter_map(|id| self.templates.get(id))
            .map(|t| t.name.as_str())
            .collect()
    }

    /// Get templates by category.
    pub fn templates_by_category(&self, category: &str) -> Vec<&TemplateDefinition> {
        self.templates
            .values()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Number of registered templates.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// Number of active instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_creation() {
        let id = TemplateId::from_name("Enemy");
        let mut template = TemplateDefinition::new(id, "Enemy");
        template.add_variable(TemplateVariable::new(
            "health",
            VariableType::Float,
            VariableValue::Float(100.0),
        ));

        assert_eq!(template.node_count(), 1);
        assert_eq!(template.variables.len(), 1);
    }

    #[test]
    fn instance_overrides() {
        let mut instance = TemplateInstance::new(
            TemplateInstanceId(1),
            TemplateId(1),
        );
        instance.set_variable("health", VariableValue::Float(50.0));
        instance.add_override(InstanceOverride::new(
            "Root",
            "Health",
            "current",
            "50",
        ));

        assert!(instance.has_overrides());
        assert_eq!(instance.override_count(), 1);
    }

    #[test]
    fn registry_instantiate() {
        let mut registry = TemplateRegistry::new();
        let id = TemplateId::from_name("Tree");
        registry.register(TemplateDefinition::new(id, "Tree"));

        let inst = registry.instantiate(id).unwrap();
        assert!(registry.get_instance(inst).is_some());
        assert_eq!(registry.instances_of(id).len(), 1);
    }

    #[test]
    fn template_inheritance() {
        let mut registry = TemplateRegistry::new();

        let base_id = TemplateId::from_name("Base");
        let mut base = TemplateDefinition::new(base_id, "Base");
        base.add_node(0, TemplateNode::new("Mesh"));
        registry.register(base);

        let derived_id = TemplateId::from_name("Derived");
        let mut derived = TemplateDefinition::new(derived_id, "Derived")
            .with_parent(base_id);
        derived.add_node(0, TemplateNode::new("Collider"));
        registry.register(derived);

        let nodes = registry.resolve_inheritance(derived_id);
        // Should have Root + Mesh (from base) + Collider (from derived).
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn variable_display() {
        let v = VariableValue::Vec3([1.0, 2.0, 3.0]);
        let s = format!("{}", v);
        assert!(s.contains("1"));
    }
}
