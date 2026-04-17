// engine/gameplay/src/physics_interaction.rs
//
// Physics interaction system: pick up objects, throw, push, pull,
// physics handles (spring joint to hand), object inspection mode,
// impact effects, and grab mechanics for first/third person games.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    #[inline] pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    #[inline] pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn length(self) -> f32 { self.length_sq().sqrt() }
    #[inline] pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self * (1.0/l) }
    }
    #[inline] pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Self::new(a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t)
    }
    #[inline] pub fn cross(self, r: Self) -> Self {
        Self::new(self.y*r.z - self.z*r.y, self.z*r.x - self.x*r.z, self.x*r.y - self.y*r.x)
    }
    #[inline] pub fn distance(self, other: Self) -> f32 { (self - other).length() }
}

impl std::ops::Add for Vec3 { type Output=Self; fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z)}}
impl std::ops::Sub for Vec3 { type Output=Self; fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z)}}
impl std::ops::Mul<f32> for Vec3 { type Output=Self; fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s)}}
impl std::ops::Neg for Vec3 { type Output=Self; fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z)}}
impl std::ops::AddAssign for Vec3 { fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;self.z+=r.z;}}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Entity handle placeholder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

/// Ray for picking.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalized() }
    }

    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Result of a physics raycast.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    pub entity: EntityId,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

/// Interaction capability of an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionType {
    /// Can be picked up and carried.
    Grabbable,
    /// Can be pushed/pulled but not fully picked up.
    Pushable,
    /// Can be inspected (rotated/examined).
    Inspectable,
    /// Can be thrown (must be grabbed first).
    Throwable,
    /// A lever or valve that can be turned.
    Turnable,
    /// No physics interaction.
    None,
}

/// Properties of a physics-interactive object.
#[derive(Debug, Clone)]
pub struct InteractableObject {
    pub entity: EntityId,
    pub interaction_type: InteractionType,
    pub mass: f32,
    pub max_grab_distance: f32,
    pub hold_distance: f32,
    pub throw_force: f32,
    pub push_force: f32,
    pub can_break: bool,
    pub break_force: f32,
    pub outline_color: [f32; 4],
    pub interaction_prompt: String,
}

impl InteractableObject {
    pub fn grabbable(entity: EntityId, mass: f32) -> Self {
        Self {
            entity,
            interaction_type: InteractionType::Grabbable,
            mass,
            max_grab_distance: 3.0,
            hold_distance: 1.5,
            throw_force: 10.0,
            push_force: 5.0,
            can_break: false,
            break_force: 100.0,
            outline_color: [1.0, 1.0, 0.0, 1.0],
            interaction_prompt: "Pick up".to_string(),
        }
    }

    pub fn pushable(entity: EntityId, mass: f32) -> Self {
        Self {
            entity,
            interaction_type: InteractionType::Pushable,
            mass,
            max_grab_distance: 2.0,
            hold_distance: 1.0,
            throw_force: 0.0,
            push_force: 8.0,
            can_break: false,
            break_force: 100.0,
            outline_color: [0.0, 1.0, 1.0, 1.0],
            interaction_prompt: "Push".to_string(),
        }
    }

    pub fn inspectable(entity: EntityId) -> Self {
        Self {
            entity,
            interaction_type: InteractionType::Inspectable,
            mass: 1.0,
            max_grab_distance: 2.0,
            hold_distance: 0.8,
            throw_force: 0.0,
            push_force: 0.0,
            can_break: false,
            break_force: 100.0,
            outline_color: [0.0, 1.0, 0.0, 1.0],
            interaction_prompt: "Inspect".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Spring joint (physics handle)
// ---------------------------------------------------------------------------

/// A spring joint connecting a grabbed object to the player's hand.
#[derive(Debug, Clone)]
pub struct SpringJoint {
    /// Target position the spring pulls toward (player hand).
    pub target: Vec3,
    /// Current object position.
    pub object_position: Vec3,
    /// Current object velocity.
    pub object_velocity: Vec3,
    /// Spring stiffness.
    pub stiffness: f32,
    /// Damping coefficient.
    pub damping: f32,
    /// Maximum force the spring can exert.
    pub max_force: f32,
    /// Break distance (if exceeded, drop the object).
    pub break_distance: f32,
    /// The entity being held.
    pub entity: EntityId,
}

impl SpringJoint {
    pub fn new(entity: EntityId, position: Vec3, target: Vec3) -> Self {
        Self {
            target,
            object_position: position,
            object_velocity: Vec3::ZERO,
            stiffness: 50.0,
            damping: 10.0,
            max_force: 200.0,
            break_distance: 5.0,
            entity,
        }
    }

    /// Update the spring joint, computing the force to apply.
    /// Returns `None` if the spring should break.
    pub fn update(&mut self, dt: f32, object_mass: f32) -> Option<Vec3> {
        let displacement = self.target - self.object_position;
        let distance = displacement.length();

        if distance > self.break_distance {
            return None; // Break the joint
        }

        let direction = if distance > 1e-6 {
            displacement * (1.0 / distance)
        } else {
            Vec3::ZERO
        };

        // Spring force: F = -k * x
        let spring_force = direction * (self.stiffness * distance);

        // Damping force: F = -c * v
        let relative_vel = self.object_velocity;
        let damping_force = relative_vel * (-self.damping);

        let mut total_force = spring_force + damping_force;

        // Clamp force magnitude.
        let force_mag = total_force.length();
        if force_mag > self.max_force {
            total_force = total_force * (self.max_force / force_mag);
        }

        // Integrate.
        let acceleration = total_force * (1.0 / object_mass);
        self.object_velocity = self.object_velocity + acceleration * dt;

        // Apply damping to velocity directly.
        self.object_velocity = self.object_velocity * (1.0 - 0.05_f32).max(0.0);

        self.object_position = self.object_position + self.object_velocity * dt;

        Some(total_force)
    }

    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
    }

    pub fn distance_to_target(&self) -> f32 {
        (self.target - self.object_position).length()
    }
}

// ---------------------------------------------------------------------------
// Grab state
// ---------------------------------------------------------------------------

/// State of the grabbed object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrabState {
    /// Not holding anything.
    Idle,
    /// Reaching for an object.
    Reaching,
    /// Holding an object.
    Holding,
    /// Throwing an object.
    Throwing,
    /// Inspecting an object (examining mode).
    Inspecting,
    /// Pushing/pulling an object.
    Pushing,
}

/// The currently held object data.
#[derive(Debug, Clone)]
struct HeldObject {
    entity: EntityId,
    spring: SpringJoint,
    original_position: Vec3,
    grab_offset: Vec3,
    grab_time: f32,
    total_hold_time: f32,
    interaction_type: InteractionType,
}

// ---------------------------------------------------------------------------
// Inspection mode
// ---------------------------------------------------------------------------

/// Configuration for object inspection mode.
#[derive(Debug, Clone)]
pub struct InspectionConfig {
    /// Distance from camera to hold the object.
    pub inspect_distance: f32,
    /// Rotation speed when inspecting.
    pub rotation_speed: f32,
    /// Zoom min/max distance.
    pub zoom_min: f32,
    pub zoom_max: f32,
    /// Whether to pause the game during inspection.
    pub pause_game: bool,
    /// Background dimming alpha.
    pub background_dim: f32,
}

impl Default for InspectionConfig {
    fn default() -> Self {
        Self {
            inspect_distance: 0.6,
            rotation_speed: 2.0,
            zoom_min: 0.3,
            zoom_max: 1.5,
            pause_game: false,
            background_dim: 0.5,
        }
    }
}

/// State during inspection.
#[derive(Debug, Clone)]
pub struct InspectionState {
    pub entity: EntityId,
    pub rotation_x: f32,
    pub rotation_y: f32,
    pub zoom: f32,
    pub elapsed: f32,
}

// ---------------------------------------------------------------------------
// Interaction events
// ---------------------------------------------------------------------------

/// Events generated by the interaction system.
#[derive(Debug, Clone)]
pub enum InteractionEvent {
    /// Player started looking at an interactable.
    FocusEnter { entity: EntityId, interaction_type: InteractionType },
    /// Player stopped looking at an interactable.
    FocusExit { entity: EntityId },
    /// Object was grabbed.
    Grabbed { entity: EntityId },
    /// Object was dropped.
    Dropped { entity: EntityId, position: Vec3 },
    /// Object was thrown.
    Thrown { entity: EntityId, velocity: Vec3 },
    /// Object was pushed.
    Pushed { entity: EntityId, force: Vec3 },
    /// Object was pulled.
    Pulled { entity: EntityId, force: Vec3 },
    /// Inspection started.
    InspectionStarted { entity: EntityId },
    /// Inspection ended.
    InspectionEnded { entity: EntityId },
    /// Spring joint broke (object too far away).
    JointBroken { entity: EntityId },
    /// Object hit something while being carried.
    CarriedImpact { entity: EntityId, impact_force: f32 },
}

// ---------------------------------------------------------------------------
// PhysicsInteraction system
// ---------------------------------------------------------------------------

/// Input from the player for physics interaction.
#[derive(Debug, Clone, Default)]
pub struct InteractionInput {
    /// Grab/drop button pressed this frame.
    pub grab_pressed: bool,
    /// Throw button pressed.
    pub throw_pressed: bool,
    /// Push forward.
    pub push_forward: f32,
    /// Pull backward.
    pub pull_backward: f32,
    /// Rotate held object (x, y) in inspection mode.
    pub rotate_x: f32,
    pub rotate_y: f32,
    /// Zoom in inspection mode.
    pub zoom: f32,
    /// Inspect button pressed.
    pub inspect_pressed: bool,
    /// Alternative use (e.g., turn a valve).
    pub alt_use: bool,
}

/// Configuration for the interaction system.
#[derive(Debug, Clone)]
pub struct PhysicsInteractionConfig {
    /// Maximum distance to reach for objects.
    pub max_reach: f32,
    /// How quickly the held object follows the target.
    pub spring_stiffness: f32,
    /// Damping for the spring joint.
    pub spring_damping: f32,
    /// Maximum carry mass.
    pub max_carry_mass: f32,
    /// Throw force multiplier.
    pub throw_multiplier: f32,
    /// Push force.
    pub push_force: f32,
    /// Pull force.
    pub pull_force: f32,
    /// Hold distance from camera.
    pub hold_distance: f32,
    /// Time to complete a grab (reach animation).
    pub grab_time: f32,
    /// Inspection configuration.
    pub inspection: InspectionConfig,
}

impl Default for PhysicsInteractionConfig {
    fn default() -> Self {
        Self {
            max_reach: 3.0,
            spring_stiffness: 50.0,
            spring_damping: 10.0,
            max_carry_mass: 50.0,
            throw_multiplier: 15.0,
            push_force: 10.0,
            pull_force: 8.0,
            hold_distance: 1.5,
            grab_time: 0.15,
            inspection: InspectionConfig::default(),
        }
    }
}

/// The main physics interaction system.
pub struct PhysicsInteractionSystem {
    config: PhysicsInteractionConfig,
    objects: HashMap<u64, InteractableObject>,
    state: GrabState,
    held: Option<HeldObject>,
    inspection: Option<InspectionState>,
    focused_entity: Option<EntityId>,
    events: Vec<InteractionEvent>,
    total_time: f32,
}

impl PhysicsInteractionSystem {
    pub fn new() -> Self {
        Self::with_config(PhysicsInteractionConfig::default())
    }

    pub fn with_config(config: PhysicsInteractionConfig) -> Self {
        Self {
            config,
            objects: HashMap::new(),
            state: GrabState::Idle,
            held: None,
            inspection: None,
            focused_entity: None,
            events: Vec::new(),
            total_time: 0.0,
        }
    }

    pub fn config(&self) -> &PhysicsInteractionConfig { &self.config }
    pub fn config_mut(&mut self) -> &mut PhysicsInteractionConfig { &mut self.config }
    pub fn state(&self) -> GrabState { self.state }
    pub fn focused_entity(&self) -> Option<EntityId> { self.focused_entity }
    pub fn is_holding(&self) -> bool { self.held.is_some() }
    pub fn is_inspecting(&self) -> bool { self.inspection.is_some() }

    /// Register an interactable object.
    pub fn register_object(&mut self, obj: InteractableObject) {
        self.objects.insert(obj.entity.0, obj);
    }

    /// Unregister an object.
    pub fn unregister_object(&mut self, entity: EntityId) {
        self.objects.remove(&entity.0);
        if self.focused_entity == Some(entity) {
            self.focused_entity = None;
        }
        if self.held.as_ref().map(|h| h.entity) == Some(entity) {
            self.drop_object();
        }
    }

    /// Get the interaction prompt for the focused object.
    pub fn interaction_prompt(&self) -> Option<&str> {
        let entity = self.focused_entity?;
        self.objects.get(&entity.0).map(|o| o.interaction_prompt.as_str())
    }

    /// Get the held entity.
    pub fn held_entity(&self) -> Option<EntityId> {
        self.held.as_ref().map(|h| h.entity)
    }

    /// Consume pending events.
    pub fn drain_events(&mut self) -> Vec<InteractionEvent> {
        std::mem::take(&mut self.events)
    }

    /// Update the system. `hit` is the result of a raycast from the camera center.
    pub fn update(
        &mut self,
        dt: f32,
        camera_position: Vec3,
        camera_forward: Vec3,
        input: &InteractionInput,
        hit: Option<RayHit>,
    ) {
        self.total_time += dt;

        // Update focus.
        self.update_focus(hit);

        // Handle state transitions.
        match self.state {
            GrabState::Idle => {
                if input.grab_pressed {
                    if let Some(entity) = self.focused_entity {
                        self.try_grab(entity, camera_position, camera_forward);
                    }
                }
                if input.inspect_pressed {
                    if let Some(entity) = self.focused_entity {
                        self.try_inspect(entity, camera_position, camera_forward);
                    }
                }
                // Push/pull without grabbing.
                if input.push_forward > 0.0 || input.pull_backward > 0.0 {
                    if let Some(entity) = self.focused_entity {
                        self.push_pull(entity, camera_forward, input.push_forward, input.pull_backward);
                    }
                }
            }
            GrabState::Reaching => {
                if let Some(ref mut held) = self.held {
                    held.grab_time += dt;
                    if held.grab_time >= self.config.grab_time {
                        self.state = GrabState::Holding;
                    }
                }
            }
            GrabState::Holding => {
                if input.grab_pressed {
                    self.drop_object();
                } else if input.throw_pressed {
                    self.throw_object(camera_forward);
                } else if input.inspect_pressed {
                    self.start_inspection();
                } else {
                    self.update_held_object(dt, camera_position, camera_forward);
                }
            }
            GrabState::Inspecting => {
                if input.inspect_pressed || input.grab_pressed {
                    self.end_inspection();
                } else {
                    self.update_inspection(dt, input, camera_position, camera_forward);
                }
            }
            GrabState::Throwing => {
                // One-frame state, auto-transition back to idle.
                self.state = GrabState::Idle;
            }
            GrabState::Pushing => {
                self.state = GrabState::Idle;
            }
        }
    }

    fn update_focus(&mut self, hit: Option<RayHit>) {
        let new_focus = hit.and_then(|h| {
            let obj = self.objects.get(&h.entity.0)?;
            if h.distance <= obj.max_grab_distance {
                Some(h.entity)
            } else {
                None
            }
        });

        if new_focus != self.focused_entity {
            if let Some(old) = self.focused_entity {
                self.events.push(InteractionEvent::FocusExit { entity: old });
            }
            if let Some(new_ent) = new_focus {
                if let Some(obj) = self.objects.get(&new_ent.0) {
                    self.events.push(InteractionEvent::FocusEnter {
                        entity: new_ent,
                        interaction_type: obj.interaction_type,
                    });
                }
            }
            self.focused_entity = new_focus;
        }
    }

    fn try_grab(&mut self, entity: EntityId, camera_pos: Vec3, camera_fwd: Vec3) {
        let obj = match self.objects.get(&entity.0) {
            Some(o) => o.clone(),
            None => return,
        };

        if obj.interaction_type != InteractionType::Grabbable
            && obj.interaction_type != InteractionType::Throwable {
            return;
        }

        if obj.mass > self.config.max_carry_mass {
            return;
        }

        let hold_pos = camera_pos + camera_fwd * self.config.hold_distance;
        let mut spring = SpringJoint::new(entity, hold_pos, hold_pos);
        spring.stiffness = self.config.spring_stiffness;
        spring.damping = self.config.spring_damping;

        self.held = Some(HeldObject {
            entity,
            spring,
            original_position: hold_pos,
            grab_offset: Vec3::ZERO,
            grab_time: 0.0,
            total_hold_time: 0.0,
            interaction_type: obj.interaction_type,
        });

        self.state = GrabState::Reaching;
        self.events.push(InteractionEvent::Grabbed { entity });
    }

    fn try_inspect(&mut self, entity: EntityId, camera_pos: Vec3, camera_fwd: Vec3) {
        let obj = match self.objects.get(&entity.0) {
            Some(o) => o.clone(),
            None => return,
        };

        if obj.interaction_type != InteractionType::Inspectable
            && obj.interaction_type != InteractionType::Grabbable {
            return;
        }

        self.inspection = Some(InspectionState {
            entity,
            rotation_x: 0.0,
            rotation_y: 0.0,
            zoom: self.config.inspection.inspect_distance,
            elapsed: 0.0,
        });

        self.state = GrabState::Inspecting;
        self.events.push(InteractionEvent::InspectionStarted { entity });
    }

    fn update_held_object(&mut self, dt: f32, camera_pos: Vec3, camera_fwd: Vec3) {
        if let Some(ref mut held) = self.held {
            held.total_hold_time += dt;
            let target = camera_pos + camera_fwd * self.config.hold_distance;
            held.spring.set_target(target);

            let mass = self.objects.get(&held.entity.0)
                .map(|o| o.mass)
                .unwrap_or(1.0);

            if held.spring.update(dt, mass).is_none() {
                // Joint broke.
                let entity = held.entity;
                let pos = held.spring.object_position;
                self.events.push(InteractionEvent::JointBroken { entity });
                self.events.push(InteractionEvent::Dropped { entity, position: pos });
                self.held = None;
                self.state = GrabState::Idle;
            }
        }
    }

    fn drop_object(&mut self) {
        if let Some(held) = self.held.take() {
            self.events.push(InteractionEvent::Dropped {
                entity: held.entity,
                position: held.spring.object_position,
            });
        }
        self.state = GrabState::Idle;
    }

    fn throw_object(&mut self, camera_fwd: Vec3) {
        if let Some(held) = self.held.take() {
            let throw_force = self.objects.get(&held.entity.0)
                .map(|o| o.throw_force)
                .unwrap_or(10.0);
            let velocity = camera_fwd * throw_force * self.config.throw_multiplier;
            self.events.push(InteractionEvent::Thrown {
                entity: held.entity,
                velocity,
            });
        }
        self.state = GrabState::Throwing;
    }

    fn push_pull(&mut self, entity: EntityId, camera_fwd: Vec3, push: f32, pull: f32) {
        let obj = match self.objects.get(&entity.0) {
            Some(o) => o.clone(),
            None => return,
        };

        if obj.interaction_type != InteractionType::Pushable
            && obj.interaction_type != InteractionType::Grabbable {
            return;
        }

        if push > 0.0 {
            let force = camera_fwd * self.config.push_force * push;
            self.events.push(InteractionEvent::Pushed { entity, force });
        }
        if pull > 0.0 {
            let force = camera_fwd * (-self.config.pull_force * pull);
            self.events.push(InteractionEvent::Pulled { entity, force });
        }
        self.state = GrabState::Pushing;
    }

    fn start_inspection(&mut self) {
        if let Some(ref held) = self.held {
            self.inspection = Some(InspectionState {
                entity: held.entity,
                rotation_x: 0.0,
                rotation_y: 0.0,
                zoom: self.config.inspection.inspect_distance,
                elapsed: 0.0,
            });
            self.state = GrabState::Inspecting;
            self.events.push(InteractionEvent::InspectionStarted { entity: held.entity });
        }
    }

    fn end_inspection(&mut self) {
        if let Some(inspection) = self.inspection.take() {
            self.events.push(InteractionEvent::InspectionEnded { entity: inspection.entity });
        }
        if self.held.is_some() {
            self.state = GrabState::Holding;
        } else {
            self.state = GrabState::Idle;
        }
    }

    fn update_inspection(&mut self, dt: f32, input: &InteractionInput, camera_pos: Vec3, camera_fwd: Vec3) {
        if let Some(ref mut insp) = self.inspection {
            insp.elapsed += dt;
            insp.rotation_x += input.rotate_x * self.config.inspection.rotation_speed * dt;
            insp.rotation_y += input.rotate_y * self.config.inspection.rotation_speed * dt;
            insp.zoom = (insp.zoom + input.zoom * dt).clamp(
                self.config.inspection.zoom_min,
                self.config.inspection.zoom_max,
            );
        }
    }

    /// Get the position where the held object should be rendered.
    pub fn held_object_position(&self) -> Option<Vec3> {
        self.held.as_ref().map(|h| h.spring.object_position)
    }

    /// Get the inspection state if inspecting.
    pub fn inspection_state(&self) -> Option<&InspectionState> {
        self.inspection.as_ref()
    }
}

impl Default for PhysicsInteractionSystem {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spring_joint() {
        let entity = EntityId(1);
        let mut spring = SpringJoint::new(entity, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let force = spring.update(1.0 / 60.0, 1.0);
        assert!(force.is_some());
        // Object should move toward target.
        assert!(spring.object_position.x > 0.0);
    }

    #[test]
    fn test_spring_break() {
        let entity = EntityId(1);
        let mut spring = SpringJoint::new(entity, Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0));
        spring.break_distance = 2.0;
        let force = spring.update(1.0 / 60.0, 1.0);
        assert!(force.is_none()); // Should break
    }

    #[test]
    fn test_interaction_system() {
        let mut sys = PhysicsInteractionSystem::new();
        let entity = EntityId(42);
        sys.register_object(InteractableObject::grabbable(entity, 5.0));

        let camera_pos = Vec3::new(0.0, 1.0, 0.0);
        let camera_fwd = Vec3::new(0.0, 0.0, -1.0);

        // Focus on object.
        let hit = RayHit {
            entity,
            point: Vec3::new(0.0, 1.0, -1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            distance: 1.0,
        };
        sys.update(1.0 / 60.0, camera_pos, camera_fwd, &InteractionInput::default(), Some(hit));
        assert_eq!(sys.focused_entity(), Some(entity));

        // Grab.
        let mut input = InteractionInput::default();
        input.grab_pressed = true;
        sys.update(1.0 / 60.0, camera_pos, camera_fwd, &input, Some(hit));
        assert_eq!(sys.state(), GrabState::Reaching);

        // Wait for grab to complete.
        let default_input = InteractionInput::default();
        for _ in 0..20 {
            sys.update(1.0 / 60.0, camera_pos, camera_fwd, &default_input, Some(hit));
        }
        assert_eq!(sys.state(), GrabState::Holding);
        assert!(sys.is_holding());
    }

    #[test]
    fn test_throw() {
        let mut sys = PhysicsInteractionSystem::new();
        let entity = EntityId(1);
        sys.register_object(InteractableObject::grabbable(entity, 2.0));

        let pos = Vec3::ZERO;
        let fwd = Vec3::new(0.0, 0.0, -1.0);
        let hit = RayHit { entity, point: Vec3::new(0.0, 0.0, -1.0), normal: Vec3::UP, distance: 1.0 };

        // Grab.
        sys.update(0.0, pos, fwd, &InteractionInput { grab_pressed: true, ..Default::default() }, Some(hit));
        for _ in 0..20 { sys.update(1.0/60.0, pos, fwd, &InteractionInput::default(), Some(hit)); }

        // Throw.
        sys.update(1.0/60.0, pos, fwd, &InteractionInput { throw_pressed: true, ..Default::default() }, Some(hit));
        let events = sys.drain_events();
        assert!(events.iter().any(|e| matches!(e, InteractionEvent::Thrown { .. })));
    }
}
