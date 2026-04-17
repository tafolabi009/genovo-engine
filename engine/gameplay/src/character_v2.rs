// engine/gameplay/src/character_v2.rs
//
// Enhanced character controller: wall running, wall jumping, ledge grabbing, sliding, dashing, double jump, ground pound, grapple hook, swimming.

use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x+self.y*r.y+self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self{x:self.y*r.z-self.z*r.y,y:self.z*r.x-self.x*r.z,z:self.x*r.y-self.y*r.x} }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f32 { self.dot(self) }
    pub fn normalize(self) -> Self { let l=self.length(); if l<1e-12{Self::ZERO}else{Self{x:self.x/l,y:self.y/l,z:self.z/l}} }
    pub fn scale(self, s: f32) -> Self { Self{x:self.x*s,y:self.y*s,z:self.z*s} }
    pub fn add(self, r: Self) -> Self { Self{x:self.x+r.x,y:self.y+r.y,z:self.z+r.z} }
    pub fn sub(self, r: Self) -> Self { Self{x:self.x-r.x,y:self.y-r.y,z:self.z-r.z} }
    pub fn neg(self) -> Self { Self{x:-self.x,y:-self.y,z:-self.z} }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharacterMovementState {
    Grounded, Airborne, WallRunning, WallSliding, LedgeGrabbing,
    Sliding, Dashing, GroundPounding, Grappling, Swimming, Diving,
    Climbing, WallJumping,
}

#[derive(Debug, Clone)]
pub struct CharacterV2Config {
    pub walk_speed: f32, pub run_speed: f32, pub sprint_speed: f32,
    pub jump_height: f32, pub double_jump_height: f32,
    pub max_jumps: u32, pub air_control: f32,
    pub gravity: f32, pub max_fall_speed: f32,
    pub wall_run_speed: f32, pub wall_run_duration: f32,
    pub wall_run_min_height: f32, pub wall_jump_force: Vec3,
    pub slide_speed: f32, pub slide_duration: f32, pub slide_cooldown: f32,
    pub dash_speed: f32, pub dash_duration: f32, pub dash_cooldown: f32,
    pub ground_pound_speed: f32, pub ground_pound_bounce: f32,
    pub grapple_speed: f32, pub grapple_max_length: f32,
    pub swim_speed: f32, pub dive_speed: f32,
    pub ledge_grab_reach: f32, pub ledge_climb_speed: f32,
    pub step_height: f32, pub slope_limit: f32,
    pub coyote_time: f32, pub jump_buffer_time: f32,
}

impl Default for CharacterV2Config {
    fn default() -> Self {
        Self {
            walk_speed: 3.0, run_speed: 6.0, sprint_speed: 9.0,
            jump_height: 1.5, double_jump_height: 1.0,
            max_jumps: 2, air_control: 0.3,
            gravity: 20.0, max_fall_speed: 50.0,
            wall_run_speed: 7.0, wall_run_duration: 1.0,
            wall_run_min_height: 1.0, wall_jump_force: Vec3::new(5.0, 8.0, 0.0),
            slide_speed: 10.0, slide_duration: 0.8, slide_cooldown: 1.0,
            dash_speed: 20.0, dash_duration: 0.2, dash_cooldown: 2.0,
            ground_pound_speed: 30.0, ground_pound_bounce: 5.0,
            grapple_speed: 15.0, grapple_max_length: 30.0,
            swim_speed: 4.0, dive_speed: 3.0,
            ledge_grab_reach: 0.5, ledge_climb_speed: 2.0,
            step_height: 0.3, slope_limit: 0.78,
            coyote_time: 0.15, jump_buffer_time: 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CharacterV2Input {
    pub move_dir: Vec3, pub look_dir: Vec3,
    pub jump: bool, pub crouch: bool, pub sprint: bool,
    pub dash: bool, pub grapple: bool, pub interact: bool,
}

impl Default for CharacterV2Input {
    fn default() -> Self {
        Self {
            move_dir: Vec3::ZERO, look_dir: Vec3::new(0.0, 0.0, -1.0),
            jump: false, crouch: false, sprint: false,
            dash: false, grapple: false, interact: false,
        }
    }
}

/// Ground detection result.
#[derive(Debug, Clone)]
pub struct GroundInfoV2 {
    pub grounded: bool,
    pub ground_normal: Vec3,
    pub ground_point: Vec3,
    pub slope_angle: f32,
    pub surface_type: SurfaceTypeV2,
    pub moving_platform_velocity: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceTypeV2 { Default, Ice, Sand, Water, Lava, Bounce, Conveyor }

impl Default for GroundInfoV2 {
    fn default() -> Self {
        Self {
            grounded: false, ground_normal: Vec3::new(0.0, 1.0, 0.0),
            ground_point: Vec3::ZERO, slope_angle: 0.0,
            surface_type: SurfaceTypeV2::Default,
            moving_platform_velocity: Vec3::ZERO,
        }
    }
}

/// Wall detection result.
#[derive(Debug, Clone)]
pub struct WallInfo {
    pub touching_wall: bool,
    pub wall_normal: Vec3,
    pub wall_point: Vec3,
    pub wall_side: WallSide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallSide { Left, Right, Front, Back }

impl Default for WallInfo {
    fn default() -> Self {
        Self { touching_wall: false, wall_normal: Vec3::ZERO, wall_point: Vec3::ZERO, wall_side: WallSide::Left }
    }
}

/// Grapple state.
#[derive(Debug, Clone)]
pub struct GrappleState {
    pub active: bool,
    pub target: Vec3,
    pub current_length: f32,
    pub max_length: f32,
    pub attached: bool,
}

impl Default for GrappleState {
    fn default() -> Self {
        Self { active: false, target: Vec3::ZERO, current_length: 0.0, max_length: 30.0, attached: false }
    }
}

/// Enhanced character controller.
pub struct CharacterControllerV2 {
    pub config: CharacterV2Config,
    pub position: Vec3,
    pub velocity: Vec3,
    pub state: CharacterMovementState,
    pub ground_info: GroundInfoV2,
    pub wall_info: WallInfo,
    pub grapple: GrappleState,
    pub jumps_remaining: u32,
    pub coyote_timer: f32,
    pub jump_buffer_timer: f32,
    pub wall_run_timer: f32,
    pub slide_timer: f32,
    pub slide_cooldown_timer: f32,
    pub dash_timer: f32,
    pub dash_cooldown_timer: f32,
    pub dash_direction: Vec3,
    pub is_sprinting: bool,
    pub is_crouching: bool,
    pub height: f32,
    pub crouch_height: f32,
    pub normal_height: f32,
    pub swim_depth: f32,
    pub prev_state: CharacterMovementState,
    pub state_time: f32,
    pub total_air_time: f32,
}

impl CharacterControllerV2 {
    pub fn new(config: CharacterV2Config, position: Vec3) -> Self {
        Self {
            config, position, velocity: Vec3::ZERO,
            state: CharacterMovementState::Grounded,
            ground_info: GroundInfoV2::default(),
            wall_info: WallInfo::default(),
            grapple: GrappleState::default(),
            jumps_remaining: 2, coyote_timer: 0.0, jump_buffer_timer: 0.0,
            wall_run_timer: 0.0, slide_timer: 0.0, slide_cooldown_timer: 0.0,
            dash_timer: 0.0, dash_cooldown_timer: 0.0, dash_direction: Vec3::ZERO,
            is_sprinting: false, is_crouching: false,
            height: 1.8, crouch_height: 0.9, normal_height: 1.8,
            swim_depth: 0.0, prev_state: CharacterMovementState::Grounded,
            state_time: 0.0, total_air_time: 0.0,
        }
    }

    /// Main update tick.
    pub fn update(&mut self, input: &CharacterV2Input, dt: f32) {
        self.update_timers(dt);
        self.update_state_transitions(input, dt);
        self.apply_movement(input, dt);
        self.apply_gravity(dt);
        self.position = self.position.add(self.velocity.scale(dt));
    }

    fn update_timers(&mut self, dt: f32) {
        if self.coyote_timer > 0.0 { self.coyote_timer -= dt; }
        if self.jump_buffer_timer > 0.0 { self.jump_buffer_timer -= dt; }
        if self.slide_cooldown_timer > 0.0 { self.slide_cooldown_timer -= dt; }
        if self.dash_cooldown_timer > 0.0 { self.dash_cooldown_timer -= dt; }
        self.state_time += dt;
    }

    fn update_state_transitions(&mut self, input: &CharacterV2Input, dt: f32) {
        self.prev_state = self.state;

        // Jump buffering
        if input.jump { self.jump_buffer_timer = self.config.jump_buffer_time; }

        match self.state {
            CharacterMovementState::Grounded => {
                self.total_air_time = 0.0;
                self.jumps_remaining = self.config.max_jumps;

                if self.jump_buffer_timer > 0.0 {
                    self.do_jump(self.config.jump_height);
                    self.state = CharacterMovementState::Airborne;
                } else if input.crouch && self.velocity.length() > self.config.walk_speed && self.slide_cooldown_timer <= 0.0 {
                    self.state = CharacterMovementState::Sliding;
                    self.slide_timer = self.config.slide_duration;
                    self.state_time = 0.0;
                } else if input.dash && self.dash_cooldown_timer <= 0.0 {
                    self.start_dash(input);
                }
            }
            CharacterMovementState::Airborne => {
                self.total_air_time += dt;
                if self.ground_info.grounded {
                    self.state = CharacterMovementState::Grounded;
                    self.state_time = 0.0;
                } else if self.wall_info.touching_wall && self.velocity.y < 0.0 {
                    if input.move_dir.length_sq() > 0.1 {
                        self.state = CharacterMovementState::WallRunning;
                        self.wall_run_timer = self.config.wall_run_duration;
                    } else {
                        self.state = CharacterMovementState::WallSliding;
                    }
                    self.state_time = 0.0;
                } else if input.jump && self.jumps_remaining > 0 {
                    self.do_jump(self.config.double_jump_height);
                    self.jumps_remaining -= 1;
                } else if input.crouch {
                    self.state = CharacterMovementState::GroundPounding;
                    self.velocity = Vec3::new(0.0, -self.config.ground_pound_speed, 0.0);
                    self.state_time = 0.0;
                } else if input.dash && self.dash_cooldown_timer <= 0.0 {
                    self.start_dash(input);
                }
            }
            CharacterMovementState::WallRunning => {
                self.wall_run_timer -= dt;
                if !self.wall_info.touching_wall || self.wall_run_timer <= 0.0 {
                    self.state = CharacterMovementState::Airborne;
                    self.state_time = 0.0;
                } else if input.jump {
                    // Wall jump
                    let jump_dir = self.wall_info.wall_normal.add(Vec3::new(0.0, 1.0, 0.0)).normalize();
                    self.velocity = jump_dir.scale(self.config.wall_jump_force.length());
                    self.state = CharacterMovementState::Airborne;
                    self.jumps_remaining = 1;
                    self.state_time = 0.0;
                }
            }
            CharacterMovementState::WallSliding => {
                self.velocity.y = (self.velocity.y).max(-2.0); // slow fall
                if !self.wall_info.touching_wall || self.ground_info.grounded {
                    self.state = if self.ground_info.grounded { CharacterMovementState::Grounded } else { CharacterMovementState::Airborne };
                    self.state_time = 0.0;
                } else if input.jump {
                    let jump_dir = self.wall_info.wall_normal.add(Vec3::new(0.0, 1.0, 0.0)).normalize();
                    self.velocity = jump_dir.scale(self.config.wall_jump_force.length());
                    self.state = CharacterMovementState::Airborne;
                    self.state_time = 0.0;
                }
            }
            CharacterMovementState::Sliding => {
                self.slide_timer -= dt;
                if self.slide_timer <= 0.0 || input.jump {
                    self.state = if input.jump { CharacterMovementState::Airborne } else { CharacterMovementState::Grounded };
                    if input.jump { self.do_jump(self.config.jump_height * 0.8); }
                    self.slide_cooldown_timer = self.config.slide_cooldown;
                    self.state_time = 0.0;
                }
            }
            CharacterMovementState::Dashing => {
                self.dash_timer -= dt;
                if self.dash_timer <= 0.0 {
                    self.state = if self.ground_info.grounded { CharacterMovementState::Grounded } else { CharacterMovementState::Airborne };
                    self.dash_cooldown_timer = self.config.dash_cooldown;
                    self.state_time = 0.0;
                }
            }
            CharacterMovementState::GroundPounding => {
                if self.ground_info.grounded {
                    self.velocity.y = self.config.ground_pound_bounce;
                    self.state = CharacterMovementState::Airborne;
                    self.state_time = 0.0;
                }
            }
            CharacterMovementState::Swimming => {
                if self.swim_depth <= 0.0 {
                    self.state = CharacterMovementState::Grounded;
                    self.state_time = 0.0;
                } else if input.crouch {
                    self.state = CharacterMovementState::Diving;
                    self.state_time = 0.0;
                }
            }
            CharacterMovementState::Diving => {
                if self.swim_depth <= 0.0 {
                    self.state = CharacterMovementState::Swimming;
                    self.state_time = 0.0;
                }
            }
            _ => {}
        }
    }

    fn apply_movement(&mut self, input: &CharacterV2Input, dt: f32) {
        let speed = match self.state {
            CharacterMovementState::Grounded => {
                if input.sprint { self.config.sprint_speed }
                else if input.crouch { self.config.walk_speed * 0.5 }
                else { self.config.run_speed }
            }
            CharacterMovementState::Airborne => self.config.run_speed * self.config.air_control,
            CharacterMovementState::WallRunning => self.config.wall_run_speed,
            CharacterMovementState::Sliding => self.config.slide_speed,
            CharacterMovementState::Dashing => self.config.dash_speed,
            CharacterMovementState::Swimming => self.config.swim_speed,
            CharacterMovementState::Diving => self.config.dive_speed,
            _ => 0.0,
        };

        match self.state {
            CharacterMovementState::Dashing => {
                self.velocity = self.dash_direction.scale(speed);
            }
            CharacterMovementState::Sliding => {
                let forward = input.look_dir;
                self.velocity.x = forward.x * speed;
                self.velocity.z = forward.z * speed;
            }
            CharacterMovementState::Grappling => {
                if self.grapple.attached {
                    let to_target = self.grapple.target.sub(self.position).normalize();
                    self.velocity = to_target.scale(self.config.grapple_speed);
                    if self.position.distance(self.grapple.target) < 1.0 {
                        self.grapple.active = false;
                        self.state = CharacterMovementState::Airborne;
                    }
                }
            }
            _ => {
                if input.move_dir.length_sq() > 0.01 {
                    let desired = input.move_dir.normalize().scale(speed);
                    let accel = if self.ground_info.grounded { 15.0 } else { 5.0 };
                    self.velocity.x = approach(self.velocity.x, desired.x, accel * dt);
                    self.velocity.z = approach(self.velocity.z, desired.z, accel * dt);
                } else if self.ground_info.grounded {
                    self.velocity.x = approach(self.velocity.x, 0.0, 20.0 * dt);
                    self.velocity.z = approach(self.velocity.z, 0.0, 20.0 * dt);
                }
            }
        }
    }

    fn apply_gravity(&mut self, dt: f32) {
        match self.state {
            CharacterMovementState::Grounded | CharacterMovementState::Sliding |
            CharacterMovementState::Dashing | CharacterMovementState::Grappling => {}
            CharacterMovementState::WallRunning => {
                self.velocity.y -= self.config.gravity * 0.1 * dt; // reduced gravity
            }
            CharacterMovementState::WallSliding => {
                self.velocity.y = self.velocity.y.max(-2.0);
            }
            CharacterMovementState::Swimming | CharacterMovementState::Diving => {
                self.velocity.y -= self.config.gravity * 0.05 * dt;
            }
            _ => {
                self.velocity.y -= self.config.gravity * dt;
                self.velocity.y = self.velocity.y.max(-self.config.max_fall_speed);
            }
        }
    }

    fn do_jump(&mut self, height: f32) {
        self.velocity.y = (2.0 * self.config.gravity * height).sqrt();
        self.jump_buffer_timer = 0.0;
    }

    fn start_dash(&mut self, input: &CharacterV2Input) {
        self.state = CharacterMovementState::Dashing;
        self.dash_timer = self.config.dash_duration;
        self.dash_direction = if input.move_dir.length_sq() > 0.01 {
            input.move_dir.normalize()
        } else {
            input.look_dir
        };
        self.velocity.y = 0.0;
        self.state_time = 0.0;
    }

    pub fn set_ground_info(&mut self, info: GroundInfoV2) { self.ground_info = info; }
    pub fn set_wall_info(&mut self, info: WallInfo) { self.wall_info = info; }
    pub fn set_swim_depth(&mut self, depth: f32) { self.swim_depth = depth; }
}

fn approach(current: f32, target: f32, max_delta: f32) -> f32 {
    if (target - current).abs() <= max_delta { target }
    else { current + (target - current).signum() * max_delta }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_creation() {
        let c = CharacterControllerV2::new(CharacterV2Config::default(), Vec3::ZERO);
        assert_eq!(c.state, CharacterMovementState::Grounded);
        assert_eq!(c.jumps_remaining, 2);
    }

    #[test]
    fn test_jump() {
        let mut c = CharacterControllerV2::new(CharacterV2Config::default(), Vec3::ZERO);
        c.update(&CharacterV2Input { jump: true, ..Default::default() }, 0.016);
        assert!(c.velocity.y > 0.0);
    }

    #[test]
    fn test_gravity() {
        let mut c = CharacterControllerV2::new(CharacterV2Config::default(), Vec3::new(0.0, 10.0, 0.0));
        c.state = CharacterMovementState::Airborne;
        let initial_y = c.velocity.y;
        c.update(&CharacterV2Input::default(), 0.1);
        assert!(c.velocity.y < initial_y);
    }
}

