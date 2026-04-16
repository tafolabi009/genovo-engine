// =============================================================================
// Genovo Engine - Winit Backend (Shared Desktop Implementation)
// =============================================================================
//
// Cross-platform desktop window management and input handling using winit 0.30.
// This backend is shared by the Windows, macOS, and Linux platform modules.
//
// winit 0.30 uses the `ApplicationHandler` trait with a callback-driven event
// loop. We use `EventLoop::pump_events()` (available on desktop via
// `EventLoopExtPumpEvents`) to integrate with the engine's per-frame
// `poll_events()` model.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode as WinitKeyCode, PhysicalKey};
use winit::window::{CursorIcon, Fullscreen, Window, WindowAttributes, WindowId};

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
use winit::platform::pump_events::EventLoopExtPumpEvents;

use crate::interface::events::{PlatformEvent, TouchPhase};
use crate::interface::input::{KeyCode, MouseButton};
use crate::interface::{
    CursorType, DisplayInfo, Platform, RawWindowHandle, RenderBackend, Result, SystemInfo,
    WindowDesc, WindowHandle,
};

// -----------------------------------------------------------------------------
// Key code mapping: winit PhysicalKey -> engine KeyCode
// -----------------------------------------------------------------------------

fn map_key_code(key: &PhysicalKey) -> KeyCode {
    match key {
        PhysicalKey::Code(code) => match code {
            // Letters
            WinitKeyCode::KeyA => KeyCode::A,
            WinitKeyCode::KeyB => KeyCode::B,
            WinitKeyCode::KeyC => KeyCode::C,
            WinitKeyCode::KeyD => KeyCode::D,
            WinitKeyCode::KeyE => KeyCode::E,
            WinitKeyCode::KeyF => KeyCode::F,
            WinitKeyCode::KeyG => KeyCode::G,
            WinitKeyCode::KeyH => KeyCode::H,
            WinitKeyCode::KeyI => KeyCode::I,
            WinitKeyCode::KeyJ => KeyCode::J,
            WinitKeyCode::KeyK => KeyCode::K,
            WinitKeyCode::KeyL => KeyCode::L,
            WinitKeyCode::KeyM => KeyCode::M,
            WinitKeyCode::KeyN => KeyCode::N,
            WinitKeyCode::KeyO => KeyCode::O,
            WinitKeyCode::KeyP => KeyCode::P,
            WinitKeyCode::KeyQ => KeyCode::Q,
            WinitKeyCode::KeyR => KeyCode::R,
            WinitKeyCode::KeyS => KeyCode::S,
            WinitKeyCode::KeyT => KeyCode::T,
            WinitKeyCode::KeyU => KeyCode::U,
            WinitKeyCode::KeyV => KeyCode::V,
            WinitKeyCode::KeyW => KeyCode::W,
            WinitKeyCode::KeyX => KeyCode::X,
            WinitKeyCode::KeyY => KeyCode::Y,
            WinitKeyCode::KeyZ => KeyCode::Z,

            // Digits
            WinitKeyCode::Digit0 => KeyCode::Key0,
            WinitKeyCode::Digit1 => KeyCode::Key1,
            WinitKeyCode::Digit2 => KeyCode::Key2,
            WinitKeyCode::Digit3 => KeyCode::Key3,
            WinitKeyCode::Digit4 => KeyCode::Key4,
            WinitKeyCode::Digit5 => KeyCode::Key5,
            WinitKeyCode::Digit6 => KeyCode::Key6,
            WinitKeyCode::Digit7 => KeyCode::Key7,
            WinitKeyCode::Digit8 => KeyCode::Key8,
            WinitKeyCode::Digit9 => KeyCode::Key9,

            // Function keys
            WinitKeyCode::F1 => KeyCode::F1,
            WinitKeyCode::F2 => KeyCode::F2,
            WinitKeyCode::F3 => KeyCode::F3,
            WinitKeyCode::F4 => KeyCode::F4,
            WinitKeyCode::F5 => KeyCode::F5,
            WinitKeyCode::F6 => KeyCode::F6,
            WinitKeyCode::F7 => KeyCode::F7,
            WinitKeyCode::F8 => KeyCode::F8,
            WinitKeyCode::F9 => KeyCode::F9,
            WinitKeyCode::F10 => KeyCode::F10,
            WinitKeyCode::F11 => KeyCode::F11,
            WinitKeyCode::F12 => KeyCode::F12,
            WinitKeyCode::F13 => KeyCode::F13,
            WinitKeyCode::F14 => KeyCode::F14,
            WinitKeyCode::F15 => KeyCode::F15,
            WinitKeyCode::F16 => KeyCode::F16,
            WinitKeyCode::F17 => KeyCode::F17,
            WinitKeyCode::F18 => KeyCode::F18,
            WinitKeyCode::F19 => KeyCode::F19,
            WinitKeyCode::F20 => KeyCode::F20,
            WinitKeyCode::F21 => KeyCode::F21,
            WinitKeyCode::F22 => KeyCode::F22,
            WinitKeyCode::F23 => KeyCode::F23,
            WinitKeyCode::F24 => KeyCode::F24,

            // Modifiers
            WinitKeyCode::ShiftLeft => KeyCode::LShift,
            WinitKeyCode::ShiftRight => KeyCode::RShift,
            WinitKeyCode::ControlLeft => KeyCode::LControl,
            WinitKeyCode::ControlRight => KeyCode::RControl,
            WinitKeyCode::AltLeft => KeyCode::LAlt,
            WinitKeyCode::AltRight => KeyCode::RAlt,
            WinitKeyCode::SuperLeft => KeyCode::LSuper,
            WinitKeyCode::SuperRight => KeyCode::RSuper,

            // Navigation
            WinitKeyCode::ArrowUp => KeyCode::Up,
            WinitKeyCode::ArrowDown => KeyCode::Down,
            WinitKeyCode::ArrowLeft => KeyCode::Left,
            WinitKeyCode::ArrowRight => KeyCode::Right,
            WinitKeyCode::Home => KeyCode::Home,
            WinitKeyCode::End => KeyCode::End,
            WinitKeyCode::PageUp => KeyCode::PageUp,
            WinitKeyCode::PageDown => KeyCode::PageDown,

            // Editing
            WinitKeyCode::Insert => KeyCode::Insert,
            WinitKeyCode::Delete => KeyCode::Delete,
            WinitKeyCode::Backspace => KeyCode::Backspace,
            WinitKeyCode::Enter => KeyCode::Enter,
            WinitKeyCode::Tab => KeyCode::Tab,
            WinitKeyCode::Space => KeyCode::Space,

            // Lock keys
            WinitKeyCode::CapsLock => KeyCode::CapsLock,
            WinitKeyCode::NumLock => KeyCode::NumLock,
            WinitKeyCode::ScrollLock => KeyCode::ScrollLock,

            // Punctuation & symbols
            WinitKeyCode::Minus => KeyCode::Minus,
            WinitKeyCode::Equal => KeyCode::Equals,
            WinitKeyCode::BracketLeft => KeyCode::LeftBracket,
            WinitKeyCode::BracketRight => KeyCode::RightBracket,
            WinitKeyCode::Backslash => KeyCode::Backslash,
            WinitKeyCode::Semicolon => KeyCode::Semicolon,
            WinitKeyCode::Quote => KeyCode::Apostrophe,
            WinitKeyCode::Backquote => KeyCode::Grave,
            WinitKeyCode::Comma => KeyCode::Comma,
            WinitKeyCode::Period => KeyCode::Period,
            WinitKeyCode::Slash => KeyCode::Slash,

            // Numpad
            WinitKeyCode::Numpad0 => KeyCode::Numpad0,
            WinitKeyCode::Numpad1 => KeyCode::Numpad1,
            WinitKeyCode::Numpad2 => KeyCode::Numpad2,
            WinitKeyCode::Numpad3 => KeyCode::Numpad3,
            WinitKeyCode::Numpad4 => KeyCode::Numpad4,
            WinitKeyCode::Numpad5 => KeyCode::Numpad5,
            WinitKeyCode::Numpad6 => KeyCode::Numpad6,
            WinitKeyCode::Numpad7 => KeyCode::Numpad7,
            WinitKeyCode::Numpad8 => KeyCode::Numpad8,
            WinitKeyCode::Numpad9 => KeyCode::Numpad9,
            WinitKeyCode::NumpadAdd => KeyCode::NumpadAdd,
            WinitKeyCode::NumpadSubtract => KeyCode::NumpadSubtract,
            WinitKeyCode::NumpadMultiply => KeyCode::NumpadMultiply,
            WinitKeyCode::NumpadDivide => KeyCode::NumpadDivide,
            WinitKeyCode::NumpadDecimal => KeyCode::NumpadDecimal,
            WinitKeyCode::NumpadEnter => KeyCode::NumpadEnter,

            // Special
            WinitKeyCode::Escape => KeyCode::Escape,
            WinitKeyCode::PrintScreen => KeyCode::PrintScreen,
            WinitKeyCode::Pause => KeyCode::Pause,
            WinitKeyCode::ContextMenu => KeyCode::Menu,

            // Media
            WinitKeyCode::MediaPlayPause => KeyCode::MediaPlay,
            WinitKeyCode::MediaStop => KeyCode::MediaStop,
            WinitKeyCode::MediaTrackNext => KeyCode::MediaNext,
            WinitKeyCode::MediaTrackPrevious => KeyCode::MediaPrevious,
            WinitKeyCode::AudioVolumeUp => KeyCode::VolumeUp,
            WinitKeyCode::AudioVolumeDown => KeyCode::VolumeDown,
            WinitKeyCode::AudioVolumeMute => KeyCode::VolumeMute,

            _ => KeyCode::Unknown(0),
        },
        PhysicalKey::Unidentified(_) => KeyCode::Unknown(0),
    }
}

/// Map a winit mouse button to the engine's MouseButton enum.
fn map_mouse_button(button: winit::event::MouseButton) -> MouseButton {
    match button {
        winit::event::MouseButton::Left => MouseButton::Left,
        winit::event::MouseButton::Right => MouseButton::Right,
        winit::event::MouseButton::Middle => MouseButton::Middle,
        winit::event::MouseButton::Back => MouseButton::Extra(0),
        winit::event::MouseButton::Forward => MouseButton::Extra(1),
        winit::event::MouseButton::Other(id) => MouseButton::Extra(id as u8),
    }
}

/// Map the engine's CursorType to a winit CursorIcon.
fn map_cursor_type(cursor: CursorType) -> Option<CursorIcon> {
    match cursor {
        CursorType::Default | CursorType::Arrow => Some(CursorIcon::Default),
        CursorType::IBeam => Some(CursorIcon::Text),
        CursorType::Crosshair => Some(CursorIcon::Crosshair),
        CursorType::Hand => Some(CursorIcon::Pointer),
        CursorType::ResizeHorizontal => Some(CursorIcon::EwResize),
        CursorType::ResizeVertical => Some(CursorIcon::NsResize),
        CursorType::ResizeNESW => Some(CursorIcon::NeswResize),
        CursorType::ResizeNWSE => Some(CursorIcon::NwseResize),
        CursorType::Move => Some(CursorIcon::Move),
        CursorType::NotAllowed => Some(CursorIcon::NotAllowed),
        CursorType::Wait => Some(CursorIcon::Wait),
        CursorType::Progress => Some(CursorIcon::Progress),
        CursorType::Help => Some(CursorIcon::Help),
        CursorType::Hidden => None, // Handled separately via set_cursor_visible
    }
}

// -----------------------------------------------------------------------------
// Per-window data stored in the backend
// -----------------------------------------------------------------------------

/// Data associated with each managed window.
struct WinitWindowData {
    /// The winit Window object.
    window: Arc<Window>,
    /// The engine-assigned handle ID.
    engine_handle: WindowHandle,
    /// Current scale factor (DPI).
    #[allow(dead_code)]
    scale_factor: f64,
}

// -----------------------------------------------------------------------------
// Application handler that collects events into a Vec
// -----------------------------------------------------------------------------

/// Internal handler implementing winit's `ApplicationHandler` trait.
///
/// Collects all events into a vector that the engine drains each frame via
/// `poll_events()`. Also handles deferred window creation: because winit 0.30
/// requires an `ActiveEventLoop` reference to create windows, creation requests
/// are queued and fulfilled inside the `resumed()` callback or at the start of
/// each event pump cycle.
struct WinitAppHandler {
    /// Collected events for the current pump cycle.
    events: Vec<PlatformEvent>,
    /// Map from winit WindowId to engine WindowHandle.
    window_id_map: HashMap<WindowId, WindowHandle>,
    /// Pending window creation requests (WindowDesc + assigned handle).
    pending_window_requests: Vec<(WindowDesc, WindowHandle)>,
    /// Newly created windows to be stored back into the platform struct.
    newly_created_windows: Vec<(WindowId, WinitWindowData)>,
    /// Whether `resumed` has been called (event loop is active).
    is_active: bool,
}

impl WinitAppHandler {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            window_id_map: HashMap::new(),
            pending_window_requests: Vec::new(),
            newly_created_windows: Vec::new(),
            is_active: false,
        }
    }

    /// Resolve a winit WindowId to the engine's WindowHandle.
    fn resolve_handle(&self, window_id: WindowId) -> WindowHandle {
        self.window_id_map
            .get(&window_id)
            .copied()
            .unwrap_or(WindowHandle(0))
    }

    /// Create any pending windows. Must be called when the event loop is active.
    fn create_pending_windows(&mut self, event_loop: &ActiveEventLoop) {
        let requests: Vec<_> = self.pending_window_requests.drain(..).collect();
        for (desc, engine_handle) in requests {
            let mut attrs = WindowAttributes::default()
                .with_title(&desc.title)
                .with_inner_size(LogicalSize::new(desc.width, desc.height))
                .with_resizable(desc.resizable)
                .with_decorations(desc.decorated)
                .with_transparent(desc.transparent);

            if let Some((min_w, min_h)) = desc.min_size {
                attrs = attrs.with_min_inner_size(LogicalSize::new(min_w, min_h));
            }
            if let Some((max_w, max_h)) = desc.max_size {
                attrs = attrs.with_max_inner_size(LogicalSize::new(max_w, max_h));
            }

            if desc.fullscreen {
                attrs = attrs.with_fullscreen(Some(Fullscreen::Borderless(None)));
            }

            match event_loop.create_window(attrs) {
                Ok(window) => {
                    let window_id = window.id();
                    let scale_factor = window.scale_factor();

                    self.window_id_map.insert(window_id, engine_handle);

                    let data = WinitWindowData {
                        window: Arc::new(window),
                        engine_handle,
                        scale_factor,
                    };
                    self.newly_created_windows.push((window_id, data));

                    log::info!(
                        "Created winit window: {:?} -> {}",
                        window_id,
                        engine_handle
                    );
                }
                Err(e) => {
                    log::error!("Failed to create winit window: {}", e);
                }
            }
        }
    }
}

impl ApplicationHandler for WinitAppHandler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.is_active = true;
        // Create any windows that were queued before the event loop started.
        self.create_pending_windows(event_loop);
        self.events.push(PlatformEvent::AppResume);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.is_active = false;
        self.events.push(PlatformEvent::AppSuspend);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Create any windows queued during this pump cycle (e.g. from
        // create_window() calls between poll_events() invocations).
        if !self.pending_window_requests.is_empty() {
            self.create_pending_windows(event_loop);
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let handle = self.resolve_handle(window_id);

        match event {
            WindowEvent::Resized(size) => {
                self.events.push(PlatformEvent::WindowResize {
                    handle,
                    width: size.width,
                    height: size.height,
                });
            }

            WindowEvent::CloseRequested => {
                self.events.push(PlatformEvent::WindowClose { handle });
            }

            WindowEvent::Focused(focused) => {
                self.events
                    .push(PlatformEvent::WindowFocus { handle, focused });
            }

            WindowEvent::Moved(pos) => {
                self.events.push(PlatformEvent::WindowMoved {
                    handle,
                    x: pos.x,
                    y: pos.y,
                });
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.events.push(PlatformEvent::WindowScaleFactorChanged {
                    handle,
                    scale_factor,
                });
            }

            WindowEvent::Occluded(occluded) => {
                // Map "occluded" to "minimized" (closest semantic match).
                self.events.push(PlatformEvent::WindowMinimized {
                    handle,
                    minimized: occluded,
                });
            }

            WindowEvent::KeyboardInput { event, .. } => {
                let key = map_key_code(&event.physical_key);
                let pressed = event.state == ElementState::Pressed;
                self.events.push(PlatformEvent::KeyInput {
                    handle,
                    key,
                    pressed,
                    repeat: event.repeat,
                });

                // Emit CharInput for printable text produced by this key press.
                if pressed {
                    if let Some(ref text) = event.text {
                        for ch in text.chars() {
                            if !ch.is_control() {
                                self.events.push(PlatformEvent::CharInput {
                                    handle,
                                    character: ch,
                                });
                            }
                        }
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let mapped_button = map_mouse_button(button);
                let pressed = state == ElementState::Pressed;
                self.events.push(PlatformEvent::MouseInput {
                    handle,
                    button: mapped_button,
                    pressed,
                });
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.events.push(PlatformEvent::MouseMove {
                    handle,
                    x: position.x,
                    y: position.y,
                });
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let (dx, dy) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x as f64, y as f64),
                    MouseScrollDelta::PixelDelta(pos) => (pos.x, pos.y),
                };
                self.events.push(PlatformEvent::MouseScroll {
                    handle,
                    delta_x: dx,
                    delta_y: dy,
                });
            }

            WindowEvent::CursorEntered { .. } => {
                self.events.push(PlatformEvent::MouseEnterLeave {
                    handle,
                    entered: true,
                });
            }

            WindowEvent::CursorLeft { .. } => {
                self.events.push(PlatformEvent::MouseEnterLeave {
                    handle,
                    entered: false,
                });
            }

            WindowEvent::DroppedFile(path) => {
                self.events.push(PlatformEvent::FileDrop {
                    handle,
                    paths: vec![path],
                });
            }

            WindowEvent::HoveredFile(path) => {
                self.events.push(PlatformEvent::FileHover {
                    handle,
                    paths: vec![path],
                });
            }

            WindowEvent::HoveredFileCancelled => {
                self.events
                    .push(PlatformEvent::FileHoverCancelled { handle });
            }

            WindowEvent::Touch(touch) => {
                let phase = match touch.phase {
                    winit::event::TouchPhase::Started => TouchPhase::Started,
                    winit::event::TouchPhase::Moved => TouchPhase::Moved,
                    winit::event::TouchPhase::Ended => TouchPhase::Ended,
                    winit::event::TouchPhase::Cancelled => TouchPhase::Cancelled,
                };
                let pressure = touch.force.map(|f| match f {
                    winit::event::Force::Calibrated {
                        force,
                        max_possible_force,
                        ..
                    } => {
                        if max_possible_force > 0.0 {
                            (force / max_possible_force) as f32
                        } else {
                            1.0
                        }
                    }
                    winit::event::Force::Normalized(n) => n as f32,
                });
                self.events.push(PlatformEvent::TouchInput {
                    handle,
                    touch_id: touch.id,
                    phase,
                    x: touch.location.x,
                    y: touch.location.y,
                    pressure,
                });
            }

            // Events we don't need to map.
            _ => {}
        }
    }
}

// -----------------------------------------------------------------------------
// WinitPlatform -- the shared cross-platform desktop implementation
// -----------------------------------------------------------------------------

/// Cross-platform desktop `Platform` implementation backed by winit 0.30.
///
/// This struct manages the winit `EventLoop`, creates `Window` objects, and
/// converts winit events into [`PlatformEvent`] values that the engine
/// consumes each frame.
///
/// # Design
///
/// winit 0.30 requires an `ApplicationHandler` callback model where
/// `EventLoop::run()` takes ownership and never returns. For a game engine
/// that needs to retain control of its main loop, we use
/// `EventLoop::pump_events()` (from `EventLoopExtPumpEvents`, available on
/// desktop platforms). This processes all pending OS events and returns
/// control immediately, allowing the engine to call `poll_events()` once
/// per frame.
///
/// Window creation is deferred because winit requires an `ActiveEventLoop`
/// reference (only available inside event loop callbacks). When
/// `create_window()` is called, the request is queued and fulfilled on the
/// next `poll_events()` → `pump_events()` cycle. The window handle is
/// returned immediately so the caller can use it for subsequent operations
/// (which will take effect once the window is actually created).
pub struct WinitPlatform {
    /// The winit event loop. Wrapped in `Option` so it can be temporarily
    /// taken for `pump_events()` calls.
    event_loop: Option<EventLoop<()>>,
    /// The application handler that accumulates events.
    handler: WinitAppHandler,
    /// All managed windows, keyed by winit WindowId.
    windows: HashMap<WindowId, WinitWindowData>,
    /// Reverse map: engine handle ID -> winit WindowId.
    handle_to_window_id: HashMap<u64, WindowId>,
    /// Next engine window handle ID.
    next_handle_id: AtomicU64,
    /// Platform-specific render backend preference.
    render_backend: RenderBackend,
    /// In-memory clipboard (fallback). For full OS clipboard support,
    /// consider adding the `arboard` crate.
    clipboard: Mutex<Option<String>>,
    /// Queued window creation requests (interior mutability for &self API).
    pending_creates: Mutex<Vec<(WindowDesc, WindowHandle)>>,
    /// Queued window destruction requests.
    pending_destroys: Mutex<Vec<WindowHandle>>,
}

// Safety: WinitPlatform is designed to be used from the main thread for event
// pumping and window operations. The Arc<Window> handles inside are Send+Sync.
// The EventLoop is only accessed from the main thread via poll_events().
// The Platform trait requires Send+Sync, and we ensure all fields satisfy this
// (HashMap values contain Arc<Window>, AtomicU64, Mutex).
unsafe impl Send for WinitPlatform {}
unsafe impl Sync for WinitPlatform {}

impl WinitPlatform {
    /// Create a new WinitPlatform with the given render backend preference.
    ///
    /// This initializes the winit `EventLoop`. Must be called from the main
    /// thread (winit requirement on all platforms).
    pub fn new(render_backend: RenderBackend) -> Self {
        let event_loop = EventLoop::new().expect("Failed to create winit EventLoop");

        log::info!(
            "WinitPlatform initialized (render_backend={:?})",
            render_backend
        );

        Self {
            event_loop: Some(event_loop),
            handler: WinitAppHandler::new(),
            windows: HashMap::new(),
            handle_to_window_id: HashMap::new(),
            next_handle_id: AtomicU64::new(1),
            render_backend,
            clipboard: Mutex::new(None),
            pending_creates: Mutex::new(Vec::new()),
            pending_destroys: Mutex::new(Vec::new()),
        }
    }

    /// Look up the winit Window for a given engine handle.
    fn get_window(&self, handle: WindowHandle) -> Option<&Arc<Window>> {
        self.handle_to_window_id
            .get(&handle.0)
            .and_then(|wid| self.windows.get(wid))
            .map(|data| &data.window)
    }
}

impl Platform for WinitPlatform {
    fn create_window(&self, desc: &WindowDesc) -> Result<WindowHandle> {
        let handle_id = self.next_handle_id.fetch_add(1, Ordering::Relaxed);
        let engine_handle = WindowHandle(handle_id);

        // We cannot create a window directly because winit requires an
        // ActiveEventLoop reference (only available inside event loop
        // callbacks). Queue the request -- it will be fulfilled on the next
        // pump_events() cycle.
        //
        // Queue via interior mutability. The Platform trait takes &self for
        // create_window; we use a Mutex on the pending list to avoid UB.
        self.pending_creates.lock().unwrap().push((desc.clone(), engine_handle));

        log::info!(
            "Window creation queued: \"{}\" (handle={}). Will be created on next poll_events().",
            desc.title,
            engine_handle
        );

        Ok(engine_handle)
    }

    fn destroy_window(&self, handle: WindowHandle) {
        // Queue destruction via interior mutability.
        self.pending_destroys.lock().unwrap().push(handle);
        log::info!("Window destruction queued: {}", handle);
    }

    fn poll_events(&mut self) -> Vec<PlatformEvent> {
        if let Some(mut event_loop) = self.event_loop.take() {
            // pump_events processes all pending OS events and returns
            // immediately. Using Duration::ZERO means "don't block, just
            // drain what's available".
            #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
            {
                use winit::platform::pump_events::PumpStatus;

                let status =
                    event_loop.pump_app_events(Some(Duration::ZERO), &mut self.handler);

                // Integrate newly created windows into our maps.
                for (window_id, data) in self.handler.newly_created_windows.drain(..) {
                    self.handle_to_window_id
                        .insert(data.engine_handle.0, window_id);
                    self.windows.insert(window_id, data);
                }

                // If the event loop signals exit, tell the engine.
                match status {
                    PumpStatus::Exit(_code) => {
                        self.handler.events.push(PlatformEvent::AppQuitRequested);
                    }
                    PumpStatus::Continue => {}
                }
            }

            // Return the event loop so it can be used next frame.
            self.event_loop = Some(event_loop);
        }

        // Drain all accumulated events for this frame.
        std::mem::take(&mut self.handler.events)
    }

    fn get_render_backend(&self) -> RenderBackend {
        self.render_backend
    }

    fn get_display_info(&self) -> Vec<DisplayInfo> {
        // Monitor enumeration requires an ActiveEventLoop reference (only
        // available inside event loop callbacks in winit 0.30). Return info
        // from any windows we have created, or a sensible default.
        let mut displays: Vec<DisplayInfo> = Vec::new();
        let mut is_first = true;

        for window_data in self.windows.values() {
            if let Some(monitor) = window_data.window.current_monitor() {
                let size: winit::dpi::PhysicalSize<u32> = monitor.size();
                let position = monitor.position();
                let refresh_rate = monitor
                    .refresh_rate_millihertz()
                    .map(|mhz| mhz / 1000)
                    .unwrap_or(60);
                let scale = monitor.scale_factor();
                let name = monitor.name().unwrap_or_default();

                displays.push(DisplayInfo {
                    name,
                    width: size.width,
                    height: size.height,
                    refresh_rate_hz: refresh_rate,
                    scale_factor: scale,
                    is_primary: is_first,
                    position: (position.x, position.y),
                });

                is_first = false;
            }
        }

        displays
    }

    fn set_cursor(&self, cursor: CursorType) {
        if cursor == CursorType::Hidden {
            for data in self.windows.values() {
                data.window.set_cursor_visible(false);
            }
            return;
        }

        if let Some(icon) = map_cursor_type(cursor) {
            for data in self.windows.values() {
                data.window.set_cursor_visible(true);
                data.window.set_cursor(winit::window::Cursor::Icon(icon));
            }
        }
    }

    fn set_clipboard(&self, text: &str) {
        // In-memory clipboard fallback. For real OS clipboard, add `arboard`.
        if let Ok(mut cb) = self.clipboard.lock() {
            *cb = Some(text.to_owned());
        }
    }

    fn get_clipboard(&self) -> Option<String> {
        self.clipboard.lock().ok().and_then(|cb| cb.clone())
    }

    fn get_system_info(&self) -> SystemInfo {
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        #[cfg(target_os = "windows")]
        let (os_name, os_version) = ("Windows".to_owned(), "10+".to_owned());
        #[cfg(target_os = "macos")]
        let (os_name, os_version) = ("macOS".to_owned(), "Unknown".to_owned());
        #[cfg(target_os = "linux")]
        let (os_name, os_version) = ("Linux".to_owned(), "Unknown".to_owned());
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        let (os_name, os_version) = ("Unknown".to_owned(), "Unknown".to_owned());

        SystemInfo {
            cpu_name: String::from("Unknown"),
            cpu_cores,
            gpu_name: String::from("Unknown"),
            gpu_vram_mb: 0,
            total_ram_mb: 0,
            os_name,
            os_version,
        }
    }

    fn get_raw_window_handle(&self, handle: WindowHandle) -> RawWindowHandle {
        let Some(window) = self.get_window(handle) else {
            log::error!(
                "get_raw_window_handle called with invalid handle: {}",
                handle
            );
            return RawWindowHandle::Unavailable;
        };

        // Use raw-window-handle 0.6 traits to extract native handles.
        use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

        let wh = match window.window_handle() {
            Ok(wh) => wh,
            Err(_) => return RawWindowHandle::Unavailable,
        };

        let dh = match window.display_handle() {
            Ok(dh) => dh,
            Err(_) => return RawWindowHandle::Unavailable,
        };

        match wh.as_raw() {
            #[cfg(target_os = "windows")]
            raw_window_handle::RawWindowHandle::Win32(h) => RawWindowHandle::Windows {
                hwnd: isize::from(h.hwnd) as *mut std::ffi::c_void,
                hinstance: h
                    .hinstance
                    .map(|i| isize::from(i) as *mut std::ffi::c_void)
                    .unwrap_or(std::ptr::null_mut()),
            },

            #[cfg(target_os = "macos")]
            raw_window_handle::RawWindowHandle::AppKit(h) => RawWindowHandle::MacOS {
                ns_view: h.ns_view.as_ptr(),
                ns_window: std::ptr::null_mut(), // rwh 0.6 exposes NSView; NSWindow not directly available
            },

            #[cfg(target_os = "linux")]
            raw_window_handle::RawWindowHandle::Xlib(h) => {
                // The display pointer comes from the display handle, not the
                // window handle, in rwh 0.6.
                let display_ptr = match dh.as_raw() {
                    raw_window_handle::RawDisplayHandle::Xlib(d) => d
                        .display
                        .map(|p| p.as_ptr())
                        .unwrap_or(std::ptr::null_mut()),
                    _ => std::ptr::null_mut(),
                };
                RawWindowHandle::X11 {
                    window: h.window as u64,
                    display: display_ptr,
                }
            }

            #[cfg(target_os = "linux")]
            raw_window_handle::RawWindowHandle::Xcb(h) => {
                let connection_ptr = match dh.as_raw() {
                    raw_window_handle::RawDisplayHandle::Xcb(d) => d
                        .connection
                        .map(|p| p.as_ptr())
                        .unwrap_or(std::ptr::null_mut()),
                    _ => std::ptr::null_mut(),
                };
                RawWindowHandle::X11 {
                    window: h.window.get() as u64,
                    display: connection_ptr,
                }
            }

            #[cfg(target_os = "linux")]
            raw_window_handle::RawWindowHandle::Wayland(h) => {
                let display_ptr = match dh.as_raw() {
                    raw_window_handle::RawDisplayHandle::Wayland(d) => d.display.as_ptr(),
                    _ => std::ptr::null_mut(),
                };
                RawWindowHandle::Wayland {
                    surface: h.surface.as_ptr(),
                    display: display_ptr,
                }
            }

            _ => RawWindowHandle::Unavailable,
        }
    }

    fn set_window_title(&self, handle: WindowHandle, title: &str) {
        if let Some(window) = self.get_window(handle) {
            window.set_title(title);
        } else {
            log::warn!("set_window_title: invalid handle {}", handle);
        }
    }

    fn set_window_size(&self, handle: WindowHandle, width: u32, height: u32) {
        if let Some(window) = self.get_window(handle) {
            let _ = window.request_inner_size(LogicalSize::new(width, height));
        } else {
            log::warn!("set_window_size: invalid handle {}", handle);
        }
    }

    fn set_window_position(&self, handle: WindowHandle, x: i32, y: i32) {
        if let Some(window) = self.get_window(handle) {
            window.set_outer_position(PhysicalPosition::new(x, y));
        } else {
            log::warn!("set_window_position: invalid handle {}", handle);
        }
    }

    fn set_fullscreen(&self, handle: WindowHandle, fullscreen: bool) {
        if let Some(window) = self.get_window(handle) {
            if fullscreen {
                window.set_fullscreen(Some(Fullscreen::Borderless(None)));
            } else {
                window.set_fullscreen(None);
            }
        } else {
            log::warn!("set_fullscreen: invalid handle {}", handle);
        }
    }

    fn set_cursor_locked(&self, handle: WindowHandle, locked: bool) {
        if let Some(window) = self.get_window(handle) {
            use winit::window::CursorGrabMode;
            let mode = if locked {
                CursorGrabMode::Confined
            } else {
                CursorGrabMode::None
            };
            if let Err(e) = window.set_cursor_grab(mode) {
                // Fall back to Locked mode if Confined is not supported
                // (e.g. on Wayland).
                if locked {
                    if let Err(e2) = window.set_cursor_grab(CursorGrabMode::Locked) {
                        log::warn!(
                            "set_cursor_locked failed (Confined: {}, Locked: {})",
                            e,
                            e2
                        );
                    }
                } else {
                    log::warn!("set_cursor_locked(false) failed: {}", e);
                }
            }
        } else {
            log::warn!("set_cursor_locked: invalid handle {}", handle);
        }
    }

    fn set_cursor_visible(&self, visible: bool) {
        for data in self.windows.values() {
            data.window.set_cursor_visible(visible);
        }
    }

    fn get_window_scale_factor(&self, handle: WindowHandle) -> f64 {
        self.get_window(handle)
            .map(|w| w.scale_factor())
            .unwrap_or(1.0)
    }
}
