# genovo-platform

Platform abstraction layer providing window management, input handling, and OS-specific functionality.

## Window Management

Built on `winit` for cross-platform window creation and event processing.

```rust
let window = Window::new(WindowDesc {
    title: "My Game",
    width: 1920,
    height: 1080,
    resizable: true,
    fullscreen: FullscreenMode::Windowed,
    vsync: true,
}).expect("Failed to create window");
```

### Supported Window Features

- Windowed, borderless fullscreen, and exclusive fullscreen modes
- Runtime resolution changes
- Multi-monitor support
- High-DPI / Retina scaling
- Raw window handle (`raw-window-handle`) for GPU backend initialization
- Cursor control (show/hide, lock, custom cursor)

## Input System

Unified input handling across keyboard, mouse, gamepad, and touch:

### Input Abstraction Layers

1. **Raw Input** - Direct key/button press/release events with scan codes
2. **Mapped Input** - Named actions and axes mapped from raw input via configuration
3. **Processed Input** - Smoothed, dead-zoned, and combined input values

```rust
// Raw input query
if input.key_pressed(KeyCode::Space) {
    player.jump();
}

// Mapped input query
if input.action_pressed("jump") {
    player.jump();
}

// Axis query (combined from multiple bindings)
let move_x = input.axis("move_horizontal");  // -1.0 to 1.0
let move_y = input.axis("move_vertical");    // -1.0 to 1.0
```

### Supported Input Devices

| Device | Status | Notes |
|--------|--------|-------|
| Keyboard | Planned (Month 2) | Key press/release, text input |
| Mouse | Planned (Month 2) | Position, delta, buttons, scroll |
| Gamepad | Planned (Month 3) | XInput/DirectInput on Windows, HID on other platforms |
| Touch | Planned (Month 8) | Multi-touch with gesture recognition (mobile) |

## File I/O

Platform-specific file system access:

- Asset directory resolution (platform-specific asset paths)
- Async file reading via IO thread
- File watch for hot-reloading (notify-based)
- Platform save directories (AppData, Library, etc.)

## Platform Targets

| Platform | Window System | GPU API | Status |
|----------|--------------|---------|--------|
| Windows 10/11 | Win32 | Vulkan, DX12 | Primary (Month 1+) |
| macOS 13+ | Cocoa | Metal (via wgpu) | Month 4 |
| Linux (X11/Wayland) | X11, Wayland | Vulkan | Month 3 |
| iOS 16+ | UIKit | Metal (via wgpu) | Month 8 |
| Android 10+ | NativeActivity | Vulkan | Month 9 |
| Web | Canvas | WebGPU | Experimental |

## Status

- Window creation and event loop (Month 1-2)
- Keyboard and mouse input (Month 2)
- Raw window handle for render backend (Month 2)
- Gamepad support (Month 3)
- File system abstraction (Month 2)
- High-DPI support (Month 3)
- Mobile platform support (Month 8-9)
