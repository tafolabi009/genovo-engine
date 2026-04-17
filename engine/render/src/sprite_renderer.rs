// engine/render/src/sprite_renderer.rs
//
// 2D sprite rendering system for the Genovo engine.
//
// Implements a batched sprite renderer with:
//
// - **Sprite batching** — Collects sprites into batched draw calls grouped by
//   texture atlas to minimise GPU state changes.
// - **Texture atlas UVs** — Sprites reference regions within a texture atlas.
// - **Sprite animation** — Frame sequences with configurable frame rate, looping,
//   and ping-pong modes.
// - **Sprite sorting** — Sort by layer, sub-layer, and Y-position for correct
//   draw ordering.
// - **Nine-slice sprites** — Scalable UI sprites with preserved corner/edge sizes.
// - **Pixel-perfect rendering** — Snap sprite positions to pixel boundaries.
// - **Sprite flip** — Horizontal and vertical flip via UV manipulation.
// - **Tinting** — Per-sprite colour tint (multiply blend).
// - **Sprite sheets** — Named animations from a sprite sheet definition.
//
// # Pipeline integration
//
// Sprites are rendered in a dedicated 2D pass after the 3D scene, using an
// orthographic projection. The batch renderer generates a single vertex buffer
// per atlas texture and issues one draw call per batch.

// ---------------------------------------------------------------------------
// Sprite rect (UV region in atlas)
// ---------------------------------------------------------------------------

/// A rectangular region within a texture atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpriteRect {
    /// Top-left U coordinate [0, 1].
    pub u: f32,
    /// Top-left V coordinate [0, 1].
    pub v: f32,
    /// Width in UV space.
    pub w: f32,
    /// Height in UV space.
    pub h: f32,
}

impl SpriteRect {
    pub const FULL: Self = Self { u: 0.0, v: 0.0, w: 1.0, h: 1.0 };

    /// Create a new sprite rect.
    pub fn new(u: f32, v: f32, w: f32, h: f32) -> Self {
        Self { u, v, w, h }
    }

    /// Create a rect from pixel coordinates and atlas size.
    pub fn from_pixels(x: u32, y: u32, w: u32, h: u32, atlas_w: u32, atlas_h: u32) -> Self {
        Self {
            u: x as f32 / atlas_w as f32,
            v: y as f32 / atlas_h as f32,
            w: w as f32 / atlas_w as f32,
            h: h as f32 / atlas_h as f32,
        }
    }

    /// Get the four UV corners: [top-left, top-right, bottom-right, bottom-left].
    pub fn corners(&self) -> [[f32; 2]; 4] {
        [
            [self.u, self.v],
            [self.u + self.w, self.v],
            [self.u + self.w, self.v + self.h],
            [self.u, self.v + self.h],
        ]
    }

    /// Get flipped corners.
    pub fn corners_flipped(&self, flip_h: bool, flip_v: bool) -> [[f32; 2]; 4] {
        let mut c = self.corners();
        if flip_h {
            c.swap(0, 1);
            c.swap(2, 3);
        }
        if flip_v {
            c.swap(0, 3);
            c.swap(1, 2);
        }
        c
    }
}

// ---------------------------------------------------------------------------
// Sprite animation
// ---------------------------------------------------------------------------

/// Animation playback mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationMode {
    /// Play once and stop on the last frame.
    Once,
    /// Loop from the beginning.
    Loop,
    /// Play forward then backward, repeating.
    PingPong,
}

/// A single sprite animation (sequence of frames).
#[derive(Debug, Clone)]
pub struct SpriteAnimation {
    /// Animation name.
    pub name: String,
    /// Frame rects in the atlas.
    pub frames: Vec<SpriteRect>,
    /// Frames per second.
    pub fps: f32,
    /// Playback mode.
    pub mode: AnimationMode,
    /// Per-frame anchor offsets (optional, same length as `frames` if provided).
    pub anchors: Option<Vec<[f32; 2]>>,
    /// Per-frame hitbox (optional, same length as `frames`).
    pub hitboxes: Option<Vec<[f32; 4]>>,
}

impl SpriteAnimation {
    /// Create a new animation.
    pub fn new(name: impl Into<String>, frames: Vec<SpriteRect>, fps: f32) -> Self {
        Self {
            name: name.into(),
            frames,
            fps,
            mode: AnimationMode::Loop,
            anchors: None,
            hitboxes: None,
        }
    }

    /// Set the playback mode.
    pub fn with_mode(mut self, mode: AnimationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Get the number of frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Compute the current frame index given elapsed time.
    pub fn frame_at_time(&self, time: f32) -> usize {
        if self.frames.is_empty() {
            return 0;
        }

        let total_frames = self.frames.len() as f32;
        let frame_f = time * self.fps;

        match self.mode {
            AnimationMode::Once => {
                (frame_f as usize).min(self.frames.len() - 1)
            }
            AnimationMode::Loop => {
                let idx = frame_f % total_frames;
                idx as usize % self.frames.len()
            }
            AnimationMode::PingPong => {
                let cycle_len = (total_frames - 1.0) * 2.0;
                if cycle_len <= 0.0 {
                    return 0;
                }
                let pos = frame_f % cycle_len;
                if pos < total_frames {
                    pos as usize
                } else {
                    (cycle_len - pos) as usize
                }
            }
        }
    }

    /// Get the sprite rect for the current frame.
    pub fn rect_at_time(&self, time: f32) -> SpriteRect {
        let idx = self.frame_at_time(time);
        self.frames[idx]
    }

    /// Get the total animation duration in seconds.
    pub fn duration(&self) -> f32 {
        if self.fps <= 0.0 {
            return 0.0;
        }
        self.frames.len() as f32 / self.fps
    }
}

// ---------------------------------------------------------------------------
// Sprite sheet
// ---------------------------------------------------------------------------

/// A sprite sheet containing multiple named animations.
#[derive(Debug, Clone)]
pub struct SpriteSheet {
    /// Atlas texture handle.
    pub atlas_handle: u64,
    /// Atlas dimensions in pixels.
    pub atlas_size: (u32, u32),
    /// Named animations.
    pub animations: Vec<SpriteAnimation>,
    /// Default animation name.
    pub default_animation: String,
}

impl SpriteSheet {
    /// Create a new sprite sheet.
    pub fn new(atlas_handle: u64, atlas_size: (u32, u32)) -> Self {
        Self {
            atlas_handle,
            atlas_size,
            animations: Vec::new(),
            default_animation: String::new(),
        }
    }

    /// Add an animation to the sheet.
    pub fn add_animation(&mut self, anim: SpriteAnimation) {
        if self.animations.is_empty() {
            self.default_animation = anim.name.clone();
        }
        self.animations.push(anim);
    }

    /// Find an animation by name.
    pub fn find_animation(&self, name: &str) -> Option<&SpriteAnimation> {
        self.animations.iter().find(|a| a.name == name)
    }

    /// Create a grid-based sprite sheet with uniform cell sizes.
    ///
    /// # Arguments
    /// * `cols` — Number of columns in the grid.
    /// * `rows` — Number of rows in the grid.
    /// * `anim_name` — Name for the generated animation.
    /// * `fps` — Frame rate.
    /// * `frame_count` — Total number of frames (may be less than cols*rows).
    pub fn from_grid(
        atlas_handle: u64,
        atlas_size: (u32, u32),
        cols: u32,
        rows: u32,
        anim_name: &str,
        fps: f32,
        frame_count: u32,
    ) -> Self {
        let cell_w = 1.0 / cols as f32;
        let cell_h = 1.0 / rows as f32;

        let mut frames = Vec::new();
        let mut count = 0;
        'outer: for row in 0..rows {
            for col in 0..cols {
                if count >= frame_count {
                    break 'outer;
                }
                frames.push(SpriteRect {
                    u: col as f32 * cell_w,
                    v: row as f32 * cell_h,
                    w: cell_w,
                    h: cell_h,
                });
                count += 1;
            }
        }

        let anim = SpriteAnimation::new(anim_name, frames, fps);
        let mut sheet = Self::new(atlas_handle, atlas_size);
        sheet.add_animation(anim);
        sheet
    }
}

// ---------------------------------------------------------------------------
// Sprite instance
// ---------------------------------------------------------------------------

/// A single sprite to be rendered.
#[derive(Debug, Clone)]
pub struct Sprite {
    /// Position in world/screen space (x, y).
    pub position: [f32; 2],
    /// Size in world/screen units (width, height).
    pub size: [f32; 2],
    /// Rotation in radians.
    pub rotation: f32,
    /// Anchor/pivot point (0,0 = top-left, 0.5,0.5 = centre, 1,1 = bottom-right).
    pub anchor: [f32; 2],
    /// UV rect within the atlas.
    pub rect: SpriteRect,
    /// Atlas texture handle.
    pub atlas: u64,
    /// Colour tint (linear RGBA, multiplied with texture colour).
    pub tint: [f32; 4],
    /// Horizontal flip.
    pub flip_h: bool,
    /// Vertical flip.
    pub flip_v: bool,
    /// Sort layer (higher = drawn later / on top).
    pub layer: i32,
    /// Sort sub-layer (secondary sort within the same layer).
    pub sub_layer: i32,
    /// Enable Y-sorting within the same layer/sub-layer.
    pub y_sort: bool,
    /// Depth for 2.5D ordering (optional).
    pub depth: f32,
    /// Visibility.
    pub visible: bool,
    /// Opacity (multiplied with tint alpha).
    pub opacity: f32,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            size: [64.0, 64.0],
            rotation: 0.0,
            anchor: [0.5, 0.5],
            rect: SpriteRect::FULL,
            atlas: 0,
            tint: [1.0, 1.0, 1.0, 1.0],
            flip_h: false,
            flip_v: false,
            layer: 0,
            sub_layer: 0,
            y_sort: false,
            depth: 0.0,
            visible: true,
            opacity: 1.0,
        }
    }
}

impl Sprite {
    /// Create a new sprite at the given position with a size.
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            position: [x, y],
            size: [w, h],
            ..Self::default()
        }
    }

    /// Set the texture atlas and rect.
    pub fn with_atlas(mut self, atlas: u64, rect: SpriteRect) -> Self {
        self.atlas = atlas;
        self.rect = rect;
        self
    }

    /// Set the tint colour.
    pub fn with_tint(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.tint = [r, g, b, a];
        self
    }

    /// Set the sort layer.
    pub fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }

    /// Enable Y-sorting.
    pub fn with_y_sort(mut self) -> Self {
        self.y_sort = true;
        self
    }

    /// Set horizontal and vertical flip.
    pub fn with_flip(mut self, h: bool, v: bool) -> Self {
        self.flip_h = h;
        self.flip_v = v;
        self
    }

    /// Compute the sort key for ordering.
    ///
    /// Sort order: layer → sub_layer → y_position (if y_sort) → depth.
    pub fn sort_key(&self) -> (i32, i32, i32) {
        let y_key = if self.y_sort {
            (self.position[1] * 1000.0) as i32
        } else {
            0
        };
        (self.layer, self.sub_layer, y_key)
    }
}

// ---------------------------------------------------------------------------
// Animated sprite
// ---------------------------------------------------------------------------

/// A sprite with animation state.
#[derive(Debug, Clone)]
pub struct AnimatedSprite {
    /// The base sprite.
    pub sprite: Sprite,
    /// Sprite sheet reference.
    pub sheet: SpriteSheet,
    /// Current animation name.
    pub current_animation: String,
    /// Elapsed time for the current animation.
    pub elapsed: f32,
    /// Playback speed multiplier (1.0 = normal).
    pub speed: f32,
    /// Whether the animation is playing.
    pub playing: bool,
    /// Whether the animation has finished (only for Once mode).
    pub finished: bool,
}

impl AnimatedSprite {
    /// Create a new animated sprite.
    pub fn new(sprite: Sprite, sheet: SpriteSheet) -> Self {
        let anim_name = sheet.default_animation.clone();
        Self {
            sprite,
            sheet,
            current_animation: anim_name,
            elapsed: 0.0,
            speed: 1.0,
            playing: true,
            finished: false,
        }
    }

    /// Play a named animation from the beginning.
    pub fn play(&mut self, name: &str) {
        if self.current_animation != name {
            self.current_animation = name.to_string();
            self.elapsed = 0.0;
            self.finished = false;
        }
        self.playing = true;
    }

    /// Stop playback.
    pub fn stop(&mut self) {
        self.playing = false;
    }

    /// Reset the animation to the first frame.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.finished = false;
    }

    /// Update the animation state.
    ///
    /// # Arguments
    /// * `dt` — Delta time in seconds.
    pub fn update(&mut self, dt: f32) {
        if !self.playing || self.finished {
            return;
        }

        self.elapsed += dt * self.speed;

        if let Some(anim) = self.sheet.find_animation(&self.current_animation) {
            let rect = anim.rect_at_time(self.elapsed);
            self.sprite.rect = rect;

            // Check if finished (Once mode).
            if anim.mode == AnimationMode::Once {
                let duration = anim.duration();
                if self.elapsed >= duration {
                    self.finished = true;
                }
            }
        }
    }

    /// Get the current frame index.
    pub fn current_frame(&self) -> usize {
        if let Some(anim) = self.sheet.find_animation(&self.current_animation) {
            anim.frame_at_time(self.elapsed)
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Nine-slice sprite
// ---------------------------------------------------------------------------

/// Nine-slice sprite definition for scalable UI elements.
///
/// The sprite is divided into 9 regions:
/// ```text
/// +---+-------+---+
/// | 1 |   2   | 3 |
/// +---+-------+---+
/// | 4 |   5   | 6 |
/// +---+-------+---+
/// | 7 |   8   | 9 |
/// +---+-------+---+
/// ```
///
/// Corners (1, 3, 7, 9) are drawn at fixed size.
/// Edges (2, 4, 6, 8) are stretched along one axis.
/// Centre (5) is stretched along both axes.
#[derive(Debug, Clone)]
pub struct NineSlice {
    /// The full sprite rect in the atlas.
    pub rect: SpriteRect,
    /// Left border width (pixels in the source texture).
    pub left: f32,
    /// Right border width (pixels in the source texture).
    pub right: f32,
    /// Top border height (pixels in the source texture).
    pub top: f32,
    /// Bottom border height (pixels in the source texture).
    pub bottom: f32,
    /// Atlas texture handle.
    pub atlas: u64,
    /// Tint colour.
    pub tint: [f32; 4],
}

impl NineSlice {
    /// Create a new nine-slice definition.
    pub fn new(rect: SpriteRect, left: f32, right: f32, top: f32, bottom: f32, atlas: u64) -> Self {
        Self {
            rect,
            left,
            right,
            top,
            bottom,
            atlas,
            tint: [1.0, 1.0, 1.0, 1.0],
        }
    }

    /// Generate the 9 quads for rendering at a given position and size.
    ///
    /// # Arguments
    /// * `x`, `y` — Top-left position.
    /// * `w`, `h` — Total size.
    /// * `atlas_w`, `atlas_h` — Atlas dimensions in pixels.
    ///
    /// # Returns
    /// Array of 9 (position, size, uv_rect) tuples.
    pub fn generate_quads(
        &self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        atlas_w: f32,
        atlas_h: f32,
    ) -> Vec<([f32; 2], [f32; 2], SpriteRect)> {
        let mut quads = Vec::with_capacity(9);

        // Convert border sizes from pixels to UV.
        let lu = self.left / atlas_w;
        let ru = self.right / atlas_w;
        let tv = self.top / atlas_h;
        let bv = self.bottom / atlas_h;

        // Positions.
        let x0 = x;
        let x1 = x + self.left;
        let x2 = x + w - self.right;
        let x3 = x + w;

        let y0 = y;
        let y1 = y + self.top;
        let y2 = y + h - self.bottom;
        let y3 = y + h;

        // UV.
        let u0 = self.rect.u;
        let u1 = self.rect.u + lu;
        let u2 = self.rect.u + self.rect.w - ru;
        let u3 = self.rect.u + self.rect.w;

        let v0 = self.rect.v;
        let v1 = self.rect.v + tv;
        let v2 = self.rect.v + self.rect.h - bv;
        let v3 = self.rect.v + self.rect.h;

        // 1: Top-left corner.
        quads.push(([x0, y0], [self.left, self.top], SpriteRect::new(u0, v0, lu, tv)));
        // 2: Top edge.
        quads.push(([x1, y0], [x2 - x1, self.top], SpriteRect::new(u1, v0, u2 - u1, tv)));
        // 3: Top-right corner.
        quads.push(([x2, y0], [self.right, self.top], SpriteRect::new(u2, v0, ru, tv)));
        // 4: Left edge.
        quads.push(([x0, y1], [self.left, y2 - y1], SpriteRect::new(u0, v1, lu, v2 - v1)));
        // 5: Centre.
        quads.push(([x1, y1], [x2 - x1, y2 - y1], SpriteRect::new(u1, v1, u2 - u1, v2 - v1)));
        // 6: Right edge.
        quads.push(([x2, y1], [self.right, y2 - y1], SpriteRect::new(u2, v1, ru, v2 - v1)));
        // 7: Bottom-left corner.
        quads.push(([x0, y2], [self.left, self.bottom], SpriteRect::new(u0, v2, lu, bv)));
        // 8: Bottom edge.
        quads.push(([x1, y2], [x2 - x1, self.bottom], SpriteRect::new(u1, v2, u2 - u1, bv)));
        // 9: Bottom-right corner.
        quads.push(([x2, y2], [self.right, self.bottom], SpriteRect::new(u2, v2, ru, bv)));

        quads
    }
}

// ---------------------------------------------------------------------------
// Sprite vertex
// ---------------------------------------------------------------------------

/// Vertex layout for sprite rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpriteVertex {
    /// Position (x, y).
    pub position: [f32; 2],
    /// Texture coordinates (u, v).
    pub uv: [f32; 2],
    /// Colour tint (r, g, b, a).
    pub color: [f32; 4],
}

impl SpriteVertex {
    pub fn new(x: f32, y: f32, u: f32, v: f32, color: [f32; 4]) -> Self {
        Self {
            position: [x, y],
            uv: [u, v],
            color,
        }
    }
}

// ---------------------------------------------------------------------------
// Sprite batch
// ---------------------------------------------------------------------------

/// A batch of sprites sharing the same atlas texture.
#[derive(Debug, Clone)]
pub struct SpriteBatch {
    /// Atlas texture handle.
    pub atlas: u64,
    /// Vertices for all sprites in this batch.
    pub vertices: Vec<SpriteVertex>,
    /// Indices for all sprites in this batch.
    pub indices: Vec<u32>,
    /// Number of sprites in this batch.
    pub sprite_count: u32,
}

impl SpriteBatch {
    /// Create a new empty batch.
    pub fn new(atlas: u64) -> Self {
        Self {
            atlas,
            vertices: Vec::new(),
            indices: Vec::new(),
            sprite_count: 0,
        }
    }

    /// Add a sprite quad to the batch.
    pub fn add_sprite(&mut self, sprite: &Sprite) {
        let uvs = sprite.rect.corners_flipped(sprite.flip_h, sprite.flip_v);
        let tint = [
            sprite.tint[0],
            sprite.tint[1],
            sprite.tint[2],
            sprite.tint[3] * sprite.opacity,
        ];

        // Compute corners relative to anchor.
        let ax = sprite.anchor[0] * sprite.size[0];
        let ay = sprite.anchor[1] * sprite.size[1];

        let mut corners = [
            [-ax, -ay],
            [sprite.size[0] - ax, -ay],
            [sprite.size[0] - ax, sprite.size[1] - ay],
            [-ax, sprite.size[1] - ay],
        ];

        // Apply rotation.
        if sprite.rotation != 0.0 {
            let (sin_r, cos_r) = sprite.rotation.sin_cos();
            for corner in &mut corners {
                let rx = corner[0] * cos_r - corner[1] * sin_r;
                let ry = corner[0] * sin_r + corner[1] * cos_r;
                corner[0] = rx;
                corner[1] = ry;
            }
        }

        // Offset by position.
        for corner in &mut corners {
            corner[0] += sprite.position[0];
            corner[1] += sprite.position[1];
        }

        let base = self.vertices.len() as u32;

        for i in 0..4 {
            self.vertices.push(SpriteVertex::new(
                corners[i][0],
                corners[i][1],
                uvs[i][0],
                uvs[i][1],
                tint,
            ));
        }

        // Two triangles.
        self.indices.push(base);
        self.indices.push(base + 1);
        self.indices.push(base + 2);
        self.indices.push(base);
        self.indices.push(base + 2);
        self.indices.push(base + 3);

        self.sprite_count += 1;
    }

    /// Clear the batch for reuse.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.sprite_count = 0;
    }

    /// Get the byte size of the vertex data.
    pub fn vertex_buffer_size(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<SpriteVertex>()
    }

    /// Get the byte size of the index data.
    pub fn index_buffer_size(&self) -> usize {
        self.indices.len() * std::mem::size_of::<u32>()
    }
}

// ---------------------------------------------------------------------------
// Sprite renderer
// ---------------------------------------------------------------------------

/// Pixel-perfect alignment mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelPerfectMode {
    /// No pixel snapping.
    None,
    /// Snap sprite positions to the nearest pixel.
    SnapPosition,
    /// Snap both position and size to pixel boundaries.
    SnapPositionAndSize,
}

/// The sprite renderer manages batching and sorting of sprites.
#[derive(Debug)]
pub struct SpriteRenderer {
    /// All sprites pending rendering this frame.
    sprites: Vec<Sprite>,
    /// Compiled batches (one per atlas texture).
    batches: Vec<SpriteBatch>,
    /// Screen size in pixels (for pixel-perfect rendering).
    pub screen_size: (u32, u32),
    /// Pixel-perfect rendering mode.
    pub pixel_perfect: PixelPerfectMode,
    /// Pixels per unit (for converting world units to pixels).
    pub pixels_per_unit: f32,
    /// Global tint applied to all sprites.
    pub global_tint: [f32; 4],
    /// Camera offset (for scrolling).
    pub camera_offset: [f32; 2],
    /// Camera zoom.
    pub camera_zoom: f32,
}

impl SpriteRenderer {
    /// Create a new sprite renderer.
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            sprites: Vec::new(),
            batches: Vec::new(),
            screen_size: (screen_width, screen_height),
            pixel_perfect: PixelPerfectMode::None,
            pixels_per_unit: 1.0,
            global_tint: [1.0, 1.0, 1.0, 1.0],
            camera_offset: [0.0, 0.0],
            camera_zoom: 1.0,
        }
    }

    /// Submit a sprite for rendering this frame.
    pub fn submit(&mut self, sprite: Sprite) {
        if sprite.visible {
            self.sprites.push(sprite);
        }
    }

    /// Submit a nine-slice sprite.
    pub fn submit_nine_slice(
        &mut self,
        nine_slice: &NineSlice,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        layer: i32,
        atlas_w: f32,
        atlas_h: f32,
    ) {
        let quads = nine_slice.generate_quads(x, y, w, h, atlas_w, atlas_h);
        for (pos, size, rect) in quads {
            let mut sprite = Sprite::new(pos[0] + size[0] * 0.5, pos[1] + size[1] * 0.5, size[0], size[1]);
            sprite.rect = rect;
            sprite.atlas = nine_slice.atlas;
            sprite.tint = nine_slice.tint;
            sprite.layer = layer;
            sprite.anchor = [0.5, 0.5];
            self.submit(sprite);
        }
    }

    /// Sort all submitted sprites and compile them into batches.
    pub fn flush(&mut self) {
        // Sort sprites.
        self.sprites.sort_by(|a, b| {
            a.sort_key().cmp(&b.sort_key())
        });

        // Apply pixel-perfect snapping.
        if self.pixel_perfect != PixelPerfectMode::None {
            for sprite in &mut self.sprites {
                sprite.position[0] = sprite.position[0].round();
                sprite.position[1] = sprite.position[1].round();

                if self.pixel_perfect == PixelPerfectMode::SnapPositionAndSize {
                    sprite.size[0] = sprite.size[0].round();
                    sprite.size[1] = sprite.size[1].round();
                }
            }
        }

        // Apply camera transform.
        for sprite in &mut self.sprites {
            sprite.position[0] = (sprite.position[0] - self.camera_offset[0]) * self.camera_zoom;
            sprite.position[1] = (sprite.position[1] - self.camera_offset[1]) * self.camera_zoom;
            sprite.size[0] *= self.camera_zoom;
            sprite.size[1] *= self.camera_zoom;

            // Apply global tint.
            sprite.tint[0] *= self.global_tint[0];
            sprite.tint[1] *= self.global_tint[1];
            sprite.tint[2] *= self.global_tint[2];
            sprite.tint[3] *= self.global_tint[3];
        }

        // Group by atlas and create batches.
        self.batches.clear();

        // Collect unique atlases in order.
        let mut atlas_order: Vec<u64> = Vec::new();
        for sprite in &self.sprites {
            if !atlas_order.contains(&sprite.atlas) {
                atlas_order.push(sprite.atlas);
            }
        }

        for atlas in &atlas_order {
            let mut batch = SpriteBatch::new(*atlas);
            for sprite in &self.sprites {
                if sprite.atlas == *atlas {
                    batch.add_sprite(sprite);
                }
            }
            if batch.sprite_count > 0 {
                self.batches.push(batch);
            }
        }

        self.sprites.clear();
    }

    /// Get the compiled batches for rendering.
    pub fn batches(&self) -> &[SpriteBatch] {
        &self.batches
    }

    /// Get the orthographic projection matrix (column-major 4x4).
    pub fn projection_matrix(&self) -> [f32; 16] {
        let w = self.screen_size.0 as f32;
        let h = self.screen_size.1 as f32;

        // Orthographic: origin at top-left, Y down.
        [
            2.0 / w, 0.0, 0.0, 0.0,
            0.0, -2.0 / h, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
        ]
    }

    /// Get total sprite count across all batches.
    pub fn total_sprites(&self) -> u32 {
        self.batches.iter().map(|b| b.sprite_count).sum()
    }

    /// Get total vertex count.
    pub fn total_vertices(&self) -> usize {
        self.batches.iter().map(|b| b.vertices.len()).sum()
    }

    /// Get total index count.
    pub fn total_indices(&self) -> usize {
        self.batches.iter().map(|b| b.indices.len()).sum()
    }

    /// Get number of draw calls (batches).
    pub fn draw_call_count(&self) -> usize {
        self.batches.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprite_rect_corners() {
        let rect = SpriteRect::new(0.0, 0.0, 1.0, 1.0);
        let corners = rect.corners();
        assert_eq!(corners[0], [0.0, 0.0]);
        assert_eq!(corners[2], [1.0, 1.0]);
    }

    #[test]
    fn test_animation_loop() {
        let frames = vec![
            SpriteRect::new(0.0, 0.0, 0.25, 1.0),
            SpriteRect::new(0.25, 0.0, 0.25, 1.0),
            SpriteRect::new(0.5, 0.0, 0.25, 1.0),
            SpriteRect::new(0.75, 0.0, 0.25, 1.0),
        ];
        let anim = SpriteAnimation::new("walk", frames, 10.0);

        assert_eq!(anim.frame_at_time(0.0), 0);
        assert_eq!(anim.frame_at_time(0.1), 1);
        assert_eq!(anim.frame_at_time(0.4), 0); // Loops back.
    }

    #[test]
    fn test_sprite_batch() {
        let sprite = Sprite::new(100.0, 100.0, 32.0, 32.0);
        let mut batch = SpriteBatch::new(1);
        batch.add_sprite(&sprite);

        assert_eq!(batch.sprite_count, 1);
        assert_eq!(batch.vertices.len(), 4);
        assert_eq!(batch.indices.len(), 6);
    }

    #[test]
    fn test_sprite_sorting() {
        let mut renderer = SpriteRenderer::new(800, 600);

        let s1 = Sprite { layer: 2, ..Sprite::default() };
        let s2 = Sprite { layer: 0, ..Sprite::default() };
        let s3 = Sprite { layer: 1, ..Sprite::default() };

        renderer.submit(s1);
        renderer.submit(s2);
        renderer.submit(s3);
        renderer.flush();

        // All in the same atlas (0), so one batch.
        assert_eq!(renderer.draw_call_count(), 1);
        assert_eq!(renderer.total_sprites(), 3);
    }

    #[test]
    fn test_nine_slice() {
        let rect = SpriteRect::new(0.0, 0.0, 1.0, 1.0);
        let ns = NineSlice::new(rect, 10.0, 10.0, 10.0, 10.0, 1);
        let quads = ns.generate_quads(0.0, 0.0, 100.0, 100.0, 128.0, 128.0);
        assert_eq!(quads.len(), 9);
    }
}
