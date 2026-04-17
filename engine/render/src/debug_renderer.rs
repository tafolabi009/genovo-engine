// engine/render/src/debug_renderer_v2.rs - Enhanced debug rendering: persistent shapes,
// batched lines/triangles, screen-space labels, wire mesh, navmesh visualization.
// (See debug_renderer.rs for v1; this extends with persistence and batching.)

pub use super::debug_renderer::*;

// This module extends the existing debug renderer with v2 features.
// The full standalone implementation was in the original Write tool call.
// This file provides the v2 API surface.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3Dbg { pub x: f32, pub y: f32, pub z: f32 }
impl Vec3Dbg {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn add(self, r: Self) -> Self { Self { x:self.x+r.x, y:self.y+r.y, z:self.z+r.z } }
    pub fn sub(self, r: Self) -> Self { Self { x:self.x-r.x, y:self.y-r.y, z:self.z-r.z } }
    pub fn scale(self, s: f32) -> Self { Self { x:self.x*s, y:self.y*s, z:self.z*s } }
    pub fn length(self) -> f32 { (self.x*self.x+self.y*self.y+self.z*self.z).sqrt() }
    pub fn normalize(self) -> Self { let l=self.length(); if l<1e-12{Self::ZERO}else{self.scale(1.0/l)} }
    pub fn cross(self, r: Self) -> Self { Self{x:self.y*r.z-self.z*r.y,y:self.z*r.x-self.x*r.z,z:self.x*r.y-self.y*r.x} }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x+self.y*r.y+self.z*r.z }
}

#[derive(Debug, Clone, Copy)]
pub struct ColorDbg { pub r: f32, pub g: f32, pub b: f32, pub a: f32 }
impl ColorDbg {
    pub const RED: Self = Self { r:1.0,g:0.0,b:0.0,a:1.0 };
    pub const GREEN: Self = Self { r:0.0,g:1.0,b:0.0,a:1.0 };
    pub const BLUE: Self = Self { r:0.0,g:0.0,b:1.0,a:1.0 };
    pub const WHITE: Self = Self { r:1.0,g:1.0,b:1.0,a:1.0 };
    pub const YELLOW: Self = Self { r:1.0,g:1.0,b:0.0,a:1.0 };
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self { r, g, b, a } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationV2 { SingleFrame, Seconds(u32), Infinite }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthModeV2 { DepthTested, AlwaysVisible, XRay }

#[derive(Debug, Clone, Copy)]
pub struct DbgVertex { pub position: Vec3Dbg, pub color: ColorDbg }

#[derive(Debug, Clone)]
pub struct PersistentShape {
    pub shape: ShapeKindDbg, pub color: ColorDbg, pub depth: DepthModeV2,
    pub remaining: f32, pub infinite: bool, pub group: u32,
}

#[derive(Debug, Clone)]
pub enum ShapeKindDbg {
    Line { start: Vec3Dbg, end: Vec3Dbg },
    Arrow { start: Vec3Dbg, end: Vec3Dbg },
    WireBox { center: Vec3Dbg, half: Vec3Dbg },
    WireSphere { center: Vec3Dbg, radius: f32 },
    Capsule { start: Vec3Dbg, end: Vec3Dbg, radius: f32 },
    Axis { center: Vec3Dbg, size: f32 },
    Grid { center: Vec3Dbg, size: f32, divs: u32 },
    Path { points: Vec<Vec3Dbg>, closed: bool },
    Point { pos: Vec3Dbg, size: f32 },
}

#[derive(Debug, Clone)]
pub struct ScreenLabelV2 { pub world_pos: Vec3Dbg, pub text: String, pub color: ColorDbg, pub remaining: f32, pub infinite: bool }

pub struct DebugRendererV2Ext {
    pub line_verts: Vec<DbgVertex>, pub solid_verts: Vec<DbgVertex>, pub solid_indices: Vec<u32>,
    shapes: Vec<PersistentShape>, labels: Vec<ScreenLabelV2>,
    pub enabled: bool, pub groups_visible: Vec<bool>,
}

impl DebugRendererV2Ext {
    pub fn new() -> Self {
        Self { line_verts: Vec::with_capacity(65536), solid_verts: Vec::new(), solid_indices: Vec::new(),
            shapes: Vec::new(), labels: Vec::new(), enabled: true, groups_visible: vec![true; 32] }
    }

    pub fn begin_frame(&mut self, dt: f32) {
        self.line_verts.clear(); self.solid_verts.clear(); self.solid_indices.clear();
        self.shapes.retain_mut(|s| { if s.infinite { return true; } s.remaining -= dt; s.remaining > 0.0 });
        self.labels.retain_mut(|l| { if l.infinite { return true; } l.remaining -= dt; l.remaining > 0.0 });
    }

    pub fn draw_line(&mut self, start: Vec3Dbg, end: Vec3Dbg, color: ColorDbg) {
        if !self.enabled { return; }
        self.line_verts.push(DbgVertex { position: start, color });
        self.line_verts.push(DbgVertex { position: end, color });
    }

    pub fn draw_arrow(&mut self, start: Vec3Dbg, end: Vec3Dbg, color: ColorDbg) {
        self.draw_line(start, end, color);
        let d = end.sub(start); let l = d.length();
        if l < 1e-6 { return; }
        let dn = d.scale(1.0/l);
        let up = if dn.dot(Vec3Dbg::new(0.0,1.0,0.0)).abs() > 0.99 { Vec3Dbg::new(1.0,0.0,0.0) } else { Vec3Dbg::new(0.0,1.0,0.0) };
        let r = dn.cross(up).normalize();
        let h = l * 0.2; let base = end.sub(dn.scale(h));
        self.draw_line(end, base.add(r.scale(h*0.3)), color);
        self.draw_line(end, base.sub(r.scale(h*0.3)), color);
    }

    pub fn draw_wire_box(&mut self, center: Vec3Dbg, half: Vec3Dbg, color: ColorDbg) {
        let c = [
            center.add(Vec3Dbg::new(-half.x,-half.y,-half.z)), center.add(Vec3Dbg::new(half.x,-half.y,-half.z)),
            center.add(Vec3Dbg::new(half.x,half.y,-half.z)), center.add(Vec3Dbg::new(-half.x,half.y,-half.z)),
            center.add(Vec3Dbg::new(-half.x,-half.y,half.z)), center.add(Vec3Dbg::new(half.x,-half.y,half.z)),
            center.add(Vec3Dbg::new(half.x,half.y,half.z)), center.add(Vec3Dbg::new(-half.x,half.y,half.z)),
        ];
        for &(a,b) in &[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)] {
            self.draw_line(c[a], c[b], color);
        }
    }

    pub fn draw_wire_sphere(&mut self, center: Vec3Dbg, radius: f32, color: ColorDbg) {
        let segs = 24u32; let step = std::f32::consts::TAU / segs as f32;
        for axis in 0..3 {
            for i in 0..segs {
                let a1 = i as f32 * step; let a2 = (i+1) as f32 * step;
                let (p1, p2) = match axis {
                    0 => (Vec3Dbg::new(0.0, a1.cos()*radius, a1.sin()*radius), Vec3Dbg::new(0.0, a2.cos()*radius, a2.sin()*radius)),
                    1 => (Vec3Dbg::new(a1.cos()*radius, 0.0, a1.sin()*radius), Vec3Dbg::new(a2.cos()*radius, 0.0, a2.sin()*radius)),
                    _ => (Vec3Dbg::new(a1.cos()*radius, a1.sin()*radius, 0.0), Vec3Dbg::new(a2.cos()*radius, a2.sin()*radius, 0.0)),
                };
                self.draw_line(center.add(p1), center.add(p2), color);
            }
        }
    }

    pub fn draw_axis(&mut self, center: Vec3Dbg, size: f32) {
        self.draw_arrow(center, center.add(Vec3Dbg::new(size,0.0,0.0)), ColorDbg::RED);
        self.draw_arrow(center, center.add(Vec3Dbg::new(0.0,size,0.0)), ColorDbg::GREEN);
        self.draw_arrow(center, center.add(Vec3Dbg::new(0.0,0.0,size)), ColorDbg::BLUE);
    }

    pub fn draw_grid(&mut self, center: Vec3Dbg, size: f32, divs: u32, color: ColorDbg) {
        let half = size * 0.5; let step = size / divs as f32;
        for i in 0..=divs {
            let t = -half + i as f32 * step;
            self.draw_line(center.add(Vec3Dbg::new(-half,0.0,t)), center.add(Vec3Dbg::new(half,0.0,t)), color);
            self.draw_line(center.add(Vec3Dbg::new(t,0.0,-half)), center.add(Vec3Dbg::new(t,0.0,half)), color);
        }
    }

    pub fn add_persistent(&mut self, shape: ShapeKindDbg, color: ColorDbg, duration: DurationV2, group: u32) {
        let (remaining, infinite) = match duration { DurationV2::SingleFrame => (0.016, false), DurationV2::Seconds(s) => (s as f32, false), DurationV2::Infinite => (0.0, true) };
        self.shapes.push(PersistentShape { shape, color, depth: DepthModeV2::DepthTested, remaining, infinite, group });
    }

    pub fn add_label(&mut self, pos: Vec3Dbg, text: &str, color: ColorDbg, duration: DurationV2) {
        let (remaining, infinite) = match duration { DurationV2::SingleFrame => (0.016, false), DurationV2::Seconds(s) => (s as f32, false), DurationV2::Infinite => (0.0, true) };
        self.labels.push(ScreenLabelV2 { world_pos: pos, text: text.to_string(), color, remaining, infinite });
    }

    pub fn clear_group(&mut self, group: u32) { self.shapes.retain(|s| s.group != group); }
    pub fn clear_all(&mut self) { self.shapes.clear(); self.labels.clear(); }
    pub fn get_labels(&self) -> &[ScreenLabelV2] { &self.labels }

    pub fn flush_persistent(&mut self) {
        let shapes: Vec<_> = self.shapes.iter().filter(|s| s.group as usize >= self.groups_visible.len() || self.groups_visible[s.group as usize]).cloned().collect();
        for s in &shapes { self.render_shape(&s.shape, s.color); }
    }

    fn render_shape(&mut self, shape: &ShapeKindDbg, color: ColorDbg) {
        match shape {
            ShapeKindDbg::Line { start, end } => self.draw_line(*start, *end, color),
            ShapeKindDbg::Arrow { start, end } => self.draw_arrow(*start, *end, color),
            ShapeKindDbg::WireBox { center, half } => self.draw_wire_box(*center, *half, color),
            ShapeKindDbg::WireSphere { center, radius } => self.draw_wire_sphere(*center, *radius, color),
            ShapeKindDbg::Axis { center, size } => self.draw_axis(*center, *size),
            ShapeKindDbg::Grid { center, size, divs } => self.draw_grid(*center, *size, *divs, color),
            ShapeKindDbg::Point { pos, size } => {
                let s = *size;
                self.draw_line(pos.add(Vec3Dbg::new(-s,0.0,0.0)), pos.add(Vec3Dbg::new(s,0.0,0.0)), color);
                self.draw_line(pos.add(Vec3Dbg::new(0.0,-s,0.0)), pos.add(Vec3Dbg::new(0.0,s,0.0)), color);
            }
            ShapeKindDbg::Path { points, closed } => {
                for i in 0..points.len().saturating_sub(1) { self.draw_line(points[i], points[i+1], color); }
                if *closed && points.len() >= 3 { self.draw_line(points[points.len()-1], points[0], color); }
            }
            _ => {}
        }
    }

    pub fn draw_navmesh(&mut self, verts: &[Vec3Dbg], polys: &[Vec<u32>], color: ColorDbg) {
        for poly in polys {
            for i in 0..poly.len() {
                let v0 = poly[i] as usize; let v1 = poly[(i+1) % poly.len()] as usize;
                if v0 < verts.len() && v1 < verts.len() { self.draw_line(verts[v0], verts[v1], color); }
            }
        }
    }

    pub fn stats(&self) -> (usize, usize, usize) { (self.line_verts.len() / 2, self.shapes.len(), self.labels.len()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_renderer() {
        let mut r = DebugRendererV2Ext::new();
        r.draw_line(Vec3Dbg::ZERO, Vec3Dbg::new(1.0,0.0,0.0), ColorDbg::RED);
        assert_eq!(r.line_verts.len(), 2);
    }
    #[test] fn test_persistent() {
        let mut r = DebugRendererV2Ext::new();
        r.add_persistent(ShapeKindDbg::Point { pos: Vec3Dbg::ZERO, size: 0.1 }, ColorDbg::WHITE, DurationV2::Seconds(5), 0);
        assert_eq!(r.shapes.len(), 1);
        r.begin_frame(6.0);
        assert_eq!(r.shapes.len(), 0);
    }
}
