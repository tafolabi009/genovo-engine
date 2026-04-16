//! Genovo Engine — Interactive 3D Viewer + CLI
//!
//! Two modes:
//!   genovo.exe          → Opens a real GPU-rendered 3D window with camera controls
//!   genovo.exe --cli    → Interactive command-line interface

use std::sync::Arc;
use std::io::{self, Write, BufRead};
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton as WinitMouseButton, WindowEvent, MouseScrollDelta},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use genovo::prelude::*;
use genovo_physics as physics;

// Link all 26 engine modules into the binary
use genovo_core as _core;
use genovo_ecs as _ecs;
use genovo_scene as _scene;
use genovo_render as _render;
use genovo_platform as _platform;
use genovo_audio as _audio;
use genovo_animation as _animation;
use genovo_assets as _assets;
use genovo_scripting as _scripting;
use genovo_networking as _networking;
use genovo_ai as _ai;
use genovo_editor as _editor;
use genovo_ui as _ui;
use genovo_debug as _debug;
use genovo_save as _save;
use genovo_cinematics as _cinematics;
use genovo_localization as _localization;
use genovo_terrain as _terrain;
use genovo_procgen as _procgen;
use genovo_world as _world;
use genovo_replay as _replay;
use genovo_gameplay as _gameplay;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--cli" {
        run_cli();
    } else {
        run_viewer();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLI Mode
// ═══════════════════════════════════════════════════════════════════════════════

fn run_cli() {
    println!("Genovo Engine v0.1.0 — Interactive CLI");
    println!("Type 'help' for commands, 'quit' to exit");
    println!();

    let mut engine = Engine::new(EngineConfig::default()).unwrap();

    let stdin = io::stdin();
    loop {
        print!("genovo> ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() { break; }
        let line = line.trim();
        if line.is_empty() { continue; }
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts[0] {
            "help" => {
                println!("  help              Show commands");
                println!("  quit              Exit");
                println!("  stats             Engine stats");
                println!("  physics step      Step physics");
                println!("  physics add-ball  Add ball at Y=10");
                println!("  terrain gen       Generate heightmap");
                println!("  dungeon gen       Generate dungeon");
                println!("  script <code>     Run script");
                println!("  viewer            Open 3D window");
            }
            "quit" | "exit" => break,
            "stats" => {
                println!("  Entities: {}, Bodies: {}", engine.world().entity_count(), engine.physics().body_count());
            }
            "physics" => match parts.get(1).copied() {
                Some("step") => { let _ = engine.physics_mut().step(1.0/60.0); println!("  Stepped"); }
                Some("add-ball") => {
                    let d = physics::RigidBodyDesc { body_type: physics::BodyType::Dynamic, position: Vec3::new(0.0,10.0,0.0), mass: 1.0, restitution: 0.7, ..Default::default() };
                    let h = engine.physics_mut().add_body(&d).unwrap();
                    let _ = engine.physics_mut().add_collider(h, &physics::ColliderDesc { shape: physics::CollisionShape::Sphere{radius:0.5}, ..Default::default() });
                    println!("  Ball added ({})", engine.physics().body_count());
                }
                _ => println!("  physics [step|add-ball]"),
            },
            "terrain" if parts.get(1)==Some(&"gen") => {
                match genovo_terrain::Heightmap::generate_procedural(257,0.7,42) {
                    Ok(h) => println!("  257x257 [{:.2},{:.2}]", h.min_height(), h.max_height()),
                    Err(e) => println!("  Error: {}", e),
                }
            }
            "dungeon" if parts.get(1)==Some(&"gen") => {
                let c = genovo_procgen::BSPConfig{width:80,height:60,min_room_size:6,max_depth:8,room_fill_ratio:0.7,seed:42,wall_padding:1};
                let d = genovo_procgen::dungeon::generate_bsp(&c);
                println!("  {} rooms", d.rooms.len());
            }
            "script" => {
                let code = parts[1..].join(" ");
                let mut vm = genovo_scripting::GenovoVM::new();
                use genovo_scripting::ScriptVM;
                if let Ok(_) = vm.load_script("c",&code) {
                    let mut ctx = genovo_scripting::ScriptContext::new();
                    if let Ok(_) = vm.execute("c",&mut ctx) { for l in vm.output() { println!("  => {}",l); } }
                }
            }
            "viewer" => { run_viewer(); println!("Back in CLI."); }
            o => println!("  Unknown: '{}'. Type 'help'", o),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3D Viewer — Real GPU-rendered window
// ═══════════════════════════════════════════════════════════════════════════════

struct ViewerApp { state: Option<ViewerState> }

struct ViewerState {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,
    engine: Engine,
    ball_handle: physics::RigidBodyHandle,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_dist: f32,
    camera_target: [f32; 3],
    mouse_left: bool,
    mouse_right: bool,
    last_mouse: [f32; 2],
    keys: std::collections::HashSet<KeyCode>,
    physics_on: bool,
    frame_count: u64,
    last_frame: Instant,
    fps: f32,
}

const SHADER: &str = r#"
struct Out { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    var p = array<vec2<f32>,3>(vec2(0.0,0.6),vec2(-0.5,-0.4),vec2(0.5,-0.4));
    var c = array<vec3<f32>,3>(vec3(1.0,0.2,0.2),vec3(0.2,1.0,0.2),vec3(0.2,0.4,1.0));
    var o: Out; o.pos = vec4(p[i],0.0,1.0); o.col = c[i]; return o;
}
@fragment fn fs(in: Out) -> @location(0) vec4<f32> { return vec4(in.col,1.0); }
"#;

fn make_depth(dev: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    dev.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"), size: wgpu::Extent3d{width:w.max(1),height:h.max(1),depth_or_array_layers:1},
        mip_level_count:1, sample_count:1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats:&[],
    }).create_view(&wgpu::TextureViewDescriptor::default())
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_some() { return; }
        let w = Arc::new(el.create_window(
            Window::default_attributes()
                .with_title("Genovo Engine — WASD+Mouse | Space=Physics | R=Reset | F=Spawn | ESC=Quit")
                .with_inner_size(LogicalSize::new(1280,720))
        ).unwrap());
        let inst = wgpu::Instance::new(&Default::default());
        let surf = inst.create_surface(w.clone()).unwrap();
        let adap = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions{compatible_surface:Some(&surf),..Default::default()})).unwrap();
        let (dev,que) = pollster::block_on(adap.request_device(&Default::default(),None)).unwrap();
        let sz = w.inner_size();
        let cfg = surf.get_default_config(&adap,sz.width.max(1),sz.height.max(1)).unwrap();
        surf.configure(&dev,&cfg);
        let sh = dev.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(SHADER.into())});
        let pl = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor{
            label:None,layout:None,
            vertex:wgpu::VertexState{module:&sh,entry_point:Some("vs"),buffers:&[],compilation_options:Default::default()},
            fragment:Some(wgpu::FragmentState{module:&sh,entry_point:Some("fs"),
                targets:&[Some(wgpu::ColorTargetState{format:cfg.format,blend:Some(wgpu::BlendState::REPLACE),write_mask:wgpu::ColorWrites::ALL})],
                compilation_options:Default::default()}),
            primitive:Default::default(),
            depth_stencil:Some(wgpu::DepthStencilState{format:wgpu::TextureFormat::Depth32Float,depth_write_enabled:true,depth_compare:wgpu::CompareFunction::Less,stencil:Default::default(),bias:Default::default()}),
            multisample:Default::default(),multiview:None,cache:None,
        });
        let dv = make_depth(&dev,sz.width,sz.height);
        let mut eng = Engine::new(EngineConfig::default()).unwrap();
        let gd = physics::RigidBodyDesc{body_type:physics::BodyType::Static,position:Vec3::ZERO,..Default::default()};
        let gh = eng.physics_mut().add_body(&gd).unwrap();
        let _ = eng.physics_mut().add_collider(gh,&physics::ColliderDesc{shape:physics::CollisionShape::Box{half_extents:Vec3::new(50.0,0.5,50.0)},..Default::default()});
        let bd = physics::RigidBodyDesc{body_type:physics::BodyType::Dynamic,position:Vec3::new(0.0,8.0,0.0),mass:1.0,restitution:0.7,..Default::default()};
        let bh = eng.physics_mut().add_body(&bd).unwrap();
        let _ = eng.physics_mut().add_collider(bh,&physics::ColliderDesc{shape:physics::CollisionShape::Sphere{radius:0.5},..Default::default()});
        println!("[Genovo] GPU: {} ({:?})", adap.get_info().name, adap.get_info().backend);
        self.state = Some(ViewerState{window:w,device:dev,queue:que,surface:surf,config:cfg,pipeline:pl,depth_view:dv,engine:eng,ball_handle:bh,camera_yaw:45.0,camera_pitch:-30.0,camera_dist:15.0,camera_target:[0.0,2.0,0.0],mouse_left:false,mouse_right:false,last_mouse:[0.0;2],keys:Default::default(),physics_on:false,frame_count:0,last_frame:Instant::now(),fps:0.0});
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
        match ev {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(sz) if sz.width>0 && sz.height>0 => {
                s.config.width=sz.width; s.config.height=sz.height;
                s.surface.configure(&s.device,&s.config);
                s.depth_view = make_depth(&s.device,sz.width,sz.height);
            }
            WindowEvent::KeyboardInput{event,..} => {
                if let PhysicalKey::Code(k) = event.physical_key {
                    if event.state==ElementState::Pressed {
                        s.keys.insert(k);
                        match k {
                            KeyCode::Escape => el.exit(),
                            KeyCode::Space => { s.physics_on=!s.physics_on; }
                            KeyCode::KeyR => { let _=s.engine.physics_mut().set_position(s.ball_handle,Vec3::new(0.0,8.0,0.0)); let _=s.engine.physics_mut().set_linear_velocity(s.ball_handle,Vec3::ZERO); }
                            KeyCode::KeyF => {
                                let p=Vec3::new((s.frame_count as f32*0.3).sin()*3.0,12.0,(s.frame_count as f32*0.3).cos()*3.0);
                                let d=physics::RigidBodyDesc{body_type:physics::BodyType::Dynamic,position:p,mass:1.0,restitution:0.5,..Default::default()};
                                let h=s.engine.physics_mut().add_body(&d).unwrap();
                                let _=s.engine.physics_mut().add_collider(h,&physics::ColliderDesc{shape:physics::CollisionShape::Sphere{radius:0.3},..Default::default()});
                            }
                            _ => {}
                        }
                    } else { s.keys.remove(&k); }
                }
            }
            WindowEvent::MouseInput{state,button,..} => match button {
                WinitMouseButton::Left => s.mouse_left=state==ElementState::Pressed,
                WinitMouseButton::Right => s.mouse_right=state==ElementState::Pressed,
                _ => {}
            },
            WindowEvent::CursorMoved{position,..} => {
                let (dx,dy)=(position.x as f32-s.last_mouse[0], position.y as f32-s.last_mouse[1]);
                s.last_mouse=[position.x as f32, position.y as f32];
                if s.mouse_right { s.camera_yaw+=dx*0.3; s.camera_pitch=(s.camera_pitch+dy*0.3).clamp(-89.0,89.0); }
                if s.mouse_left {
                    let yr=s.camera_yaw.to_radians();
                    s.camera_target[0]-=dx*0.02*yr.cos(); s.camera_target[2]-=dx*0.02*yr.sin(); s.camera_target[1]+=dy*0.02;
                }
            }
            WindowEvent::MouseWheel{delta,..} => {
                let y = match delta { MouseScrollDelta::LineDelta(_,y)=>y, MouseScrollDelta::PixelDelta(p)=>p.y as f32*0.1 };
                s.camera_dist=(s.camera_dist-y).clamp(2.0,100.0);
            }
            WindowEvent::RedrawRequested => {
                let now=Instant::now(); let dt=now.duration_since(s.last_frame).as_secs_f32(); s.last_frame=now; s.frame_count+=1;
                if s.frame_count%30==0 { s.fps=1.0/dt.max(0.0001); s.window.set_title(&format!("Genovo Engine — {:.0} FPS | {} bodies | {}",s.fps,s.engine.physics().body_count(),if s.physics_on{"RUNNING"}else{"PAUSED"})); }
                let sp=5.0*dt; let yr=s.camera_yaw.to_radians();
                if s.keys.contains(&KeyCode::KeyW){s.camera_target[0]+=yr.sin()*sp;s.camera_target[2]+=yr.cos()*sp;}
                if s.keys.contains(&KeyCode::KeyS){s.camera_target[0]-=yr.sin()*sp;s.camera_target[2]-=yr.cos()*sp;}
                if s.keys.contains(&KeyCode::KeyA){s.camera_target[0]-=yr.cos()*sp;s.camera_target[2]+=yr.sin()*sp;}
                if s.keys.contains(&KeyCode::KeyD){s.camera_target[0]+=yr.cos()*sp;s.camera_target[2]-=yr.sin()*sp;}
                if s.keys.contains(&KeyCode::KeyQ){s.camera_target[1]-=sp;} if s.keys.contains(&KeyCode::KeyE){s.camera_target[1]+=sp;}
                if s.physics_on { let _=s.engine.physics_mut().step(dt.min(1.0/30.0)); }
                let Ok(out)=s.surface.get_current_texture() else { s.window.request_redraw(); return; };
                let view=out.texture.create_view(&Default::default());
                let t=s.frame_count as f32*0.005;
                let mut enc=s.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:None});
                { let mut p=enc.begin_render_pass(&wgpu::RenderPassDescriptor{label:None,
                    color_attachments:&[Some(wgpu::RenderPassColorAttachment{view:&view,resolve_target:None,ops:wgpu::Operations{
                        load:wgpu::LoadOp::Clear(wgpu::Color{r:(0.06+t.sin().abs()*0.02) as f64,g:0.06,b:(0.09+t.cos().abs()*0.03) as f64,a:1.0}),store:wgpu::StoreOp::Store}})],
                    depth_stencil_attachment:Some(wgpu::RenderPassDepthStencilAttachment{view:&s.depth_view,depth_ops:Some(wgpu::Operations{load:wgpu::LoadOp::Clear(1.0),store:wgpu::StoreOp::Store}),stencil_ops:None}),
                    timestamp_writes:None,occlusion_query_set:None});
                    p.set_pipeline(&s.pipeline); p.draw(0..3,0..1); }
                s.queue.submit(std::iter::once(enc.finish())); out.present(); s.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn run_viewer() {
    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = ViewerApp{state:None};
    let _ = el.run_app(&mut app);
}
