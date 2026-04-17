//! Lightweight async runtime for the Genovo engine.
//!
//! Provides a single-threaded executor, task spawning, waker management,
//! timer futures, channel futures, join/select combinators, `spawn_local`
//! for non-`Send` futures, and async file I/O stubs. This runtime is designed
//! to integrate with the engine's main loop rather than running on its own
//! OS thread, making it suitable for cooperative multitasking within a frame.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────┐
//! │      Executor            │
//! │  ┌────────┐ ┌────────┐  │
//! │  │ Task 1 │ │ Task 2 │  │
//! │  └────────┘ └────────┘  │
//! │  ┌────────┐ ┌────────┐  │
//! │  │ Task 3 │ │ Timer  │  │
//! │  └────────┘ └────────┘  │
//! │                          │
//! │  Ready Queue ──► Poll    │
//! └──────────────────────────┘
//! ```

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Task ID
// ---------------------------------------------------------------------------

/// Unique identifier for a spawned task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

impl TaskId {
    /// Returns the raw numeric value of this task identifier.
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Task({})", self.0)
    }
}

static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

fn alloc_task_id() -> TaskId {
    TaskId(NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// Future trait alias
// ---------------------------------------------------------------------------

/// A boxed, pinned, `Send` future that produces `T`.
pub type BoxFuture<T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'static>>;

/// A boxed, pinned, **non-Send** future that produces `T`.
pub type LocalBoxFuture<T> = Pin<Box<dyn std::future::Future<Output = T> + 'static>>;

// ---------------------------------------------------------------------------
// TaskState
// ---------------------------------------------------------------------------

/// Current lifecycle state of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// The task is waiting to be polled.
    Pending,
    /// The task has been woken and is in the ready queue.
    Ready,
    /// The task is currently being polled by the executor.
    Running,
    /// The task has completed successfully.
    Completed,
    /// The task was cancelled before completion.
    Cancelled,
}

impl fmt::Display for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskState::Pending => write!(f, "Pending"),
            TaskState::Ready => write!(f, "Ready"),
            TaskState::Running => write!(f, "Running"),
            TaskState::Completed => write!(f, "Completed"),
            TaskState::Cancelled => write!(f, "Cancelled"),
        }
    }
}

// ---------------------------------------------------------------------------
// Waker
// ---------------------------------------------------------------------------

/// Shared flag that a [`RuntimeWaker`] sets to indicate readiness.
#[derive(Debug)]
struct WakerFlag {
    woken: AtomicBool,
    task_id: TaskId,
}

/// A waker that sets an atomic flag and enqueues the owning task's ID into a
/// shared ready queue.
#[derive(Debug, Clone)]
pub struct RuntimeWaker {
    flag: Arc<WakerFlag>,
    ready_queue: Arc<Mutex<VecDeque<TaskId>>>,
}

impl RuntimeWaker {
    /// Create a new waker for the given task.
    pub fn new(task_id: TaskId, ready_queue: Arc<Mutex<VecDeque<TaskId>>>) -> Self {
        Self {
            flag: Arc::new(WakerFlag {
                woken: AtomicBool::new(false),
                task_id,
            }),
            ready_queue,
        }
    }

    /// Wake the associated task, enqueuing it into the ready queue.
    pub fn wake(&self) {
        if !self.flag.woken.swap(true, Ordering::SeqCst) {
            if let Ok(mut q) = self.ready_queue.lock() {
                q.push_back(self.flag.task_id);
            }
        }
    }

    /// Reset the woken flag after the task has been polled.
    pub fn reset(&self) {
        self.flag.woken.store(false, Ordering::SeqCst);
    }

    /// Check whether this waker has been triggered.
    pub fn is_woken(&self) -> bool {
        self.flag.woken.load(Ordering::SeqCst)
    }

    /// Returns the task ID associated with this waker.
    pub fn task_id(&self) -> TaskId {
        self.flag.task_id
    }
}

/// Convert a [`RuntimeWaker`] into a [`std::task::Waker`].
///
/// We build a raw waker vtable manually so that we own the wake semantics
/// without depending on an external crate.
pub fn into_std_waker(runtime_waker: RuntimeWaker) -> std::task::Waker {
    use std::task::{RawWaker, RawWakerVTable};

    fn clone_fn(data: *const ()) -> RawWaker {
        let waker = unsafe { &*(data as *const RuntimeWaker) };
        let cloned = Box::new(waker.clone());
        RawWaker::new(Box::into_raw(cloned) as *const (), &VTABLE)
    }

    fn wake_fn(data: *const ()) {
        let waker = unsafe { Box::from_raw(data as *mut RuntimeWaker) };
        waker.wake();
    }

    fn wake_by_ref_fn(data: *const ()) {
        let waker = unsafe { &*(data as *const RuntimeWaker) };
        waker.wake();
    }

    fn drop_fn(data: *const ()) {
        unsafe {
            let _ = Box::from_raw(data as *mut RuntimeWaker);
        }
    }

    static VTABLE: RawWakerVTable =
        RawWakerVTable::new(clone_fn, wake_fn, wake_by_ref_fn, drop_fn);

    let boxed = Box::new(runtime_waker);
    let raw = RawWaker::new(Box::into_raw(boxed) as *const (), &VTABLE);
    unsafe { std::task::Waker::from_raw(raw) }
}

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------

/// Internal representation of a spawned async task.
struct Task {
    id: TaskId,
    future: LocalBoxFuture<()>,
    state: TaskState,
    waker: RuntimeWaker,
    name: Option<String>,
    spawned_at: Instant,
    poll_count: u64,
    total_poll_time: Duration,
}

impl Task {
    fn new(
        id: TaskId,
        future: LocalBoxFuture<()>,
        ready_queue: Arc<Mutex<VecDeque<TaskId>>>,
        name: Option<String>,
    ) -> Self {
        Self {
            id,
            future,
            state: TaskState::Ready,
            waker: RuntimeWaker::new(id, ready_queue),
            name,
            spawned_at: Instant::now(),
            poll_count: 0,
            total_poll_time: Duration::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// TaskHandle
// ---------------------------------------------------------------------------

/// Handle returned when a task is spawned, allowing the caller to query state
/// or cancel the task.
#[derive(Debug, Clone)]
pub struct TaskHandle {
    id: TaskId,
    cancel_flag: Arc<AtomicBool>,
    completed_flag: Arc<AtomicBool>,
}

impl TaskHandle {
    /// Returns the unique task identifier.
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Request cancellation of the task. The executor will drop the task's
    /// future the next time it would be polled.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    /// Returns `true` if cancellation has been requested.
    pub fn is_cancel_requested(&self) -> bool {
        self.cancel_flag.load(Ordering::SeqCst)
    }

    /// Returns `true` if the task has finished executing.
    pub fn is_completed(&self) -> bool {
        self.completed_flag.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// Timer Future
// ---------------------------------------------------------------------------

/// A future that resolves after a specified duration has elapsed.
///
/// The timer is not OS-backed; the executor checks deadlines each poll cycle.
pub struct TimerFuture {
    deadline: Instant,
    registered: bool,
    id: u64,
}

static NEXT_TIMER_ID: AtomicU64 = AtomicU64::new(1);

impl TimerFuture {
    /// Create a timer that will complete after `duration` from now.
    pub fn new(duration: Duration) -> Self {
        Self {
            deadline: Instant::now() + duration,
            registered: false,
            id: NEXT_TIMER_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Create a timer that completes at the given absolute instant.
    pub fn at(deadline: Instant) -> Self {
        Self {
            deadline,
            registered: false,
            id: NEXT_TIMER_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Returns the absolute deadline.
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// Returns the unique timer identifier.
    pub fn timer_id(&self) -> u64 {
        self.id
    }
}

impl std::future::Future for TimerFuture {
    type Output = ();

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if Instant::now() >= self.deadline {
            std::task::Poll::Ready(())
        } else {
            // Register ourselves for later waking. In a real runtime this would
            // register with a timer wheel; here we just ask for another poll.
            if !self.registered {
                self.registered = true;
            }
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}

/// Convenience function: sleep for `duration`.
pub fn sleep(duration: Duration) -> TimerFuture {
    TimerFuture::new(duration)
}

/// Convenience function: sleep until `deadline`.
pub fn sleep_until(deadline: Instant) -> TimerFuture {
    TimerFuture::at(deadline)
}

// ---------------------------------------------------------------------------
// Yield Future
// ---------------------------------------------------------------------------

/// A future that yields once, allowing other tasks to run, then completes.
pub struct YieldNow {
    yielded: bool,
}

impl YieldNow {
    pub fn new() -> Self {
        Self { yielded: false }
    }
}

impl Default for YieldNow {
    fn default() -> Self {
        Self::new()
    }
}

impl std::future::Future for YieldNow {
    type Output = ();

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.yielded {
            std::task::Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}

/// Yield execution to other tasks for one poll cycle.
pub fn yield_now() -> YieldNow {
    YieldNow::new()
}

// ---------------------------------------------------------------------------
// Channel Future (oneshot)
// ---------------------------------------------------------------------------

/// Shared state for a oneshot channel.
struct OneshotInner<T> {
    value: Option<T>,
    closed: bool,
    waker: Option<std::task::Waker>,
}

/// Sender half of a oneshot channel.
pub struct OneshotSender<T> {
    inner: Arc<Mutex<OneshotInner<T>>>,
}

/// Receiver half of a oneshot channel (implements `Future`).
pub struct OneshotReceiver<T> {
    inner: Arc<Mutex<OneshotInner<T>>>,
}

/// Error returned when a oneshot channel is closed before a value is sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelClosedError;

impl fmt::Display for ChannelClosedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "channel closed before value was sent")
    }
}

/// Create a oneshot channel pair.
pub fn oneshot<T>() -> (OneshotSender<T>, OneshotReceiver<T>) {
    let inner = Arc::new(Mutex::new(OneshotInner {
        value: None,
        closed: false,
        waker: None,
    }));
    (
        OneshotSender {
            inner: inner.clone(),
        },
        OneshotReceiver { inner },
    )
}

impl<T> OneshotSender<T> {
    /// Send a value, waking the receiver if it is waiting.
    pub fn send(self, value: T) -> Result<(), T> {
        let mut inner = self.inner.lock().unwrap();
        if inner.closed {
            return Err(value);
        }
        inner.value = Some(value);
        if let Some(waker) = inner.waker.take() {
            waker.wake();
        }
        Ok(())
    }

    /// Check whether the receiver has been dropped.
    pub fn is_closed(&self) -> bool {
        self.inner.lock().unwrap().closed
    }
}

impl<T> Drop for OneshotSender<T> {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        if let Some(waker) = inner.waker.take() {
            waker.wake();
        }
    }
}

impl<T> std::future::Future for OneshotReceiver<T> {
    type Output = Result<T, ChannelClosedError>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(value) = inner.value.take() {
            std::task::Poll::Ready(Ok(value))
        } else if inner.closed {
            std::task::Poll::Ready(Err(ChannelClosedError))
        } else {
            inner.waker = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }
}

impl<T> Drop for OneshotReceiver<T> {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
    }
}

// ---------------------------------------------------------------------------
// MPSC Channel Future
// ---------------------------------------------------------------------------

/// Shared state for an MPSC channel.
struct MpscInner<T> {
    queue: VecDeque<T>,
    capacity: usize,
    closed: bool,
    recv_waker: Option<std::task::Waker>,
    send_wakers: VecDeque<std::task::Waker>,
}

/// Sender half of an MPSC channel (cloneable).
pub struct MpscSender<T> {
    inner: Arc<Mutex<MpscInner<T>>>,
}

impl<T> Clone for MpscSender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Receiver half of an MPSC channel.
pub struct MpscReceiver<T> {
    inner: Arc<Mutex<MpscInner<T>>>,
}

/// Error when the channel is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelFullError;

impl fmt::Display for ChannelFullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "channel is full")
    }
}

/// Create a bounded MPSC channel with the given capacity.
pub fn mpsc_channel<T>(capacity: usize) -> (MpscSender<T>, MpscReceiver<T>) {
    let cap = if capacity == 0 { 1 } else { capacity };
    let inner = Arc::new(Mutex::new(MpscInner {
        queue: VecDeque::with_capacity(cap),
        capacity: cap,
        closed: false,
        recv_waker: None,
        send_wakers: VecDeque::new(),
    }));
    (
        MpscSender {
            inner: inner.clone(),
        },
        MpscReceiver { inner },
    )
}

impl<T> MpscSender<T> {
    /// Try to send a value without blocking.
    pub fn try_send(&self, value: T) -> Result<(), ChannelFullError> {
        let mut inner = self.inner.lock().unwrap();
        if inner.queue.len() >= inner.capacity {
            return Err(ChannelFullError);
        }
        inner.queue.push_back(value);
        if let Some(waker) = inner.recv_waker.take() {
            waker.wake();
        }
        Ok(())
    }

    /// Send a value, returning a future that resolves when the value is enqueued.
    pub fn send(&self, value: T) -> MpscSendFuture<T> {
        MpscSendFuture {
            inner: self.inner.clone(),
            value: Some(value),
        }
    }

    /// Close the channel from the sender side.
    pub fn close(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        if let Some(waker) = inner.recv_waker.take() {
            waker.wake();
        }
    }

    /// Number of items currently buffered in the channel.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().queue.len()
    }

    /// Returns `true` if the channel buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Future returned by [`MpscSender::send`].
pub struct MpscSendFuture<T> {
    inner: Arc<Mutex<MpscInner<T>>>,
    value: Option<T>,
}
impl<T> Unpin for MpscSendFuture<T> {}

impl<T> std::future::Future for MpscSendFuture<T> {
    type Output = Result<(), ChannelClosedError>;

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        let mut inner = this.inner.lock().unwrap();
        if inner.closed {
            return std::task::Poll::Ready(Err(ChannelClosedError));
        }
        if inner.queue.len() < inner.capacity {
            if let Some(value) = this.value.take() {
                inner.queue.push_back(value);
                if let Some(waker) = inner.recv_waker.take() {
                    waker.wake();
                }
                return std::task::Poll::Ready(Ok(()));
            }
        }
        inner.send_wakers.push_back(cx.waker().clone());
        std::task::Poll::Pending
    }
}

impl<T> MpscReceiver<T> {
    /// Try to receive a value without blocking.
    pub fn try_recv(&self) -> Option<T> {
        let mut inner = self.inner.lock().unwrap();
        let value = inner.queue.pop_front();
        if value.is_some() {
            // Wake a sender that might be waiting.
            if let Some(waker) = inner.send_wakers.pop_front() {
                waker.wake();
            }
        }
        value
    }

    /// Receive a value, returning a future.
    pub fn recv(&self) -> MpscRecvFuture<'_, T> {
        MpscRecvFuture { receiver: self }
    }

    /// Close the receiving end.
    pub fn close(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        for waker in inner.send_wakers.drain(..) {
            waker.wake();
        }
    }
}

/// Future returned by [`MpscReceiver::recv`].
pub struct MpscRecvFuture<'a, T> {
    receiver: &'a MpscReceiver<T>,
}

impl<'a, T> std::future::Future for MpscRecvFuture<'a, T> {
    type Output = Option<T>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let mut inner = self.receiver.inner.lock().unwrap();
        if let Some(value) = inner.queue.pop_front() {
            if let Some(waker) = inner.send_wakers.pop_front() {
                waker.wake();
            }
            std::task::Poll::Ready(Some(value))
        } else if inner.closed {
            std::task::Poll::Ready(None)
        } else {
            inner.recv_waker = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }
}

impl<T> Drop for MpscReceiver<T> {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        for waker in inner.send_wakers.drain(..) {
            waker.wake();
        }
    }
}

// ---------------------------------------------------------------------------
// Join Combinator
// ---------------------------------------------------------------------------

/// Future that polls two futures concurrently and resolves when both complete.
pub struct Join<A, B>
where
    A: std::future::Future,
    B: std::future::Future,
{
    a: Option<Pin<Box<A>>>,
    b: Option<Pin<Box<B>>>,
    result_a: Option<A::Output>,
    result_b: Option<B::Output>,
}
impl<A: std::future::Future, B: std::future::Future> Unpin for Join<A, B> {}

/// Join two futures, returning both results when complete.
pub fn join<A, B>(a: A, b: B) -> Join<A, B>
where
    A: std::future::Future,
    B: std::future::Future,
{
    Join {
        a: Some(Box::pin(a)),
        b: Some(Box::pin(b)),
        result_a: None,
        result_b: None,
    }
}

impl<A, B> std::future::Future for Join<A, B>
where
    A: std::future::Future,
    B: std::future::Future,
{
    type Output = (A::Output, B::Output);

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // Poll A if not yet done.
        if self.result_a.is_none() {
            if let Some(fut) = self.a.as_mut() {
                if let std::task::Poll::Ready(val) = fut.as_mut().poll(cx) {
                    self.result_a = Some(val);
                    self.a = None;
                }
            }
        }
        // Poll B if not yet done.
        if self.result_b.is_none() {
            if let Some(fut) = self.b.as_mut() {
                if let std::task::Poll::Ready(val) = fut.as_mut().poll(cx) {
                    self.result_b = Some(val);
                    self.b = None;
                }
            }
        }
        // Both done?
        if self.result_a.is_some() && self.result_b.is_some() {
            let a = self.result_a.take().unwrap();
            let b = self.result_b.take().unwrap();
            std::task::Poll::Ready((a, b))
        } else {
            std::task::Poll::Pending
        }
    }
}

/// Join three futures concurrently.
pub struct Join3<A, B, C>
where
    A: std::future::Future,
    B: std::future::Future,
    C: std::future::Future,
{
    a: Option<Pin<Box<A>>>,
    b: Option<Pin<Box<B>>>,
    c: Option<Pin<Box<C>>>,
    result_a: Option<A::Output>,
    result_b: Option<B::Output>,
    result_c: Option<C::Output>,
}
impl<A: std::future::Future, B: std::future::Future, C: std::future::Future> Unpin for Join3<A, B, C> {}

/// Join three futures, returning all results when every future completes.
pub fn join3<A, B, C>(a: A, b: B, c: C) -> Join3<A, B, C>
where
    A: std::future::Future,
    B: std::future::Future,
    C: std::future::Future,
{
    Join3 {
        a: Some(Box::pin(a)),
        b: Some(Box::pin(b)),
        c: Some(Box::pin(c)),
        result_a: None,
        result_b: None,
        result_c: None,
    }
}

impl<A, B, C> std::future::Future for Join3<A, B, C>
where
    A: std::future::Future,
    B: std::future::Future,
    C: std::future::Future,
{
    type Output = (A::Output, B::Output, C::Output);

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.result_a.is_none() {
            if let Some(fut) = self.a.as_mut() {
                if let std::task::Poll::Ready(val) = fut.as_mut().poll(cx) {
                    self.result_a = Some(val);
                    self.a = None;
                }
            }
        }
        if self.result_b.is_none() {
            if let Some(fut) = self.b.as_mut() {
                if let std::task::Poll::Ready(val) = fut.as_mut().poll(cx) {
                    self.result_b = Some(val);
                    self.b = None;
                }
            }
        }
        if self.result_c.is_none() {
            if let Some(fut) = self.c.as_mut() {
                if let std::task::Poll::Ready(val) = fut.as_mut().poll(cx) {
                    self.result_c = Some(val);
                    self.c = None;
                }
            }
        }
        if self.result_a.is_some() && self.result_b.is_some() && self.result_c.is_some() {
            let a = self.result_a.take().unwrap();
            let b = self.result_b.take().unwrap();
            let c = self.result_c.take().unwrap();
            std::task::Poll::Ready((a, b, c))
        } else {
            std::task::Poll::Pending
        }
    }
}

// ---------------------------------------------------------------------------
// Select Combinator
// ---------------------------------------------------------------------------

/// Outcome of a [`select`] operation.
pub enum SelectResult<A, B> {
    /// The first future completed first.
    First(A),
    /// The second future completed first.
    Second(B),
}

/// Future that polls two futures and resolves as soon as either one completes.
pub struct Select<A, B>
where
    A: std::future::Future,
    B: std::future::Future,
{
    a: Option<Pin<Box<A>>>,
    b: Option<Pin<Box<B>>>,
}

/// Race two futures against each other, returning the result of whichever
/// finishes first.
pub fn select<A, B>(a: A, b: B) -> Select<A, B>
where
    A: std::future::Future,
    B: std::future::Future,
{
    Select {
        a: Some(Box::pin(a)),
        b: Some(Box::pin(b)),
    }
}

impl<A, B> std::future::Future for Select<A, B>
where
    A: std::future::Future,
    B: std::future::Future,
{
    type Output = SelectResult<A::Output, B::Output>;

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // Poll A first.
        if let Some(fut) = self.a.as_mut() {
            if let std::task::Poll::Ready(val) = fut.as_mut().poll(cx) {
                self.a = None;
                return std::task::Poll::Ready(SelectResult::First(val));
            }
        }
        // Then B.
        if let Some(fut) = self.b.as_mut() {
            if let std::task::Poll::Ready(val) = fut.as_mut().poll(cx) {
                self.b = None;
                return std::task::Poll::Ready(SelectResult::Second(val));
            }
        }
        std::task::Poll::Pending
    }
}

// ---------------------------------------------------------------------------
// JoinAll Combinator
// ---------------------------------------------------------------------------

/// A future that waits for all futures in a vector to complete.
pub struct JoinAll<F: std::future::Future> {
    futures: Vec<Option<Pin<Box<F>>>>,
    results: Vec<Option<F::Output>>,
}
impl<F: std::future::Future> Unpin for JoinAll<F> {}

/// Join a collection of futures, returning all results.
pub fn join_all<F: std::future::Future>(futures: Vec<F>) -> JoinAll<F> {
    let len = futures.len();
    JoinAll {
        futures: futures.into_iter().map(|f| Some(Box::pin(f))).collect(),
        results: (0..len).map(|_| None).collect(),
    }
}

impl<F: std::future::Future> std::future::Future for JoinAll<F> {
    type Output = Vec<F::Output>;

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let mut all_done = true;
        for i in 0..self.futures.len() {
            if self.results[i].is_some() {
                continue;
            }
            if let Some(fut) = self.futures[i].as_mut() {
                match fut.as_mut().poll(cx) {
                    std::task::Poll::Ready(val) => {
                        self.results[i] = Some(val);
                        self.futures[i] = None;
                    }
                    std::task::Poll::Pending => {
                        all_done = false;
                    }
                }
            }
        }
        if all_done {
            let results: Vec<F::Output> = self
                .results
                .iter_mut()
                .map(|r| r.take().unwrap())
                .collect();
            std::task::Poll::Ready(results)
        } else {
            std::task::Poll::Pending
        }
    }
}

// ---------------------------------------------------------------------------
// Async File I/O Stubs
// ---------------------------------------------------------------------------

/// Async file I/O errors.
#[derive(Debug, Clone)]
pub enum AsyncIoError {
    /// File not found at the given path.
    NotFound(String),
    /// Insufficient permissions.
    PermissionDenied(String),
    /// Generic I/O error with description.
    IoError(String),
    /// The operation was cancelled.
    Cancelled,
    /// The operation timed out.
    TimedOut,
}

impl fmt::Display for AsyncIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsyncIoError::NotFound(path) => write!(f, "file not found: {}", path),
            AsyncIoError::PermissionDenied(path) => {
                write!(f, "permission denied: {}", path)
            }
            AsyncIoError::IoError(msg) => write!(f, "I/O error: {}", msg),
            AsyncIoError::Cancelled => write!(f, "operation cancelled"),
            AsyncIoError::TimedOut => write!(f, "operation timed out"),
        }
    }
}

/// Result type for async I/O operations.
pub type AsyncIoResult<T> = Result<T, AsyncIoError>;

/// Stub: asynchronously read an entire file into a byte vector.
///
/// In a real implementation this would dispatch to a thread pool.
/// Here we simulate an async boundary by yielding once.
pub struct ReadFileFuture {
    path: String,
    yielded: bool,
}

impl ReadFileFuture {
    pub fn new(path: String) -> Self {
        Self {
            path,
            yielded: false,
        }
    }
}

impl std::future::Future for ReadFileFuture {
    type Output = AsyncIoResult<Vec<u8>>;

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if !self.yielded {
            self.yielded = true;
            cx.waker().wake_by_ref();
            return std::task::Poll::Pending;
        }
        // Simulate a blocking read on first poll after yield.
        match std::fs::read(&self.path) {
            Ok(data) => std::task::Poll::Ready(Ok(data)),
            Err(e) => {
                let err = match e.kind() {
                    std::io::ErrorKind::NotFound => AsyncIoError::NotFound(self.path.clone()),
                    std::io::ErrorKind::PermissionDenied => {
                        AsyncIoError::PermissionDenied(self.path.clone())
                    }
                    _ => AsyncIoError::IoError(e.to_string()),
                };
                std::task::Poll::Ready(Err(err))
            }
        }
    }
}

/// Convenience: read a file asynchronously.
pub fn read_file(path: &str) -> ReadFileFuture {
    ReadFileFuture::new(path.to_owned())
}

/// Stub: asynchronously write data to a file.
pub struct WriteFileFuture {
    path: String,
    data: Vec<u8>,
    yielded: bool,
}

impl WriteFileFuture {
    pub fn new(path: String, data: Vec<u8>) -> Self {
        Self {
            path,
            data,
            yielded: false,
        }
    }
}

impl std::future::Future for WriteFileFuture {
    type Output = AsyncIoResult<()>;

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if !self.yielded {
            self.yielded = true;
            cx.waker().wake_by_ref();
            return std::task::Poll::Pending;
        }
        match std::fs::write(&self.path, &self.data) {
            Ok(()) => std::task::Poll::Ready(Ok(())),
            Err(e) => {
                let err = match e.kind() {
                    std::io::ErrorKind::PermissionDenied => {
                        AsyncIoError::PermissionDenied(self.path.clone())
                    }
                    _ => AsyncIoError::IoError(e.to_string()),
                };
                std::task::Poll::Ready(Err(err))
            }
        }
    }
}

/// Convenience: write bytes to a file asynchronously.
pub fn write_file(path: &str, data: Vec<u8>) -> WriteFileFuture {
    WriteFileFuture::new(path.to_owned(), data)
}

/// Stub: asynchronously append data to a file.
pub struct AppendFileFuture {
    path: String,
    data: Vec<u8>,
    yielded: bool,
}

impl AppendFileFuture {
    pub fn new(path: String, data: Vec<u8>) -> Self {
        Self {
            path,
            data,
            yielded: false,
        }
    }
}

impl std::future::Future for AppendFileFuture {
    type Output = AsyncIoResult<()>;

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if !self.yielded {
            self.yielded = true;
            cx.waker().wake_by_ref();
            return std::task::Poll::Pending;
        }
        use std::io::Write;
        let result = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)
            .and_then(|mut f| f.write_all(&self.data));
        match result {
            Ok(()) => std::task::Poll::Ready(Ok(())),
            Err(e) => std::task::Poll::Ready(Err(AsyncIoError::IoError(e.to_string()))),
        }
    }
}

/// Convenience: append bytes to a file asynchronously.
pub fn append_file(path: &str, data: Vec<u8>) -> AppendFileFuture {
    AppendFileFuture::new(path.to_owned(), data)
}

// ---------------------------------------------------------------------------
// Executor Configuration
// ---------------------------------------------------------------------------

/// Configuration for the single-threaded executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of tasks that can be spawned concurrently.
    pub max_tasks: usize,
    /// Maximum number of polls per [`Executor::poll_once`] call.
    pub max_polls_per_tick: usize,
    /// Time budget per tick; the executor will stop polling once elapsed.
    pub tick_budget: Duration,
    /// Whether to collect profiling data.
    pub enable_profiling: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_tasks: 4096,
            max_polls_per_tick: 256,
            tick_budget: Duration::from_millis(4),
            enable_profiling: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Executor Statistics
// ---------------------------------------------------------------------------

/// Runtime statistics collected by the executor.
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    /// Total number of tasks spawned since creation.
    pub tasks_spawned: u64,
    /// Total number of tasks that ran to completion.
    pub tasks_completed: u64,
    /// Total number of tasks cancelled.
    pub tasks_cancelled: u64,
    /// Total number of polls across all tasks.
    pub total_polls: u64,
    /// Number of polls in the most recent tick.
    pub polls_last_tick: u64,
    /// Time spent polling in the most recent tick.
    pub poll_time_last_tick: Duration,
    /// Currently active (not completed/cancelled) tasks.
    pub active_tasks: usize,
    /// Peak active task count.
    pub peak_active_tasks: usize,
}

impl fmt::Display for ExecutorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Executor Statistics:")?;
        writeln!(f, "  spawned:    {}", self.tasks_spawned)?;
        writeln!(f, "  completed:  {}", self.tasks_completed)?;
        writeln!(f, "  cancelled:  {}", self.tasks_cancelled)?;
        writeln!(f, "  active:     {}", self.active_tasks)?;
        writeln!(f, "  peak:       {}", self.peak_active_tasks)?;
        writeln!(f, "  total polls:{}", self.total_polls)?;
        writeln!(f, "  last tick:  {} polls", self.polls_last_tick)?;
        writeln!(
            f,
            "  last tick time: {:.2}ms",
            self.poll_time_last_tick.as_secs_f64() * 1000.0
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Single-threaded cooperative executor.
///
/// Call [`Executor::poll_once`] each frame from the engine's main loop.
/// Tasks make progress only during that call.
pub struct Executor {
    /// All live tasks, keyed by ID.
    tasks: HashMap<TaskId, Task>,
    /// Queue of task IDs that are ready to be polled.
    ready_queue: Arc<Mutex<VecDeque<TaskId>>>,
    /// Cancel flags keyed by task ID.
    cancel_flags: HashMap<TaskId, Arc<AtomicBool>>,
    /// Completion flags keyed by task ID.
    completed_flags: HashMap<TaskId, Arc<AtomicBool>>,
    /// Timer registrations: deadline -> task IDs to wake.
    timers: BTreeMap<Instant, Vec<TaskId>>,
    /// Configuration.
    config: ExecutorConfig,
    /// Cumulative statistics.
    stats: ExecutorStats,
}

impl Executor {
    /// Create a new executor with default configuration.
    pub fn new() -> Self {
        Self::with_config(ExecutorConfig::default())
    }

    /// Create a new executor with the given configuration.
    pub fn with_config(config: ExecutorConfig) -> Self {
        Self {
            tasks: HashMap::new(),
            ready_queue: Arc::new(Mutex::new(VecDeque::new())),
            cancel_flags: HashMap::new(),
            completed_flags: HashMap::new(),
            timers: BTreeMap::new(),
            config,
            stats: ExecutorStats::default(),
        }
    }

    /// Spawn a `Send` future as a new task.
    pub fn spawn<F>(&mut self, future: F) -> TaskHandle
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        self.spawn_named(future, None)
    }

    /// Spawn a `Send` future with a debug name.
    pub fn spawn_named<F>(&mut self, future: F, name: Option<String>) -> TaskHandle
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let id = alloc_task_id();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let completed_flag = Arc::new(AtomicBool::new(false));

        let task = Task::new(
            id,
            Box::pin(future),
            self.ready_queue.clone(),
            name,
        );

        // Enqueue as ready immediately.
        if let Ok(mut q) = self.ready_queue.lock() {
            q.push_back(id);
        }

        self.tasks.insert(id, task);
        self.cancel_flags.insert(id, cancel_flag.clone());
        self.completed_flags.insert(id, completed_flag.clone());

        self.stats.tasks_spawned += 1;
        self.stats.active_tasks = self.tasks.len();
        if self.stats.active_tasks > self.stats.peak_active_tasks {
            self.stats.peak_active_tasks = self.stats.active_tasks;
        }

        TaskHandle {
            id,
            cancel_flag,
            completed_flag,
        }
    }

    /// Spawn a **non-`Send`** future (must only run on the main thread).
    pub fn spawn_local<F>(&mut self, future: F) -> TaskHandle
    where
        F: std::future::Future<Output = ()> + 'static,
    {
        self.spawn_local_named(future, None)
    }

    /// Spawn a non-`Send` future with a debug name.
    pub fn spawn_local_named<F>(&mut self, future: F, name: Option<String>) -> TaskHandle
    where
        F: std::future::Future<Output = ()> + 'static,
    {
        let id = alloc_task_id();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let completed_flag = Arc::new(AtomicBool::new(false));

        let task = Task::new(
            id,
            Box::pin(future),
            self.ready_queue.clone(),
            name,
        );

        if let Ok(mut q) = self.ready_queue.lock() {
            q.push_back(id);
        }

        self.tasks.insert(id, task);
        self.cancel_flags.insert(id, cancel_flag.clone());
        self.completed_flags.insert(id, completed_flag.clone());

        self.stats.tasks_spawned += 1;
        self.stats.active_tasks = self.tasks.len();
        if self.stats.active_tasks > self.stats.peak_active_tasks {
            self.stats.peak_active_tasks = self.stats.active_tasks;
        }

        TaskHandle {
            id,
            cancel_flag,
            completed_flag,
        }
    }

    /// Register a timer that will wake a task at the given deadline.
    pub fn register_timer(&mut self, task_id: TaskId, deadline: Instant) {
        self.timers.entry(deadline).or_default().push(task_id);
    }

    /// Process expired timers, waking the associated tasks.
    fn process_timers(&mut self) {
        let now = Instant::now();
        let expired: Vec<Instant> = self
            .timers
            .range(..=now)
            .map(|(k, _)| *k)
            .collect();

        for deadline in expired {
            if let Some(task_ids) = self.timers.remove(&deadline) {
                if let Ok(mut q) = self.ready_queue.lock() {
                    for id in task_ids {
                        if self.tasks.contains_key(&id) {
                            q.push_back(id);
                        }
                    }
                }
            }
        }
    }

    /// Poll ready tasks, respecting the per-tick budget and poll count limit.
    ///
    /// Call this once per frame from the engine loop.
    pub fn poll_once(&mut self) {
        let tick_start = Instant::now();
        let mut polls_this_tick: u64 = 0;

        // Process any expired timers first.
        self.process_timers();

        // Drain the ready queue.
        let ready_ids: Vec<TaskId> = {
            let mut q = self.ready_queue.lock().unwrap();
            q.drain(..).collect()
        };

        for id in ready_ids {
            // Budget checks.
            if polls_this_tick >= self.config.max_polls_per_tick as u64 {
                // Put remaining back in the queue.
                break;
            }
            if tick_start.elapsed() >= self.config.tick_budget {
                break;
            }

            // Check for cancellation.
            if let Some(flag) = self.cancel_flags.get(&id) {
                if flag.load(Ordering::SeqCst) {
                    // Remove the task.
                    self.tasks.remove(&id);
                    self.cancel_flags.remove(&id);
                    if let Some(cf) = self.completed_flags.get(&id) {
                        cf.store(true, Ordering::SeqCst);
                    }
                    self.completed_flags.remove(&id);
                    self.stats.tasks_cancelled += 1;
                    self.stats.active_tasks = self.tasks.len();
                    continue;
                }
            }

            // Get the task.
            let task = match self.tasks.get_mut(&id) {
                Some(t) => t,
                None => continue,
            };

            task.state = TaskState::Running;
            task.poll_count += 1;
            polls_this_tick += 1;

            // Build a waker and poll.
            let waker = into_std_waker(task.waker.clone());
            let mut cx = std::task::Context::from_waker(&waker);

            let poll_start = Instant::now();
            let result = task.future.as_mut().poll(&mut cx);
            let poll_elapsed = poll_start.elapsed();
            task.total_poll_time += poll_elapsed;

            match result {
                std::task::Poll::Ready(()) => {
                    task.state = TaskState::Completed;
                    if let Some(cf) = self.completed_flags.get(&id) {
                        cf.store(true, Ordering::SeqCst);
                    }
                    self.tasks.remove(&id);
                    self.cancel_flags.remove(&id);
                    self.completed_flags.remove(&id);
                    self.stats.tasks_completed += 1;
                }
                std::task::Poll::Pending => {
                    if let Some(t) = self.tasks.get_mut(&id) {
                        t.state = TaskState::Pending;
                        t.waker.reset();
                    }
                }
            }
        }

        self.stats.total_polls += polls_this_tick;
        self.stats.polls_last_tick = polls_this_tick;
        self.stats.poll_time_last_tick = tick_start.elapsed();
        self.stats.active_tasks = self.tasks.len();
    }

    /// Run the executor until all tasks complete or the timeout expires.
    ///
    /// This is useful for testing; in production, prefer `poll_once` per frame.
    pub fn run_until_complete(&mut self, timeout: Duration) {
        let start = Instant::now();
        while !self.tasks.is_empty() && start.elapsed() < timeout {
            self.poll_once();
        }
    }

    /// Returns a snapshot of the current executor statistics.
    pub fn stats(&self) -> &ExecutorStats {
        &self.stats
    }

    /// Number of currently active tasks.
    pub fn active_task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Returns `true` if no tasks are active.
    pub fn is_idle(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Cancel all active tasks.
    pub fn cancel_all(&mut self) {
        let ids: Vec<TaskId> = self.tasks.keys().copied().collect();
        for id in ids {
            if let Some(flag) = self.cancel_flags.get(&id) {
                flag.store(true, Ordering::SeqCst);
            }
        }
        // Process cancellations on next poll.
    }

    /// Remove completed/cancelled task data, freeing memory.
    pub fn gc(&mut self) {
        self.cancel_flags.retain(|id, _| self.tasks.contains_key(id));
        self.completed_flags.retain(|id, _| self.tasks.contains_key(id));
    }

    /// Returns a list of (TaskId, name, state, poll_count) for active tasks.
    pub fn task_info(&self) -> Vec<(TaskId, Option<String>, TaskState, u64)> {
        self.tasks
            .values()
            .map(|t| (t.id, t.name.clone(), t.state, t.poll_count))
            .collect()
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Executor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Executor")
            .field("active_tasks", &self.tasks.len())
            .field("stats", &self.stats)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_id_uniqueness() {
        let a = alloc_task_id();
        let b = alloc_task_id();
        assert_ne!(a, b);
    }

    #[test]
    fn test_yield_now_completes() {
        let mut exec = Executor::new();
        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = flag.clone();
        exec.spawn(async move {
            yield_now().await;
            flag2.store(true, Ordering::SeqCst);
        });
        exec.run_until_complete(Duration::from_secs(1));
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn test_oneshot_channel() {
        let mut exec = Executor::new();
        let (tx, rx) = oneshot::<u32>();
        let result = Arc::new(Mutex::new(None));
        let result2 = result.clone();
        exec.spawn(async move {
            let val = rx.await.unwrap();
            *result2.lock().unwrap() = Some(val);
        });
        exec.spawn(async move {
            yield_now().await;
            tx.send(42).unwrap();
        });
        exec.run_until_complete(Duration::from_secs(1));
        assert_eq!(*result.lock().unwrap(), Some(42));
    }

    #[test]
    fn test_mpsc_channel() {
        let mut exec = Executor::new();
        let (tx, rx) = mpsc_channel::<i32>(4);
        let collected = Arc::new(Mutex::new(Vec::new()));
        let collected2 = collected.clone();

        exec.spawn(async move {
            tx.try_send(1).unwrap();
            tx.try_send(2).unwrap();
            tx.try_send(3).unwrap();
            tx.close();
        });
        exec.spawn(async move {
            loop {
                match rx.recv().await {
                    Some(val) => collected2.lock().unwrap().push(val),
                    None => break,
                }
            }
        });
        exec.run_until_complete(Duration::from_secs(1));
        assert_eq!(*collected.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_cancel_task() {
        let mut exec = Executor::new();
        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = flag.clone();
        let handle = exec.spawn(async move {
            loop {
                yield_now().await;
            }
        });
        // Poll once to start the task.
        exec.poll_once();
        assert!(!handle.is_completed());
        // Cancel it.
        handle.cancel();
        exec.poll_once();
        assert_eq!(exec.stats().tasks_cancelled, 1);
    }

    #[test]
    fn test_join_combinator() {
        let mut exec = Executor::new();
        let result = Arc::new(Mutex::new((0u32, 0u32)));
        let result2 = result.clone();
        exec.spawn(async move {
            let (a, b) = join(
                async { 10u32 },
                async { 20u32 },
            ).await;
            *result2.lock().unwrap() = (a, b);
        });
        exec.run_until_complete(Duration::from_secs(1));
        assert_eq!(*result.lock().unwrap(), (10, 20));
    }

    #[test]
    fn test_executor_stats() {
        let mut exec = Executor::new();
        exec.spawn(async { yield_now().await; });
        exec.spawn(async { yield_now().await; });
        exec.run_until_complete(Duration::from_secs(1));
        assert_eq!(exec.stats().tasks_spawned, 2);
        assert_eq!(exec.stats().tasks_completed, 2);
    }

    #[test]
    fn test_spawn_local() {
        let mut exec = Executor::new();
        let counter = std::rc::Rc::new(Cell::new(0u32));
        let counter2 = counter.clone();
        exec.spawn_local(async move {
            counter2.set(counter2.get() + 1);
            yield_now().await;
            counter2.set(counter2.get() + 1);
        });
        exec.run_until_complete(Duration::from_secs(1));
        assert_eq!(counter.get(), 2);
    }
}
