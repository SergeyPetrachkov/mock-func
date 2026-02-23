//! `mock-func` provides lightweight, ergonomic mocks for Rust unit tests.
//!
//! The crate includes both single-threaded and thread-safe variants, in sync and async forms:
//!
//! - [`MockFunc`] / [`ThreadSafeMockFunc`]
//! - [`AsyncMockFunc`] / [`ThreadSafeAsyncMockFunc`]
//!
//! # Quick start
//!
//! ```
//! use mock_func::MockFunc;
//!
//! let mut mock = MockFunc::with_return(200);
//! let status = mock.call_and_return("/health");
//!
//! assert_eq!(status, 200);
//! assert!(mock.called_once());
//! assert_eq!(mock.input(), Some(&"/health"));
//! ```
//!
//! # Predicate assertions
//!
//! ```
//! use mock_func::MockFunc;
//!
//! let mut mock = MockFunc::with_return(());
//! let _ = mock.call_and_return("a");
//! let _ = mock.call_and_return("b");
//! let _ = mock.call_and_return("b");
//!
//! assert!(mock.called_times(3));
//! assert!(mock.called_with(|value| *value == "a"));
//! assert_eq!(mock.called_with_times(|value| *value == "b"), 2);
//! ```
//!
//! # Sequential result scripting
//!
//! ```
//! use mock_func::MockFunc;
//!
//! let mut mock: MockFunc<(), Result<i32, &'static str>> = MockFunc::with_return(Ok(100));
//! mock.then_succeeds(1)
//!     .then_fails("boom")
//!     .then_succeeds(3);
//!
//! assert_eq!(mock.call(), Ok(1));
//! assert_eq!(mock.call(), Err("boom"));
//! assert_eq!(mock.call(), Ok(3));
//! assert_eq!(mock.call(), Ok(100));
//! ```
//!
//! # Async usage
//!
//! ```no_run
//! use std::sync::Arc;
//! use mock_func::ThreadSafeAsyncMockFunc;
//!
//! let mock = Arc::new(ThreadSafeAsyncMockFunc::<&'static str, bool>::with_return(true));
//! mock.then_returns(false);
//!
//! // Use your preferred runtime/executor in tests.
//! // Example (tokio):
//! // let result = mock.call_and_return("input").await;
//! // assert!(!result);
//! ```
//!
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

type LocalFuture<T> = Pin<Box<dyn Future<Output = T>>>;
type SendFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Non-thread-safe function mock.
///
/// Mirrors Swift `MockFunc` semantics:
/// - records invocations
/// - allows `when_called` callback
/// - supports `returns` / `returns_with`
pub struct MockFunc<Input, Output>
where
    Input: Clone,
    Output: Clone,
{
    invocations: Vec<Input>,
    scripted_outputs: VecDeque<Output>,
    result: Box<dyn Fn(&Input) -> Output>,
    did_call: Option<Box<dyn Fn(&Input)>>,
}

impl<Input, Output> MockFunc<Input, Output>
where
    Input: Clone,
    Output: Clone,
{
    /// Creates a mock with explicit result behavior.
    pub fn new<F>(result: F) -> Self
    where
        F: Fn(&Input) -> Output + 'static,
    {
        Self {
            invocations: Vec::new(),
            scripted_outputs: VecDeque::new(),
            result: Box::new(result),
            did_call: None,
        }
    }

    /// Creates a mock that always returns `value`.
    pub fn with_return(value: Output) -> Self
    where
        Output: 'static,
    {
        Self::new(move |_| value.clone())
    }

    /// Replaces result behavior with a constant value.
    pub fn returns(&mut self, value: Output)
    where
        Output: 'static,
    {
        self.scripted_outputs.clear();
        self.result = Box::new(move |_| value.clone());
    }

    /// Replaces result behavior with a custom function.
    pub fn returns_with<F>(&mut self, result: F)
    where
        F: Fn(&Input) -> Output + 'static,
    {
        self.scripted_outputs.clear();
        self.result = Box::new(result);
    }

    /// Scripts the next returned value in FIFO order.
    pub fn then_returns(&mut self, value: Output) -> &mut Self {
        self.scripted_outputs.push_back(value);
        self
    }

    /// Scripts multiple values to be returned in FIFO order.
    pub fn returns_sequence<I>(&mut self, values: I) -> &mut Self
    where
        I: IntoIterator<Item = Output>,
    {
        self.scripted_outputs.extend(values);
        self
    }

    /// Registers callback invoked whenever mock is called.
    pub fn when_called<F>(&mut self, callback: F)
    where
        F: Fn(&Input) + 'static,
    {
        self.did_call = Some(Box::new(callback));
    }

    /// Simulates invocation and returns configured output.
    pub fn call_and_return(&mut self, input: Input) -> Output {
        self.invocations.push(input.clone());
        if let Some(callback) = &self.did_call {
            callback(&input);
        }
        if let Some(value) = self.scripted_outputs.pop_front() {
            return value;
        }
        (self.result)(&input)
    }

    /// Invocation history.
    pub fn invocations(&self) -> &[Input] {
        &self.invocations
    }

    /// Number of invocations.
    pub fn count(&self) -> usize {
        self.invocations.len()
    }

    /// True if called at least once.
    pub fn called(&self) -> bool {
        !self.invocations.is_empty()
    }

    /// True if called exactly once.
    pub fn called_once(&self) -> bool {
        self.count() == 1
    }

    /// `true` if called exactly `times` times.
    pub fn called_times(&self, times: usize) -> bool {
        self.count() == times
    }

    /// `true` if any invocation matches a predicate.
    pub fn called_with<P>(&self, predicate: P) -> bool
    where
        P: Fn(&Input) -> bool,
    {
        self.invocations.iter().any(predicate)
    }

    /// Number of invocations matching a predicate.
    pub fn called_with_times<P>(&self, predicate: P) -> usize
    where
        P: Fn(&Input) -> bool,
    {
        self.invocations.iter().filter(|value| predicate(value)).count()
    }

    /// Last input if available.
    pub fn input(&self) -> Option<&Input> {
        self.invocations.last()
    }
}

impl<Input, T, E> MockFunc<Input, Result<T, E>>
where
    Input: Clone,
    T: Clone,
    E: Clone,
{
    /// Sets fallback successful value.
    pub fn succeeds(&mut self, value: T)
    where
        T: 'static,
        E: 'static,
    {
        self.returns(Ok(value));
    }

    /// Sets fallback error value.
    pub fn fails(&mut self, error: E)
    where
        T: 'static,
        E: 'static,
    {
        self.returns(Err(error));
    }

    /// Scripts next successful value.
    pub fn then_succeeds(&mut self, value: T) -> &mut Self {
        self.then_returns(Ok(value))
    }

    /// Scripts next error value.
    pub fn then_fails(&mut self, error: E) -> &mut Self {
        self.then_returns(Err(error))
    }
}

impl<Output> MockFunc<(), Output>
where
    Output: Clone,
{
    /// Convenience call for `Input = ()`.
    pub fn call(&mut self) -> Output {
        self.call_and_return(())
    }
}

/// Thread-safe function mock.
///
/// Mirrors Swift `ThreadSafeMockFunc` intent with synchronized state.
pub struct ThreadSafeMockFunc<Input, Output>
where
    Input: Clone + Send + 'static,
    Output: Clone + Send + 'static,
{
    state: Mutex<ThreadSafeMockState<Input, Output>>,
}

struct ThreadSafeMockState<Input, Output>
where
    Input: Clone + Send + 'static,
    Output: Clone + Send + 'static,
{
    invocations: Vec<Input>,
    scripted_outputs: VecDeque<Output>,
    result: Arc<dyn Fn(&Input) -> Output + Send + Sync>,
    did_call: Option<Arc<dyn Fn(&Input) + Send + Sync>>,
}

impl<Input, Output> ThreadSafeMockFunc<Input, Output>
where
    Input: Clone + Send + 'static,
    Output: Clone + Send + 'static,
{
    /// Creates a thread-safe mock with explicit result behavior.
    pub fn new<F>(result: F) -> Self
    where
        F: Fn(&Input) -> Output + Send + Sync + 'static,
    {
        Self {
            state: Mutex::new(ThreadSafeMockState {
                invocations: Vec::new(),
                scripted_outputs: VecDeque::new(),
                result: Arc::new(result),
                did_call: None,
            }),
        }
    }

    /// Creates a thread-safe mock that always returns `value`.
    pub fn with_return(value: Output) -> Self
    where
        Output: Sync,
    {
        Self::new(move |_| value.clone())
    }

    /// Replaces result behavior with a constant value.
    pub fn returns(&self, value: Output)
    where
        Output: Sync,
    {
        let mut state = self.state.lock().expect("ThreadSafeMockFunc poisoned");
        state.scripted_outputs.clear();
        state.result = Arc::new(move |_| value.clone());
    }

    /// Replaces result behavior with a custom function.
    pub fn returns_with<F>(&self, result: F)
    where
        F: Fn(&Input) -> Output + Send + Sync + 'static,
    {
        let mut state = self.state.lock().expect("ThreadSafeMockFunc poisoned");
        state.scripted_outputs.clear();
        state.result = Arc::new(result);
    }

    /// Scripts the next returned value in FIFO order.
    pub fn then_returns(&self, value: Output) {
        let mut state = self.state.lock().expect("ThreadSafeMockFunc poisoned");
        state.scripted_outputs.push_back(value);
    }

    /// Scripts multiple returned values in FIFO order.
    pub fn returns_sequence<I>(&self, values: I)
    where
        I: IntoIterator<Item = Output>,
    {
        let mut state = self.state.lock().expect("ThreadSafeMockFunc poisoned");
        state.scripted_outputs.extend(values);
    }

    /// Registers callback invoked whenever mock is called.
    pub fn when_called<F>(&self, callback: F)
    where
        F: Fn(&Input) + Send + Sync + 'static,
    {
        let mut state = self.state.lock().expect("ThreadSafeMockFunc poisoned");
        state.did_call = Some(Arc::new(callback));
    }

    /// Simulates invocation and returns configured output.
    pub fn call_and_return(&self, input: Input) -> Output {
        let scripted = {
            let mut state = self.state.lock().expect("ThreadSafeMockFunc poisoned");
            state.invocations.push(input.clone());
            if let Some(value) = state.scripted_outputs.pop_front() {
                return value;
            }
            (Arc::clone(&state.result), state.did_call.as_ref().map(Arc::clone))
        };

        if let Some(did_call) = scripted.1 {
            did_call(&input);
        }

        scripted.0(&input)
    }

    /// Invocation history snapshot.
    pub fn invocations(&self) -> Vec<Input> {
        self.state
            .lock()
            .expect("ThreadSafeMockFunc poisoned")
            .invocations
            .clone()
    }

    /// Number of invocations.
    pub fn count(&self) -> usize {
        self.state
            .lock()
            .expect("ThreadSafeMockFunc poisoned")
            .invocations
            .len()
    }

    /// True if called at least once.
    pub fn called(&self) -> bool {
        self.count() > 0
    }

    /// True if called exactly once.
    pub fn called_once(&self) -> bool {
        self.count() == 1
    }

    /// `true` if called exactly `times` times.
    pub fn called_times(&self, times: usize) -> bool {
        self.count() == times
    }

    /// `true` if any invocation matches a predicate.
    pub fn called_with<P>(&self, predicate: P) -> bool
    where
        P: Fn(&Input) -> bool,
    {
        self.state
            .lock()
            .expect("ThreadSafeMockFunc poisoned")
            .invocations
            .iter()
            .any(predicate)
    }

    /// Number of invocations matching a predicate.
    pub fn called_with_times<P>(&self, predicate: P) -> usize
    where
        P: Fn(&Input) -> bool,
    {
        self.state
            .lock()
            .expect("ThreadSafeMockFunc poisoned")
            .invocations
            .iter()
            .filter(|value| predicate(value))
            .count()
    }

    /// Last input if available.
    pub fn input(&self) -> Option<Input> {
        self.state
            .lock()
            .expect("ThreadSafeMockFunc poisoned")
            .invocations
            .last()
            .cloned()
    }
}

impl<Input, T, E> ThreadSafeMockFunc<Input, Result<T, E>>
where
    Input: Clone + Send + 'static,
    T: Clone + Send + 'static,
    E: Clone + Send + 'static,
{
    /// Sets fallback successful value.
    pub fn succeeds(&self, value: T)
    where
        T: Sync,
        E: Sync,
    {
        self.returns(Ok(value));
    }

    /// Sets fallback error value.
    pub fn fails(&self, error: E)
    where
        T: Sync,
        E: Sync,
    {
        self.returns(Err(error));
    }

    /// Scripts next successful value.
    pub fn then_succeeds(&self, value: T) {
        self.then_returns(Ok(value));
    }

    /// Scripts next error value.
    pub fn then_fails(&self, error: E) {
        self.then_returns(Err(error));
    }
}

impl<Output> ThreadSafeMockFunc<(), Output>
where
    Output: Clone + Send + 'static,
{
    /// Convenience call for `Input = ()`.
    pub fn call(&self) -> Output {
        self.call_and_return(())
    }
}

/// Non-thread-safe async function mock.
pub struct AsyncMockFunc<Input, Output>
where
    Input: Clone,
    Output: Clone,
{
    invocations: Vec<Input>,
    scripted_outputs: VecDeque<Output>,
    result: Box<dyn Fn(Input) -> LocalFuture<Output>>,
    did_call: Option<Box<dyn Fn(&Input)>>,
}

impl<Input, Output> AsyncMockFunc<Input, Output>
where
    Input: Clone,
    Output: Clone,
{
    /// Creates an async mock with explicit async result behavior.
    pub fn new<F, Fut>(result: F) -> Self
    where
        F: Fn(Input) -> Fut + 'static,
        Fut: Future<Output = Output> + 'static,
    {
        Self {
            invocations: Vec::new(),
            scripted_outputs: VecDeque::new(),
            result: Box::new(move |input| Box::pin(result(input))),
            did_call: None,
        }
    }

    /// Creates an async mock that always returns `value`.
    pub fn with_return(value: Output) -> Self
    where
        Output: 'static,
    {
        Self::new(move |_| {
            let value = value.clone();
            async move { value }
        })
    }

    /// Replaces async result behavior with a constant value.
    pub fn returns(&mut self, value: Output)
    where
        Output: 'static,
    {
        self.scripted_outputs.clear();
        self.result = Box::new(move |_| {
            let value = value.clone();
            Box::pin(async move { value })
        });
    }

    /// Replaces async result behavior with a custom async function.
    pub fn returns_with<F, Fut>(&mut self, result: F)
    where
        F: Fn(Input) -> Fut + 'static,
        Fut: Future<Output = Output> + 'static,
    {
        self.scripted_outputs.clear();
        self.result = Box::new(move |input| Box::pin(result(input)));
    }

    /// Scripts the next returned value in FIFO order.
    pub fn then_returns(&mut self, value: Output) -> &mut Self {
        self.scripted_outputs.push_back(value);
        self
    }

    /// Scripts multiple returned values in FIFO order.
    pub fn returns_sequence<I>(&mut self, values: I) -> &mut Self
    where
        I: IntoIterator<Item = Output>,
    {
        self.scripted_outputs.extend(values);
        self
    }

    /// Registers callback invoked whenever mock is called.
    pub fn when_called<F>(&mut self, callback: F)
    where
        F: Fn(&Input) + 'static,
    {
        self.did_call = Some(Box::new(callback));
    }

    /// Simulates async invocation and returns configured output.
    pub async fn call_and_return(&mut self, input: Input) -> Output {
        self.invocations.push(input.clone());
        if let Some(callback) = &self.did_call {
            callback(&input);
        }
        if let Some(value) = self.scripted_outputs.pop_front() {
            return value;
        }
        (self.result)(input).await
    }

    /// Invocation history.
    pub fn invocations(&self) -> &[Input] {
        &self.invocations
    }

    /// Number of invocations.
    pub fn count(&self) -> usize {
        self.invocations.len()
    }

    /// True if called at least once.
    pub fn called(&self) -> bool {
        !self.invocations.is_empty()
    }

    /// True if called exactly once.
    pub fn called_once(&self) -> bool {
        self.count() == 1
    }

    /// `true` if called exactly `times` times.
    pub fn called_times(&self, times: usize) -> bool {
        self.count() == times
    }

    /// `true` if any invocation matches a predicate.
    pub fn called_with<P>(&self, predicate: P) -> bool
    where
        P: Fn(&Input) -> bool,
    {
        self.invocations.iter().any(predicate)
    }

    /// Number of invocations matching a predicate.
    pub fn called_with_times<P>(&self, predicate: P) -> usize
    where
        P: Fn(&Input) -> bool,
    {
        self.invocations.iter().filter(|value| predicate(value)).count()
    }

    /// Last input if available.
    pub fn input(&self) -> Option<&Input> {
        self.invocations.last()
    }
}

impl<Output> AsyncMockFunc<(), Output>
where
    Output: Clone,
{
    /// Convenience call for `Input = ()`.
    pub async fn call(&mut self) -> Output {
        self.call_and_return(()).await
    }
}

impl<Input, T, E> AsyncMockFunc<Input, Result<T, E>>
where
    Input: Clone,
    T: Clone,
    E: Clone,
{
    /// Sets fallback successful value.
    pub fn succeeds(&mut self, value: T)
    where
        T: 'static,
        E: 'static,
    {
        self.returns(Ok(value));
    }

    /// Sets fallback error value.
    pub fn fails(&mut self, error: E)
    where
        T: 'static,
        E: 'static,
    {
        self.returns(Err(error));
    }

    /// Scripts next successful value.
    pub fn then_succeeds(&mut self, value: T) -> &mut Self {
        self.then_returns(Ok(value))
    }

    /// Scripts next error value.
    pub fn then_fails(&mut self, error: E) -> &mut Self {
        self.then_returns(Err(error))
    }
}

/// Thread-safe async function mock.
pub struct ThreadSafeAsyncMockFunc<Input, Output>
where
    Input: Clone + Send + 'static,
    Output: Clone + Send + 'static,
{
    state: Mutex<ThreadSafeAsyncMockState<Input, Output>>,
}

struct ThreadSafeAsyncMockState<Input, Output>
where
    Input: Clone + Send + 'static,
    Output: Clone + Send + 'static,
{
    invocations: Vec<Input>,
    scripted_outputs: VecDeque<Output>,
    result: Arc<dyn Fn(Input) -> SendFuture<Output> + Send + Sync>,
    did_call: Option<Arc<dyn Fn(&Input) + Send + Sync>>,
}

impl<Input, Output> ThreadSafeAsyncMockFunc<Input, Output>
where
    Input: Clone + Send + 'static,
    Output: Clone + Send + 'static,
{
    /// Creates a thread-safe async mock with explicit behavior.
    pub fn new<F, Fut>(result: F) -> Self
    where
        F: Fn(Input) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Output> + Send + 'static,
    {
        Self {
            state: Mutex::new(ThreadSafeAsyncMockState {
                invocations: Vec::new(),
                scripted_outputs: VecDeque::new(),
                result: Arc::new(move |input| Box::pin(result(input))),
                did_call: None,
            }),
        }
    }

    /// Creates a thread-safe async mock that always returns `value`.
    pub fn with_return(value: Output) -> Self
    where
        Output: Sync,
    {
        Self::new(move |_| {
            let value = value.clone();
            async move { value }
        })
    }

    /// Replaces async result behavior with a constant value.
    pub fn returns(&self, value: Output)
    where
        Output: Sync,
    {
        let mut state = self
            .state
            .lock()
            .expect("ThreadSafeAsyncMockFunc poisoned");
        state.scripted_outputs.clear();
        state.result = Arc::new(move |_| {
            let value = value.clone();
            Box::pin(async move { value })
        });
    }

    /// Replaces async result behavior with a custom async function.
    pub fn returns_with<F, Fut>(&self, result: F)
    where
        F: Fn(Input) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Output> + Send + 'static,
    {
        let mut state = self
            .state
            .lock()
            .expect("ThreadSafeAsyncMockFunc poisoned");
        state.scripted_outputs.clear();
        state.result = Arc::new(move |input| Box::pin(result(input)));
    }

    /// Scripts the next returned value in FIFO order.
    pub fn then_returns(&self, value: Output) {
        let mut state = self
            .state
            .lock()
            .expect("ThreadSafeAsyncMockFunc poisoned");
        state.scripted_outputs.push_back(value);
    }

    /// Scripts multiple returned values in FIFO order.
    pub fn returns_sequence<I>(&self, values: I)
    where
        I: IntoIterator<Item = Output>,
    {
        let mut state = self
            .state
            .lock()
            .expect("ThreadSafeAsyncMockFunc poisoned");
        state.scripted_outputs.extend(values);
    }

    /// Registers callback invoked whenever mock is called.
    pub fn when_called<F>(&self, callback: F)
    where
        F: Fn(&Input) + Send + Sync + 'static,
    {
        let mut state = self
            .state
            .lock()
            .expect("ThreadSafeAsyncMockFunc poisoned");
        state.did_call = Some(Arc::new(callback));
    }

    /// Simulates async invocation and returns configured output.
    pub async fn call_and_return(&self, input: Input) -> Output {
        let (result, callback, scripted) = {
            let mut state = self
                .state
                .lock()
                .expect("ThreadSafeAsyncMockFunc poisoned");
            state.invocations.push(input.clone());
            let scripted = state.scripted_outputs.pop_front();
            (
                Arc::clone(&state.result),
                state.did_call.as_ref().map(Arc::clone),
                scripted,
            )
        };

        if let Some(did_call) = callback {
            did_call(&input);
        }

        if let Some(value) = scripted {
            return value;
        }

        result(input).await
    }

    /// Invocation history snapshot.
    pub fn invocations(&self) -> Vec<Input> {
        self.state
            .lock()
            .expect("ThreadSafeAsyncMockFunc poisoned")
            .invocations
            .clone()
    }

    /// Number of invocations.
    pub fn count(&self) -> usize {
        self.state
            .lock()
            .expect("ThreadSafeAsyncMockFunc poisoned")
            .invocations
            .len()
    }

    /// True if called at least once.
    pub fn called(&self) -> bool {
        self.count() > 0
    }

    /// True if called exactly once.
    pub fn called_once(&self) -> bool {
        self.count() == 1
    }

    /// `true` if called exactly `times` times.
    pub fn called_times(&self, times: usize) -> bool {
        self.count() == times
    }

    /// `true` if any invocation matches a predicate.
    pub fn called_with<P>(&self, predicate: P) -> bool
    where
        P: Fn(&Input) -> bool,
    {
        self.state
            .lock()
            .expect("ThreadSafeAsyncMockFunc poisoned")
            .invocations
            .iter()
            .any(predicate)
    }

    /// Number of invocations matching a predicate.
    pub fn called_with_times<P>(&self, predicate: P) -> usize
    where
        P: Fn(&Input) -> bool,
    {
        self.state
            .lock()
            .expect("ThreadSafeAsyncMockFunc poisoned")
            .invocations
            .iter()
            .filter(|value| predicate(value))
            .count()
    }

    /// Last input if available.
    pub fn input(&self) -> Option<Input> {
        self.state
            .lock()
            .expect("ThreadSafeAsyncMockFunc poisoned")
            .invocations
            .last()
            .cloned()
    }
}

impl<Output> ThreadSafeAsyncMockFunc<(), Output>
where
    Output: Clone + Send + 'static,
{
    /// Convenience call for `Input = ()`.
    pub async fn call(&self) -> Output {
        self.call_and_return(()).await
    }
}

impl<Input, T, E> ThreadSafeAsyncMockFunc<Input, Result<T, E>>
where
    Input: Clone + Send + 'static,
    T: Clone + Send + 'static,
    E: Clone + Send + 'static,
{
    /// Sets fallback successful value.
    pub fn succeeds(&self, value: T)
    where
        T: Sync,
        E: Sync,
    {
        self.returns(Ok(value));
    }

    /// Sets fallback error value.
    pub fn fails(&self, error: E)
    where
        T: Sync,
        E: Sync,
    {
        self.returns(Err(error));
    }

    /// Scripts next successful value.
    pub fn then_succeeds(&self, value: T) {
        self.then_returns(Ok(value));
    }

    /// Scripts next error value.
    pub fn then_fails(&self, error: E) {
        self.then_returns(Err(error));
    }
}

#[cfg(test)]
mod tests {
    use super::{AsyncMockFunc, MockFunc, ThreadSafeAsyncMockFunc, ThreadSafeMockFunc};
    use futures::executor::block_on;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn records_invocations_and_returns_value() {
        let mut mock = MockFunc::with_return(42);

        let result = mock.call_and_return("abc");

        assert_eq!(result, 42);
        assert_eq!(mock.count(), 1);
        assert!(mock.called());
        assert!(mock.called_once());
        assert_eq!(mock.input(), Some(&"abc"));
        assert_eq!(mock.invocations(), ["abc"]);
    }

    #[test]
    fn supports_when_called_callback() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let observed_calls = Rc::clone(&calls);

        let mut mock = MockFunc::with_return(true);
        mock.when_called(move |value: &&str| {
            observed_calls.borrow_mut().push((*value).to_string());
        });

        let _ = mock.call_and_return("first");
        let _ = mock.call_and_return("second");

        let actual = calls.borrow().clone();
        assert_eq!(actual, vec!["first".to_string(), "second".to_string()]);
        assert_eq!(mock.count(), 2);
    }

    #[test]
    fn supports_custom_result_function() {
        let mut mock = MockFunc::new(|input: &&str| input.len());

        let result = mock.call_and_return("hello");

        assert_eq!(result, 5);
        assert_eq!(mock.input(), Some(&"hello"));
    }

    #[test]
    fn supports_void_input_call_helper() {
        let mut mock = MockFunc::with_return("done");

        let result = mock.call();

        assert_eq!(result, "done");
        assert_eq!(mock.count(), 1);
    }

    #[test]
    fn predicate_assertions_work() {
        let mut mock = MockFunc::with_return(0);
        let _ = mock.call_and_return(1i32);
        let _ = mock.call_and_return(2i32);
        let _ = mock.call_and_return(2i32);

        assert!(mock.called_times(3));
        assert!(mock.called_with(|value| *value == 1));
        assert_eq!(mock.called_with_times(|value| *value == 2), 2);
    }

    #[test]
    fn sequential_result_scripting_works() {
        let mut mock: MockFunc<(), Result<i32, &'static str>> = MockFunc::with_return(Ok(100));
        mock.then_succeeds(1)
            .then_fails("boom")
            .then_succeeds(3);

        assert_eq!(mock.call(), Ok(1));
        assert_eq!(mock.call(), Err("boom"));
        assert_eq!(mock.call(), Ok(3));
        assert_eq!(mock.call(), Ok(100));
    }

    #[test]
    fn thread_safe_records_concurrent_invocations() {
        let mock = Arc::new(ThreadSafeMockFunc::with_return(1usize));
        let mut handles = Vec::new();

        for idx in 0..16usize {
            let clone = Arc::clone(&mock);
            handles.push(thread::spawn(move || clone.call_and_return(idx)));
        }

        for handle in handles {
            let output = handle.join().expect("thread panicked");
            assert_eq!(output, 1usize);
        }

        assert_eq!(mock.count(), 16);
        assert!(mock.called());
        assert!(!mock.called_once());

        let invocations = mock.invocations();
        assert_eq!(invocations.len(), 16);
    }

    #[test]
    fn thread_safe_supports_when_called_callback() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let observed_clone = Arc::clone(&observed);

        let mock = Arc::new(ThreadSafeMockFunc::with_return(true));
        mock.when_called(move |value: &&str| {
            observed_clone
                .lock()
                .expect("poisoned")
                .push((*value).to_string());
        });

        let m1 = Arc::clone(&mock);
        let m2 = Arc::clone(&mock);
        let first = thread::spawn(move || m1.call_and_return("first"));
        let second = thread::spawn(move || m2.call_and_return("second"));

        assert!(first.join().expect("thread panicked"));
        assert!(second.join().expect("thread panicked"));

        let mut values = observed.lock().expect("poisoned").clone();
        values.sort();
        assert_eq!(values, vec!["first".to_string(), "second".to_string()]);
        assert_eq!(mock.count(), 2);
    }

    #[test]
    fn thread_safe_predicate_assertions_work() {
        let mock = ThreadSafeMockFunc::with_return(0);
        let _ = mock.call_and_return(4i32);
        let _ = mock.call_and_return(7i32);
        let _ = mock.call_and_return(7i32);

        assert!(mock.called_times(3));
        assert!(mock.called_with(|value| *value == 4));
        assert_eq!(mock.called_with_times(|value| *value == 7), 2);
    }

    #[test]
    fn async_mock_supports_sequence_and_predicates() {
        let mut mock = AsyncMockFunc::with_return(10usize);
        mock.then_returns(1).then_returns(2);

        let first = block_on(mock.call_and_return("a"));
        let second = block_on(mock.call_and_return("b"));
        let third = block_on(mock.call_and_return("c"));

        assert_eq!(first, 1);
        assert_eq!(second, 2);
        assert_eq!(third, 10);
        assert!(mock.called_times(3));
        assert_eq!(mock.called_with_times(|value| *value == "b"), 1);
    }

    #[test]
    fn thread_safe_async_mock_supports_sequence() {
        let mock = ThreadSafeAsyncMockFunc::with_return(99usize);
        mock.then_returns(5);
        mock.then_returns(6);

        let first = block_on(mock.call_and_return(1));
        let second = block_on(mock.call_and_return(2));
        let third = block_on(mock.call_and_return(3));

        assert_eq!(first, 5);
        assert_eq!(second, 6);
        assert_eq!(third, 99);
        assert!(mock.called_with(|value| *value == 2));
    }

    #[test]
    fn thread_safe_async_mock_works_from_multiple_threads() {
        let mock = Arc::new(ThreadSafeAsyncMockFunc::with_return(true));
        let mut handles = Vec::new();

        for i in 0..8usize {
            let clone = Arc::clone(&mock);
            handles.push(thread::spawn(move || block_on(clone.call_and_return(i))));
        }

        for handle in handles {
            assert!(handle.join().expect("thread panicked"));
        }

        assert_eq!(mock.count(), 8);
    }
}
