# mock-func

Tiny, ergonomic function mocks for Rust unit tests.

This crate provides:

- `MockFunc<Input, Output>`: non-thread-safe mock for single-threaded tests.
- `ThreadSafeMockFunc<Input, Output>`: synchronized mock for concurrent tests.
- `AsyncMockFunc<Input, Output>`: non-thread-safe async mock.
- `ThreadSafeAsyncMockFunc<Input, Output>`: synchronized async mock.

Key capabilities:

- Predicate assertions: `called_times`, `called_with`, `called_with_times`.
- Sequential scripting: `then_returns`, `returns_sequence`.
- Result-oriented helpers: `succeeds`, `fails`, `then_succeeds`, `then_fails`.

## Installation

```toml
[dependencies]
mock-func = "0.1"
```

## Quick Start

```rust
use mock_func::MockFunc;

let mut fetch = MockFunc::with_return(200);
fetch.when_called( | path: & & str| {
assert ! (path.starts_with('/'));
});

let status = fetch.call_and_return("/health");
assert_eq!(status, 200);
assert_eq!(fetch.count(), 1);
assert_eq!(fetch.input(), Some(&"/health"));
```

## Predicate Assertions

```rust
use mock_func::MockFunc;

let mut mock = MockFunc::with_return(());
let _ = mock.call_and_return("a");
let _ = mock.call_and_return("b");
let _ = mock.call_and_return("b");

assert!(mock.called_times(3));
assert!(mock.called_with(|value| *value == "a"));
assert_eq!(mock.called_with_times(|value| *value == "b"), 2);
```

## Sequential Return/Fail Scripting

```rust
use mock_func::MockFunc;

let mut mock: MockFunc<(), Result<i32, & 'static str> > = MockFunc::with_return(Ok(100));
mock.then_succeeds(1)
.then_fails("boom")
.then_succeeds(3);

assert_eq!(mock.call(), Ok(1));
assert_eq!(mock.call(), Err("boom"));
assert_eq!(mock.call(), Ok(3));
assert_eq!(mock.call(), Ok(100));
```

## Thread-safe Usage

```rust
use mock_func::ThreadSafeMockFunc;
use std::sync::Arc;
use std::thread;

let mock = Arc::new(ThreadSafeMockFunc::with_return(true));
let mut handles = Vec::new();
for i in 0..8 {
let m = Arc::clone( & mock);
handles.push(thread::spawn( move | | m.call_and_return(i)));
}
for h in handles {
assert!(h.join().unwrap());
}
assert_eq!(mock.count(), 8);
```

## Async Usage

```rust
use futures::executor::block_on;
use mock_func::AsyncMockFunc;

let mut mock = AsyncMockFunc::with_return(10usize);
mock.then_returns(1).then_returns(2);

assert_eq!(block_on(mock.call_and_return("one")), 1);
assert_eq!(block_on(mock.call_and_return("two")), 2);
assert_eq!(block_on(mock.call_and_return("three")), 10);
```

## Design Notes

- Non-thread-safe and thread-safe variants intentionally mirror Swift-style mock semantics.
- No runtime dependencies.
- `unsafe` is forbidden.


## License

Licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT))
