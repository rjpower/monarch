use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::ControlFlow;

use crate::Slice;
use crate::selection::Selection;
use crate::selection::routing::RoutingAction;
use crate::selection::routing::RoutingFrame;
use crate::selection::routing::RoutingFrameKey;
use crate::selection::routing::RoutingStep;

/// Parse an input string to a selection.
pub fn parse(input: &str) -> Selection {
    use nom::combinator::all_consuming;

    use crate::selection::parse::expression;

    let (_, selection) = all_consuming(expression)(input).unwrap();
    selection
}

#[macro_export]
macro_rules! assert_structurally_eq {
    ($expected:expr, $actual:expr) => {{
        let expected = &$expected;
        let actual = &$actual;
        assert!(
            $crate::selection::structurally_equal(expected, actual),
            "Selections do not match.\nExpected: {:#?}\nActual:   {:#?}",
            expected,
            actual,
        );
    }};
}

#[macro_export]
macro_rules! assert_round_trip {
    ($selection:expr) => {{
        let selection: Selection = $selection; // take ownership
        // Convert `Selection` to representation as compact
        // syntax.
        let compact = $crate::selection::pretty::compact(&selection).to_string();
        // Parse a `Selection` from the compact syntax
        // representation.
        let parsed = $crate::selection::test_utils::parse(&compact);
        // Check that the input and parsed `Selection`s are
        // structurally equivalent.
        assert!(
            $crate::selection::structurally_equal(&selection, &parsed),
            "input: {} \n compact: {}\n parsed: {}",
            selection,
            compact,
            parsed
        );
    }};
}

// == Testing (`collect_routed_paths` mesh simulation) ===

/// Message type used in the `collect_routed_paths` mesh routing
/// simulation.
///
/// Each message tracks the current routing state (`frame`) and
/// the full path (`path`) taken from the origin to the current
/// node, represented as a list of flat indices.
///
/// As the message is forwarded, `path` is extended. This allows
/// complete routing paths to be observed at the point of
/// delivery.
pub struct RoutedMessage<T> {
    pub path: Vec<usize>,
    pub frame: RoutingFrame,
    pub _payload: std::marker::PhantomData<T>,
}

impl<T> RoutedMessage<T> {
    pub fn new(path: Vec<usize>, frame: RoutingFrame) -> Self {
        Self {
            path,
            frame,
            _payload: std::marker::PhantomData,
        }
    }
}

/// Simulates routing from the origin through a slice using a
/// `Selection`, collecting all delivery destinations **along with
/// their routing paths**.
//
/// Each returned entry is a tuple `(dst, path)`, where `dst` is the
/// flat index of a delivery node, and `path` is the list of flat
/// indices representing the route taken from the origin to that node.
//
/// Routing begins at `[0, 0, ..., 0]` and proceeds
/// dimension-by-dimension. At each hop, `next_steps` determines the
/// next set of forwarding frames.
//
/// A node is considered a delivery target if:
/// - its `selection` is `Selection::True`, and
/// - it is at the final dimension.
//
///   Useful in tests for verifying full routing paths and ensuring
///   correctness.
pub fn collect_routed_paths(
    selection: &Selection,
    slice: &Slice,
) -> std::collections::HashMap<usize, Vec<usize>> {
    use std::collections::VecDeque;

    let mut pending = VecDeque::new();
    let mut delivered = HashMap::new();
    let mut seen = HashSet::new();

    let root_frame = RoutingFrame::root(selection.clone(), slice.clone());
    let origin = slice.location(&root_frame.here).unwrap();
    pending.push_back(RoutedMessage::<()>::new(vec![origin], root_frame));

    while let Some(RoutedMessage { path, frame, .. }) = pending.pop_front() {
        let mut visitor = |step: RoutingStep| {
            if let RoutingStep::Forward(next_frame) = step {
                let key = RoutingFrameKey::new(&next_frame);
                if seen.insert(key) {
                    let next_rank = slice.location(&next_frame.here).unwrap();
                    let mut next_path = path.clone();
                    next_path.push(next_rank);

                    match next_frame.action() {
                        RoutingAction::Deliver => {
                            delivered.insert(next_rank, next_path);
                        }
                        RoutingAction::Forward => {
                            pending.push_back(RoutedMessage::new(next_path, next_frame));
                        }
                    }
                }
            }
            ControlFlow::Continue(())
        };

        frame.next_steps(
            &mut |_| panic!("Choice encountered in collect_routed_nodes"),
            &mut visitor,
        );
    }

    delivered
}

/// Simulates routing from the origin and returns the set of
/// destination nodes (as flat indices) selected by the
/// `Selection`.
///
/// This function discards routing paths and retains only the
/// final delivery targets. It is useful in tests to compare
/// routing results against selection evaluation.
pub fn collect_routed_nodes(selection: &Selection, slice: &Slice) -> Vec<usize> {
    collect_routed_paths(selection, slice)
        .keys()
        .cloned()
        .collect()
}
