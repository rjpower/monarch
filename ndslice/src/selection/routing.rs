//! # Routing
//!
//! This module defines [`RoutingFrame`] and its [`next_steps`] method,
//! which model how messages propagate through a multidimensional mesh
//! based on a [`Selection`] expression.
//!
//! A [`RoutingFrame`] represents the state of routing at a particular
//! point in the mesh. It tracks the current coordinate (`here`), the
//! remaining selection to apply (`selection`), the mesh layout
//! (`slice`), and the current dimension of traversal (`dim`).
//!
//! [`next_steps`] defines a routing-specific evaluation strategy for
//! `Selection`. Unlike [`Selection::eval`], which produces flat
//! indices that match a selection, this method produces intermediate
//! routing states — new frames or deferred steps to continue
//! traversing.
//!
//! Rather than returning raw frames directly, [`next_steps`] produces
//! a list of [`RoutingStep`]s — each representing a distinct kind of
//! routing progression:
//!
//! - [`RoutingStep::Forward`] indicates that routing proceeds
//!   deterministically to a new [`RoutingFrame`] — the next coordinate
//!   is fully determined by the current selection and frame state.
//! - [`RoutingStep::Choice`] represents a deferred decision: it
//!   returns a set of admissible indices, and **the caller must select
//!   one** (e.g., for load balancing or policy-based routing) **before
//!   routing can proceed**.
//!
//! In this way, non-determinism is treated as a **first-class,
//! policy-driven** aspect of the routing system — enabling
//! inspection, customization, and future extensions without
//! complicating the core traversal logic.
//
//! A frame is considered a delivery target if its selection is
//! [`Selection::True`] and all dimensions have been traversed, as
//! determined by [`RoutingFrame::deliver_here`]. All other frames are
//! forwarded further using [`RoutingFrame::should_route`].
//!
//! This design enables **compositional**, **local**, and **scalable**
//! routing:
//! - **Compositional**: complex selection expressions decompose into
//!   simpler, independently evaluated sub-selections.
//! - **Local**: each frame carries exactly the state needed for its
//!   next step — no global coordination or lookahead is required.
//! - **Scalable**: routing unfolds recursively, one hop at a time,
//!   allowing for efficient traversal even in high-dimensional spaces.
//!
//! This module provides the foundation for building structured,
//! recursive routing logic over multidimensional coordinate spaces.
use std::collections::HashSet;
use std::fmt::Write;
use std::hash::Hash;
use std::sync::Arc;

use enum_as_inner::EnumAsInner;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::SliceError;
use crate::selection::NormalizedSelectionKey;
use crate::selection::Selection;
use crate::selection::Slice;

/// Represents the outcome of evaluating a routing step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingAction {
    Deliver,
    Forward,
}

/// `RoutingFrame` captures the state of a selection being evaluated:
/// the current coordinate (`here`), the remaining selection to apply,
/// the shape and layout information (`slice`), and the current dimension (`dim`).
///
/// Each frame represents an independent routing decision and produces
/// zero or more new frames via `next_steps`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingFrame {
    /// The current coordinate in the mesh where this frame is being
    /// evaluated.
    ///
    /// This is the source location for the next routing step.
    pub here: Vec<usize>,

    /// The residual selection expression describing where routing
    /// should continue.
    ///
    /// At each step, only the current dimension (tracked by `dim`) of
    /// this selection is considered.
    pub selection: Selection,

    /// The shape and layout of the full multidimensional space being
    /// routed.
    ///
    /// This determines the bounds and stride information used to
    /// compute coordinates and flat indices.
    pub slice: Arc<Slice>,

    /// The current axis of traversal within the selection and slice.
    ///
    /// Routing proceeds dimension-by-dimension; this value tracks how
    /// many dimensions have already been routed.
    pub dim: usize,
}

// Compile-time check: ensure `RoutingFrame` is thread-safe and fully
// owned.
fn _assert_routing_frame_traits()
where
    RoutingFrame: Send + Sync + Serialize + DeserializeOwned + 'static,
{
}

/// A `RoutingStep` represents a unit of progress in the routing
/// process.
///
/// Returned by [`RoutingFrame::next_steps`], each step describes how
/// routing should proceed from a given frame:
///
/// - [`RoutingStep::Forward`] represents a deterministic hop to the
///   next coordinate in the mesh, with an updated [`RoutingFrame`].
///
/// - [`RoutingStep::Choice`] indicates that routing cannot proceed
///   until the caller selects one of several admissible indices. This
///   allows for policy-driven or non-deterministic routing behavior,
///   such as load balancing.
#[derive(Debug, Clone, EnumAsInner)]
pub enum RoutingStep {
    /// A deterministic routing hop to the next coordinate. Carries an
    /// updated [`RoutingFrame`] describing the new position and
    /// residual selection.
    Forward(RoutingFrame),

    /// A deferred routing decision at the current dimension. Contains
    /// a set of admissible indices and a residual [`RoutingFrame`] to
    /// continue routing once a choice is made.
    Choice(Choice),
}

/// A deferred routing decision as contained in a
/// [`RoutingStep::Choice`].
///
/// A `Choice` contains:
/// - `candidates`: the admissible indices at the current dimension
/// - `frame`: the residual [`RoutingFrame`] describing how routing
///   continues once a choice is made
///
/// To continue routing, the caller must select one of the
/// `candidates` and call [`Choice::choose`] to produce the
/// corresponding [`RoutingStep::Forward`].
#[derive(Debug, Clone)]
pub struct Choice {
    pub(crate) candidates: Vec<usize>,
    pub(crate) frame: RoutingFrame,
}

impl Choice {
    /// Returns the list of admissible indices at the current
    /// dimension.
    ///
    /// These represent the valid choices that the caller can select
    /// from when resolving this deferred routing step.
    pub fn candidates(&self) -> &[usize] {
        &self.candidates
    }

    /// Returns a reference to the residual [`RoutingFrame`]
    /// associated with this choice.
    ///
    /// This frame encodes the selection and mesh context to be used
    /// once a choice is made, and routing continues at the next
    /// dimension.
    pub fn frame(&self) -> &RoutingFrame {
        &self.frame
    }

    /// Resolves the choice by selecting a specific index.
    ///
    /// Constrains the residual selection to the chosen index at the
    /// current dimension and returns a [`RoutingStep::Forward`] for
    /// continued routing.
    pub fn choose(self, index: usize) -> RoutingStep {
        // The only thing `next()` has to do is constrain the
        // selection to a concrete choice at the current dimension.
        // `self.frame.selection` is the residual (inner) selection to
        // be applied *past* the current dimension.
        RoutingStep::Forward(RoutingFrame {
            selection: crate::dsl::range(index..=index, self.frame.selection),
            ..self.frame
        })
    }
}

/// Key used to deduplicate routing frames.
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct RoutingFrameKey {
    here: Vec<usize>,
    dim: usize,
    selection: NormalizedSelectionKey,
}

impl RoutingFrameKey {
    /// Constructs a new `RoutingFrameKey` from the given coordinate,
    /// dimension, and normalized selection.
    ///
    /// This is used to uniquely identify a routing frame during
    /// traversal for purposes such as deduplication and memoization.
    pub fn new(here: Vec<usize>, dim: usize, selection: NormalizedSelectionKey) -> Self {
        Self {
            here,
            dim,
            selection,
        }
    }
}

impl RoutingFrame {
    /// Constructs the initial frame at the root coordinate (all
    /// zeros).
    pub fn root(selection: Selection, slice: Slice) -> Self {
        RoutingFrame {
            here: vec![0; slice.num_dim()],
            selection,
            slice: Arc::new(slice),
            dim: 0,
        }
    }

    /// Produces a new frame advanced to the next dimension with
    /// updated position and selection.
    pub fn advance(&self, here: Vec<usize>, selection: Selection) -> Self {
        RoutingFrame {
            here,
            selection,
            slice: Arc::clone(&self.slice),
            dim: self.dim + 1,
        }
    }

    /// Returns a new frame with the same position and dimension but a
    /// different selection.
    pub fn with_selection(&self, selection: Selection) -> Self {
        RoutingFrame {
            here: self.here.clone(),
            selection,
            slice: Arc::clone(&self.slice),
            dim: self.dim,
        }
    }

    /// Determines the appropriate routing action for this frame.
    ///
    /// Returns [`RoutingAction::Deliver`] if the message should be
    /// delivered at this coordinate, or [`RoutingAction::Forward`] if
    /// it should be routed further.
    pub fn action(&self) -> RoutingAction {
        if self.deliver_here() {
            RoutingAction::Deliver
        } else {
            RoutingAction::Forward
        }
    }

    /// Returns the location of this frame in the underlying slice.
    pub fn location(&self) -> Result<usize, SliceError> {
        self.slice.location(&self.here)
    }
}

impl RoutingFrame {
    /// Computes the next routing steps from the current frame.
    ///
    /// Evaluates the selection at the current dimension, returning a
    /// list of new `RoutingFrame`s — each with an updated coordinate,
    /// residual selection, and incremented dimension.
    ///
    /// ### Evaluation Strategy
    ///
    /// - **Selection::True**
    ///   Indicates a match. Produces no further hops — delivery is
    ///   implied when a frame carries `Selection::True` and is at the
    ///   final dimension.
    ///
    /// - **Selection::False**
    ///   No match at this level — returns an empty list.
    ///
    /// - **Selection::All / Selection::Range**
    ///   Generate one frame per value along the current dimension,
    ///   each carrying the inner selection for further recursive routing.
    ///
    /// - **Selection::Union**
    ///   Evaluate both branches independently and concatenate the results —
    ///   each frame continues with the residual from its respective branch.
    ///
    /// - **Selection::Intersection**
    ///   Evaluate both branches independently and intersect the
    ///   resulting frames. Only coordinates present in both branches
    ///   are kept, and their residual selections are intersected at
    ///   the next dimension. It's also important to reduce trivial
    ///   intersections here.
    ///
    /// ### Delivery Semantics
    ///
    /// Message delivery occurs when `deliver_here()` returns true.
    ///
    /// The `dim` field tracks the current axis of traversal. At each
    /// step, only the corresponding dimension of the selection is
    /// evaluated — never any future ones.
    ///
    /// This design ensures routing is recursive, local, and
    /// compositional — each hop carries precisely the information
    /// needed for the next step.
    pub fn next_steps(&self) -> Vec<RoutingStep> {
        let selection = self
            .selection
            .clone()
            .promote_terminal_true(self.dim, self.slice.num_dim());
        match selection {
            Selection::True => vec![],
            Selection::False => vec![],
            Selection::All(inner) => {
                let size = self.slice.sizes()[self.dim];
                (0..size)
                    .map(|i| {
                        let mut coord = self.here.clone();
                        coord[self.dim] = i;
                        RoutingStep::Forward(self.advance(coord, *inner.clone()))
                    })
                    .collect()
            }

            Selection::Range(range, inner) => {
                let size = self.slice.sizes()[self.dim];
                let (min, max, step) = range.resolve(size);
                (min..max)
                    .step_by(step)
                    .map(|i| {
                        let mut coord = self.here.clone();
                        coord[self.dim] = i;
                        RoutingStep::Forward(self.advance(coord, *inner.clone()))
                    })
                    .collect()
            }

            Selection::Any(inner) => {
                let size = self.slice.sizes()[self.dim];
                if size == 0 {
                    return vec![];
                }
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let i = rng.gen_range(0..size);

                let mut coord = self.here.clone();
                coord[self.dim] = i;
                vec![RoutingStep::Forward(self.advance(coord, *inner))]
            }

            Selection::Union(a, b) => {
                let mut left = self.with_selection(*a).next_steps();
                let right = self.with_selection(*b).next_steps();
                left.extend(right);
                left
            }

            Selection::Intersection(a, b) => {
                let mut result = vec![];

                let hops_a = self.with_selection(*a).next_steps();
                let hops_b = self.with_selection(*b).next_steps();

                for step_a in &hops_a {
                    for step_b in &hops_b {
                        match (step_a, step_b) {
                            (RoutingStep::Forward(frame_a), RoutingStep::Forward(frame_b))
                                if frame_a.here == frame_b.here =>
                            {
                                let residual = frame_a
                                    .selection
                                    .clone()
                                    .reduce_intersection(frame_b.selection.clone());
                                result.push(RoutingStep::Forward(
                                    self.advance(frame_a.here.clone(), residual),
                                ));
                            }
                            (RoutingStep::Forward(_), RoutingStep::Forward(_)) => {
                                // Different coordinates. Empty intersection.
                            }
                            (RoutingStep::Choice(_), _) | (_, RoutingStep::Choice(_)) => {
                                unimplemented!(
                                    "choices inside intersections aren't supported at this time"
                                );
                            }
                        }
                    }
                }

                result
            }
            /*
                        // TODO(SF, 2025-04-30): This term is not in the algebra yet.
                        LoadBalanced(inner) => {
                            let size = self.slice.sizes()[self.dim];
                            if size == 0 {
                                vec![]
                            } else {
                                let candidates = (0..size).collect();
                                vec![RoutingStep::Choice(Choice {
                                    candidates,
                                    frame: self.with_selection(*inner),
                                })]
                            }
                        }
            */
            // Catch-all for future combinators (e.g., Label).
            _ => unimplemented!(),
        }
    }

    /// Returns true if this frame represents a terminal delivery
    /// point — i.e., the selection is `True` and all dimensions have
    /// been traversed.
    pub fn deliver_here(&self) -> bool {
        matches!(self.selection, Selection::True) && self.dim == self.slice.num_dim()
    }

    /// Returns true if the message has not yet reached its final
    /// destination and should be forwarded to the next routing step.
    pub fn should_route(&self) -> bool {
        !self.deliver_here()
    }
}

impl RoutingFrame {
    /// Traces the unique routing path to the given destination
    /// coordinate.
    ///
    /// Returns `Some(vec![root, ..., dest])` if `dest` is selected,
    /// or `None` if not.
    pub fn trace_route(&self, dest: &[usize]) -> Option<Vec<Vec<usize>>> {
        use std::collections::HashSet;

        use crate::selection::routing::NormalizedSelectionKey;
        use crate::selection::routing::RoutingFrameKey;

        fn go(
            frame: RoutingFrame,
            dest: &[usize],
            mut path: Vec<Vec<usize>>,
            seen: &mut HashSet<RoutingFrameKey>,
        ) -> Option<Vec<Vec<usize>>> {
            let key = RoutingFrameKey::new(
                frame.here.clone(),
                frame.dim,
                NormalizedSelectionKey::new(&frame.selection),
            );

            if !seen.insert(key) {
                return None;
            }

            path.push(frame.here.clone());

            if frame.deliver_here() && frame.here == dest {
                return Some(path);
            }

            for step in frame.next_steps() {
                let next = step.into_forward().unwrap();
                if let Some(found) = go(next, dest, path.clone(), seen) {
                    return Some(found);
                }
            }

            None
        }

        let mut seen = HashSet::new();
        go(self.clone(), dest, Vec::new(), &mut seen)
    }
}

/// Formats a routing path as a string, showing each hop in order.
///
/// Each line shows the hop index, an arrow (`→` for intermediate
/// hops, `⇨` for the final destination), and the coordinate as a
/// tuple (e.g., `(0, 1)`).
/// # Example
///
/// ```text
///  0 → (0, 0)
///  1 → (0, 1)
///  2 ⇨ (1, 1)
/// ```
#[track_caller]
#[allow(dead_code)]
pub fn format_route(route: &[Vec<usize>]) -> String {
    let mut out = String::new();
    for (i, hop) in route.iter().enumerate() {
        let arrow = if i == route.len() - 1 { "⇨" } else { "→" };
        let coord = format!(
            "({})",
            hop.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(&mut out, "{:>2} {} {}", i, arrow, coord);
    }
    out
}

/// Formats a routing tree as an indented string.
///
/// Traverses the tree of `RoutingFrame`s starting from the root,
/// displaying each step with indentation by dimension. Delivery
/// targets are marked `✅`.
///
/// # Example
/// ```text
/// (0, 0)
///   (0, 1) ✅
/// (1, 0)
///   (1, 1) ✅
/// ```
#[track_caller]
#[allow(dead_code)]
pub fn format_routing_tree(selection: Selection, slice: &Slice) -> String {
    let root = RoutingFrame::root(selection, slice.clone());
    let mut out = String::new();
    let mut seen = HashSet::new();
    format_routing_tree_rec(&root, 0, &mut out, &mut seen).unwrap();
    out
}

fn format_routing_tree_rec(
    frame: &RoutingFrame,
    indent: usize,
    out: &mut String,
    seen: &mut HashSet<RoutingFrameKey>,
) -> std::fmt::Result {
    use crate::selection::routing::NormalizedSelectionKey;
    use crate::selection::routing::RoutingFrameKey;

    let key = RoutingFrameKey::new(
        frame.here.clone(),
        frame.dim,
        NormalizedSelectionKey::new(&frame.selection),
    );

    if !seen.insert(key) {
        return Ok(()); // already visited
    }

    let indent_str = "  ".repeat(indent);
    let coord_str = format!(
        "({})",
        frame
            .here
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );

    match frame.action() {
        RoutingAction::Deliver => {
            writeln!(out, "{}{} ✅", indent_str, coord_str)?;
        }
        RoutingAction::Forward => {
            writeln!(out, "{}{}", indent_str, coord_str)?;
            for step in frame.next_steps() {
                let next = step.into_forward().unwrap();
                format_routing_tree_rec(&next, indent + 1, out, seen)?;
            }
        }
    }

    Ok(())
}

// Pretty-prints a routing path from source to destination.
//
// Each hop is shown as a numbered step with directional arrows.
#[track_caller]
#[allow(dead_code)]
pub fn print_route(route: &[Vec<usize>]) {
    println!("{}", format_route(route));
}

/// Prints the routing tree for a selection over a slice.
///
/// Traverses the routing structure from the root, printing each step
/// with indentation by dimension. Delivery points are marked with
/// `✅`.
#[track_caller]
#[allow(dead_code)]
pub fn print_routing_tree(selection: Selection, slice: &Slice) {
    println!("{}", format_routing_tree(selection, slice));
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::collections::VecDeque;
    use std::marker::PhantomData;

    use super::RoutingFrame;
    use super::RoutingFrameKey;
    use super::print_route;
    use super::print_routing_tree;
    use crate::Slice;
    use crate::selection::EvalOpts;
    use crate::selection::NormalizedSelectionKey;
    use crate::selection::Selection;
    use crate::selection::dsl::*;
    use crate::selection::pretty;
    use crate::selection::routing::RoutingAction;
    use crate::shape;

    // A test slice: (zones = 2, hosts = 4, gpus = 8).
    fn test_slice() -> Slice {
        Slice::new(0usize, vec![2, 4, 8], vec![32, 8, 1]).unwrap()
    }

    // Message type used in the mesh routing simulation.
    //
    // Each message tracks its sender (`from`) and its current routing
    // state (`frame`).
    struct RoutedMessage<T> {
        #[allow(dead_code)] // 'from' isn't read
        from: Vec<usize>,
        frame: RoutingFrame,
        _payload: PhantomData<T>,
    }

    impl<T> RoutedMessage<T> {
        pub fn new(from: Vec<usize>, frame: RoutingFrame) -> Self {
            Self {
                from,
                frame,
                _payload: PhantomData,
            }
        }
    }

    // Simulates message routing from the origin coordinate through a
    // slice, collecting all destination nodes determined by a
    // `Selection`.
    //
    // Starts from coordinate `[0, 0, ..., 0]` and proceeds
    // dimension-by-dimension. At each step, `next_steps` computes the
    // set of `RoutingFrame`s to forward the message to.
    //
    // A frame is considered a delivery target if:
    // - its `selection` is `Selection::True`, and
    // - it is at the final dimension of the slice.
    //
    // Routing continues recursively for all other frames.
    pub fn collect_routed_nodes(selection: Selection, slice: &Slice) -> Vec<usize> {
        let mut pending = VecDeque::new();
        let mut delivered = HashSet::new();
        let mut seen = HashSet::new();

        let root_frame = RoutingFrame::root(selection, slice.clone());
        pending.push_back(RoutedMessage::<()>::new(
            root_frame.here.clone(),
            root_frame,
        ));

        while let Some(RoutedMessage { frame, .. }) = pending.pop_front() {
            for step in frame.next_steps() {
                let next_frame = step.into_forward().unwrap();
                let key = RoutingFrameKey::new(
                    next_frame.here.clone(),
                    next_frame.dim,
                    NormalizedSelectionKey::new(&next_frame.selection),
                );

                if !seen.insert(key) {
                    continue; // already visited this frame
                }

                match next_frame.action() {
                    RoutingAction::Deliver => {
                        delivered.insert(next_frame.slice.location(&next_frame.here).unwrap());
                    }
                    RoutingAction::Forward => {
                        pending.push_back(RoutedMessage::new(frame.here.clone(), next_frame));
                    }
                }
            }
        }

        let mut delivered: Vec<_> = delivered.into_iter().collect();
        delivered.sort();
        delivered
    }

    macro_rules! assert_routing_eq {
        ($slice:expr, $sel:expr) => {{
            let sel = $sel;
            let slice = $slice.clone();
            let mut expected: Vec<_> = sel.eval(&EvalOpts::lenient(), &slice).unwrap().collect();
            expected.sort();
            let mut actual = collect_routed_nodes(sel.clone(), &slice);
            actual.sort();
            assert_eq!(actual, expected, "Mismatch for selection: {}", sel);
        }};
    }

    #[test]
    fn test_routing_04() {
        use crate::selection::dsl::*;

        let slice = test_slice(); // [2, 4, 8], strides [32, 8, 1]

        // Destination: GPU 2 on host 2 in zone 1.
        let dest = vec![1, 2, 2];
        let selection = range(1, range(2, range(2, true_())));
        let root = RoutingFrame::root(selection.clone(), slice.clone());
        let path = root.trace_route(&dest).expect("no route found");
        println!(
            "\ndest: {:?}, (singleton-)selection: ({})\n",
            &dest, &selection
        );
        print_route(&path);
        println!("\n");
        assert_eq!(path.last(), Some(&dest));

        // Destination: "Right back where we started from 🙂".
        let dest = vec![0, 0, 0];
        let selection = range(0, range(0, range(0, true_())));
        let root = RoutingFrame::root(selection.clone(), slice.clone());
        let path = root.trace_route(&dest).expect("no route found");
        println!(
            "\ndest: {:?}, (singleton-)selection: ({})\n",
            &dest, &selection
        );
        print_route(&path);
        println!("\n");
        assert_eq!(path.last(), Some(&dest));
    }

    #[test]
    fn test_routing_05() {
        use crate::selection::dsl::*;

        // "Jun's example" -- a 2 x 2 row major mesh.
        let slice = Slice::new(0usize, vec![2, 2], vec![2, 1]).unwrap();
        // Thats is,
        //  (0, 0)    (0, 1)
        //  (0, 1)    (1, 0)
        //
        // and we want to cast to {(0, 1), (1, 0) and (1, 1)}:
        //
        //  (0, 0)❌    (0, 1)✅
        //  (0, 1)✅    (1, 0)✅
        //
        // One reasonable selection expression describing the
        // destination set.
        let selection = union(range(0, range(1, true_())), range(1, all(true_())));

        // Now print the routing tree.
        print_routing_tree(selection, &slice);

        // Prints:
        // (0, 0)
        //   (0, 0)
        //     (0, 1) ✅
        //   (1, 0)
        //     (1, 0) ✅
        //     (1, 1) ✅

        // Another example: (zones = 2, hosts = 4, gpus = 8).
        let slice = Slice::new(0usize, vec![2, 4, 8], vec![32, 8, 1]).unwrap();
        // Let's have all the odd GPUs on hosts 1, 2 and 3 in zone 0.
        let selection = range(
            0,
            range(1..4, range(shape::Range(1, None, /*step*/ 2), true_())),
        );

        // Now print the routing tree.
        print_routing_tree(selection, &slice);

        // Prints:
        // (0, 0, 0)
        //   (0, 0, 0)
        //     (0, 1, 0)
        //       (0, 1, 1) ✅
        //       (0, 1, 3) ✅
        //       (0, 1, 5) ✅
        //       (0, 1, 7) ✅
        //     (0, 2, 0)
        //       (0, 2, 1) ✅
        //       (0, 2, 3) ✅
        //       (0, 2, 5) ✅
        //       (0, 2, 7) ✅
        //     (0, 3, 0)
        //       (0, 3, 1) ✅
        //       (0, 3, 3) ✅
        //       (0, 3, 5) ✅
        //       (0, 3, 7) ✅
    }

    #[test]
    fn test_routing_00() {
        let slice = test_slice();

        assert_routing_eq!(slice, false_());
        assert_routing_eq!(slice, true_());
        assert_routing_eq!(slice, all(true_()));
        assert_routing_eq!(slice, all(all(true_())));
        assert_routing_eq!(slice, all(all(false_())));
        assert_routing_eq!(slice, all(all(all(true_()))));
        assert_routing_eq!(slice, all(range(0..=0, all(true_()))));
        assert_routing_eq!(slice, all(all(range(0..4, true_()))));
        assert_routing_eq!(slice, all(range(1..=2, all(true_()))));
        assert_routing_eq!(slice, all(all(range(2..6, true_()))));
        assert_routing_eq!(slice, all(all(range(3..=3, true_()))));
        assert_routing_eq!(slice, all(range(1..3, all(true_()))));
        assert_routing_eq!(slice, all(all(range(0..=0, true_()))));
        assert_routing_eq!(slice, range(1..=1, range(3..=3, range(0..=2, true_()))));
        assert_routing_eq!(slice, all(all(range(shape::Range(0, Some(8), 2), true_()))));
        assert_routing_eq!(slice, all(range(shape::Range(1, Some(4), 2), all(true_()))));
    }

    #[test]
    fn test_routing_03() {
        let slice = test_slice();

        assert_routing_eq!(
            slice,
            // sel!(0 & (0,(1|3), *))
            intersection(
                range(0, true_()),
                range(0, union(range(1, all(true_())), range(3, all(true_()))))
            )
        );
        assert_routing_eq!(
            slice,
            // sel!(0 & (0, (3|1), *)),
            intersection(
                range(0, true_()),
                range(0, union(range(3, all(true_())), range(1, all(true_()))))
            )
        );
        assert_routing_eq!(
            slice,
            // sel!((*, *, *) & (*, *, (2 | 4)))
            intersection(
                all(all(all(true_()))),
                all(all(union(range(2, true_()), range(4, true_()))))
            )
        );
        assert_routing_eq!(
            slice,
            // sel!((*, *, *) & (*, *, (4 | 2)))
            intersection(
                all(all(all(true_()))),
                all(all(union(range(4, true_()), range(2, true_()))))
            )
        );
        assert_routing_eq!(
            slice,
            // sel!((*, (1 | 2)) & (*, (2 | 1)))
            intersection(
                all(union(range(1, true_()), range(2, true_()))),
                all(union(range(2, true_()), range(1, true_())))
            )
        );
        assert_routing_eq!(slice, intersection(all(all(all(true_()))), all(true_())));
        assert_routing_eq!(slice, intersection(true_(), all(all(all(true_())))));
        assert_routing_eq!(slice, intersection(all(all(all(true_()))), false_()));
        assert_routing_eq!(slice, intersection(false_(), all(all(all(true_())))));
        assert_routing_eq!(
            slice,
            intersection(
                all(all(range(0..4, true_()))),
                all(all(range(0..4, true_())))
            )
        );
        assert_routing_eq!(
            slice,
            intersection(all(all(range(1, true_()))), all(all(range(2, true_()))))
        );
        assert_routing_eq!(
            slice,
            intersection(all(all(range(2, true_()))), all(all(range(1, true_()))))
        );
        assert_routing_eq!(
            slice,
            intersection(
                all(all(range(1, true_()))),
                intersection(all(all(true_())), all(all(range(1, true_()))))
            )
        );
        assert_routing_eq!(
            slice,
            intersection(
                range(0, true_()),
                range(0, all(union(range(1, true_()), range(3, true_()))))
            )
        );
        assert_routing_eq!(
            slice,
            range(
                0,
                intersection(true_(), all(union(range(1, true_()), range(3, true_()))))
            )
        );
        assert_routing_eq!(
            slice,
            intersection(all(range(1..=2, true_())), all(range(2..=3, true_())))
        );
        assert_routing_eq!(
            slice,
            intersection(
                range(0, true_()),
                intersection(range(0, all(true_())), range(0, range(1, all(true_()))))
            )
        );
        assert_routing_eq!(
            slice,
            intersection(
                range(0, range(1, all(true_()))),
                intersection(range(0, all(true_())), range(0, true_()))
            )
        );
        assert_routing_eq!(
            slice,
            // sel!( (*, *, *) & ((*, *, *) & (*, *, *)) ),
            intersection(
                all(all(all(true_()))),
                intersection(all(all(all(true_()))), all(all(all(true_()))))
            )
        );
        assert_routing_eq!(
            slice,
            union(
                intersection(range(0, true_()), range(0, range(1, all(true_())))),
                range(1, all(all(true_())))
            )
        );
        assert_routing_eq!(
            slice,
            // sel!((1, *, *) | (0 & (0, 3, *)))
            union(
                range(1, all(all(true_()))),
                intersection(range(0, true_()), range(0, range(3, all(true_()))))
            )
        );
        assert_routing_eq!(
            slice,
            intersection(
                union(range(0, true_()), range(1, true_())),
                union(range(1, true_()), range(0, true_()))
            )
        );
        assert_routing_eq!(
            slice,
            union(
                intersection(range(0, range(1, true_())), range(0, range(1, true_()))),
                intersection(range(1, range(3, true_())), range(1, range(3, true_())))
            )
        );
        assert_routing_eq!(
            slice,
            // sel!(*, 8 : 8)
            all(range(8..8, true_()))
        );
        assert_routing_eq!(
            slice,
            // sel!((*, 1) & (*, 8 : 8))
            intersection(all(range(1..2, true_())), all(range(8..8, true_())))
        );
        assert_routing_eq!(
            slice,
            // sel!((*, 8 : 8) | (*, 1))
            union(all(range(8..8, true_())), all(range(1..2, true_())))
        );
        assert_routing_eq!(
            slice,
            // sel!((*, 1) | (*, 2:8))
            union(all(range(1..2, true_())), all(range(2..8, true_())))
        );
        assert_routing_eq!(
            slice,
            // sel!((*, *, *) & (*, *, 2:8))
            intersection(all(all(all(true_()))), all(all(range(2..8, true_()))))
        );
    }

    #[test]
    fn test_routing_02() {
        let slice = test_slice();

        // zone 0 or 1: sel!(0 | 1, *, *)
        assert_routing_eq!(slice, union(range(0, true_()), range(1, true_())));
        assert_routing_eq!(slice, union(range(0, all(true_())), range(1, all(true_()))));
        // hosts 1 and 3 in zone 0: sel!(0, (1 | 3), *)
        assert_routing_eq!(
            slice,
            range(0, union(range(1, all(true_())), range(3, all(true_()))))
        );
        // sel!(0, 1:3 | 5:7, *)
        assert_routing_eq!(
            slice,
            range(
                0,
                union(
                    range(shape::Range(1, Some(3), 1), all(true_())),
                    range(shape::Range(5, Some(7), 1), all(true_()))
                )
            )
        );

        // sel!(* | *): We start with `union(true_(), true_())`.
        //
        // Evaluating the left branch generates routing frames
        // recursively. Evaluating the right branch generates the same
        // frames again.
        //
        // As a result, we produce duplicate `RoutingFrame`s that
        // have:
        // - the same `here` coordinate,
        // - the same dimension (`dim`), and
        // - the same residual selection (`True`).
        //
        // When both frames reach the delivery condition, the second
        // call to `delivered.insert()` returns `false`. If we put an
        // `assert!` on that line this would trigger assertion failure
        // in the routing simulation.
        //
        // TODO: We need memoization to avoid redundant work.
        //
        // This can be achieved without transforming the algebra itself.
        // However, adding normalization will make memoization more
        // effective, so we should plan to implement both.
        //
        // Once that's done, we can safely restore the `assert!`.
        assert_routing_eq!(slice, union(true_(), true_()));
        // sel!(*, *, * | *, *, *)
        assert_routing_eq!(slice, union(all(all(all(true_()))), all(all(all(true_())))));
        // no 'false' support in sel!
        assert_routing_eq!(slice, union(false_(), all(all(all(true_())))));
        assert_routing_eq!(slice, union(all(all(all(true_()))), false_()));
        // sel!(0, 0:4, 0 | 1 | 2)
        assert_routing_eq!(
            slice,
            range(
                0,
                range(
                    shape::Range(0, Some(4), 1),
                    union(
                        range(0, true_()),
                        union(range(1, true_()), range(2, true_()))
                    )
                )
            )
        );
        assert_routing_eq!(
            slice,
            range(
                0,
                union(range(2, range(4, true_())), range(3, range(5, true_())),),
            )
        );
        assert_routing_eq!(
            slice,
            range(0, range(2, union(range(4, true_()), range(5, true_()),),),)
        );
        assert_routing_eq!(
            slice,
            range(
                0,
                union(range(2, range(4, true_())), range(3, range(5, true_())),),
            )
        );
        assert_routing_eq!(
            slice,
            union(
                range(
                    0,
                    union(range(2, range(4, true_())), range(3, range(5, true_())))
                ),
                range(
                    1,
                    union(range(2, range(4, true_())), range(3, range(5, true_())))
                )
            )
        );
    }

    #[test]
    fn test_routing_01() {
        let slice = test_slice();
        let sel = range(0..=0, all(true_()));

        let expected_fanouts: &[&[&[usize]]] = &[
            &[&[0, 0, 0]],
            &[&[0, 0, 0], &[0, 1, 0], &[0, 2, 0], &[0, 3, 0]],
            &[
                &[0, 0, 0],
                &[0, 0, 1],
                &[0, 0, 2],
                &[0, 0, 3],
                &[0, 0, 4],
                &[0, 0, 5],
                &[0, 0, 6],
                &[0, 0, 7],
            ],
            &[
                &[0, 1, 0],
                &[0, 1, 1],
                &[0, 1, 2],
                &[0, 1, 3],
                &[0, 1, 4],
                &[0, 1, 5],
                &[0, 1, 6],
                &[0, 1, 7],
            ],
            &[
                &[0, 2, 0],
                &[0, 2, 1],
                &[0, 2, 2],
                &[0, 2, 3],
                &[0, 2, 4],
                &[0, 2, 5],
                &[0, 2, 6],
                &[0, 2, 7],
            ],
            &[
                &[0, 3, 0],
                &[0, 3, 1],
                &[0, 3, 2],
                &[0, 3, 3],
                &[0, 3, 4],
                &[0, 3, 5],
                &[0, 3, 6],
                &[0, 3, 7],
            ],
        ];

        let expected_deliveries: &[bool] = &[
            false, false, false, false, false, false, // Steps 0–5: no deliveries
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, // Steps 6–37: all deliveries
        ];

        let mut step = 0;
        let mut pending = VecDeque::new();

        pending.push_back(RoutingFrame::root(sel.clone(), slice.clone()));

        println!("Fan-out trace for selection: {}", sel);

        while let Some(frame) = pending.pop_front() {
            let fanout = frame.next_steps();
            let next_coords: Vec<_> = fanout
                .iter()
                .map(|f| f.as_forward().unwrap().here.clone())
                .collect();
            let deliver_here = frame.deliver_here();

            println!(
                "Step {:>2}: from {:?} (flat = {:>2}) | deliver = {} | fan-out count = {} | selection = {:?}",
                step,
                frame.here,
                frame.slice.location(&frame.here).unwrap(),
                deliver_here,
                next_coords.len(),
                format!("{}", pretty(&frame.selection)),
            );

            for next in &next_coords {
                println!("         → {:?}", next);
            }

            if step < expected_fanouts.len() {
                let expected = expected_fanouts[step]
                    .iter()
                    .map(|v| v.to_vec())
                    .collect::<Vec<_>>();
                assert_eq!(
                    next_coords, expected,
                    "Mismatch in next_coords at step {}",
                    step
                );
            }

            if step < expected_deliveries.len() {
                assert_eq!(
                    deliver_here, expected_deliveries[step],
                    "Mismatch in deliver_here at step {} (coord = {:?})",
                    step, frame.here
                );
            }

            for next in fanout {
                pending.push_back(next.into_forward().unwrap());
            }

            step += 1;
        }
    }

    #[test]
    fn test_routing_06() {
        use crate::selection::NormalizedSelectionKey;
        use crate::selection::dsl::*;
        use crate::selection::routing::RoutingFrameKey;

        let slice = test_slice();
        let selection = union(all(true_()), all(true_()));

        let mut pending = VecDeque::new();
        let mut dedup_delivered = Vec::new();
        let mut nodup_delivered = Vec::new();
        let mut seen = HashSet::new();

        let root = RoutingFrame::root(selection.clone(), slice.clone());
        pending.push_back(RoutedMessage::<()>::new(root.here.clone(), root));

        while let Some(RoutedMessage { frame, .. }) = pending.pop_front() {
            for step in frame.next_steps() {
                // Reject choices
                let next = step.into_forward().unwrap();

                // Always record for non-dedup case
                if next.action() == RoutingAction::Deliver {
                    nodup_delivered.push(next.slice.location(&next.here).unwrap());
                }

                // Check if this frame is new (deduplication)
                let key = RoutingFrameKey::new(
                    next.here.clone(),
                    next.dim,
                    NormalizedSelectionKey::new(&next.selection),
                );

                // Only record deliver for dedup case if new
                if seen.insert(key) && next.action() == RoutingAction::Deliver {
                    dedup_delivered.push(next.slice.location(&next.here).unwrap());
                }

                // Always continue traversal for forward hops.
                if next.action() == RoutingAction::Forward {
                    pending.push_back(RoutedMessage::new(frame.here.clone(), next));
                }
            }
        }

        assert_eq!(dedup_delivered.len(), 64);
        assert_eq!(nodup_delivered.len(), 128);
    }

    #[test]
    fn test_routing_07() {
        use crate::selection::dsl::*;
        use crate::selection::routing::RoutingFrame;

        let slice = test_slice(); // shape: [2, 4, 8]

        // Selection: any zone, all hosts, all gpus.
        let selection = any(all(all(true_())));

        let frame = RoutingFrame::root(selection, slice.clone());
        let hops = frame.next_steps();

        // Only one hop should be produced at the `any` dimension.
        assert_eq!(hops.len(), 1);

        // Reject choices.
        let hop = &hops[0].as_forward().unwrap();

        // There should be 3 components to the frame's coordinate.
        assert_eq!(hop.here.len(), 3);

        // The selected zone (dim 0) should be in bounds.
        let zone = hop.here[0];
        assert!(zone < 2, "zone out of bounds: {}", zone);

        // Inner selection should still be All(All(True))
        assert!(matches!(hop.selection, Selection::All(_)));
    }
}
