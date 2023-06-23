//! Some simple helpers for recursive sampling.
use crate::{Random, Sample};
use std::ops::Range;

pub trait Recursion {
    type Output;

    fn recurse(&self, g: &mut Random, inner: RecursiveSampler<Self>) -> Self::Output
    where
        Self: Sized;
}

#[derive(Debug, Clone)]
pub struct RecursiveSampler<G> {
    pub depth: Option<Range<usize>>,
    pub node: G,
}

impl<G: Clone> RecursiveSampler<G> {
    fn lower(&self) -> Self {
        let depth = self
            .depth
            .as_ref()
            .map(|d| d.start.saturating_sub(1)..d.end.saturating_sub(1));
        Self {
            depth,
            ..self.clone()
        }
    }
}

impl<G, N> Sample for RecursiveSampler<G>
where
    G: Recursion<Output = N> + Sample<Output = N> + Clone,
    N: 'static,
{
    type Output = N;

    fn generate(&self, g: &mut Random) -> Self::Output {
        match &self.depth {
            Some(depth) => {
                if depth.start > 0 {
                    self.node.recurse(g, self.lower())
                } else {
                    if depth.end > 0 {
                        self.node.recurse(g, self.lower())
                    } else {
                        self.node.generate(g)
                    }
                }
            }
            None => self.node.recurse(g, self.lower()),
        }
    }
}
