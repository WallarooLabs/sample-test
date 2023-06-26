use std::collections::HashMap;
use std::ops::Range;

use once_cell::sync::Lazy;
use sample_std::{
    recursive::{Recursion, RecursiveSampler},
    sampler_choice, Random, Regex, Sample, VecSampler,
};
use sample_test::sample_test;

#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Map(HashMap<String, Json>),
}

impl Json {
    fn depth(&self) -> usize {
        match self {
            Json::Array(bs) => bs.iter().map(Json::depth).max().unwrap_or(1),
            _ => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JsonSampler {
    branch: Range<usize>,
}

impl JsonSampler {
    fn string() -> impl Sample<Output = String> {
        Regex::new("[a-z]{20}")
    }

    fn array(&self, inner: JsonTree) -> impl Sample<Output = Vec<Json>> {
        VecSampler {
            length: self.branch.clone(),
            el: inner,
        }
    }
}

impl Sample for JsonSampler {
    type Output = Json;

    fn generate(&self, g: &mut Random) -> Json {
        match g.gen_range(0..=3) {
            0 => Json::Bool(g.arbitrary()),
            1 => Json::Number(g.arbitrary()),
            2 => Json::String(Self::string().generate(g)),
            _ => Json::Null,
        }
    }
}

pub type JsonTree = RecursiveSampler<JsonSampler>;

impl Recursion for JsonSampler {
    type Output = Json;

    fn recurse(&self, g: &mut Random, inner: JsonTree) -> Self::Output {
        match g.gen_range(0..=1) {
            _ => Json::Array(self.array(inner).generate(g)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tree {
    Branch(Vec<Tree>),
    Leaf(usize),
}

impl Tree {
    fn depth(&self) -> usize {
        match self {
            Tree::Branch(bs) => 1 + bs.iter().map(Tree::depth).max().unwrap_or(0),
            Tree::Leaf(..) => 0,
        }
    }

    fn level(self) -> Option<Vec<Tree>> {
        match self {
            Tree::Branch(l) => Some(l),
            Tree::Leaf(_) => None,
        }
    }

    fn leaf(self) -> Option<usize> {
        match self {
            Self::Leaf(v) => Some(v),
            _ => None,
        }
    }
}

pub type TreeSampler = Box<dyn Sample<Output = Tree> + 'static>;

pub fn tree_sampler<LS>(depth: Range<usize>, branch: Range<usize>, leaf: LS) -> TreeSampler
where
    LS: Sample<Output = usize> + Clone + 'static,
{
    let leaf = Box::new(leaf.try_convert(Tree::Leaf, Tree::leaf));
    let mut inner: TreeSampler = leaf.clone();
    for ix in (0..(depth.end - 1)).rev() {
        let el: TreeSampler = if ix > depth.start {
            Box::new(sampler_choice([leaf.clone(), inner]))
        } else {
            inner
        };

        let length = if ix < depth.start - 1 {
            1..depth.end
        } else {
            branch.clone()
        };

        let level = VecSampler { length, el };

        inner = Box::new(level.try_convert(Tree::Branch, Tree::level))
    }

    inner
}

#[sample_test]
fn tree_bounds(#[sample(tree_sampler(2..5, 0..3, 0..100))] tree: Tree) {
    assert!(tree.depth() >= 2);
    assert!(tree.depth() < 5);
}

static JSON: Lazy<JsonTree> = Lazy::new(|| JsonTree {
    depth: Some(0..3),
    node: JsonSampler { branch: 1..10 },
});

#[sample_test]
fn json(#[sample(JSON.clone())] json: Json) {
    assert!(json.depth() < 4);
}
