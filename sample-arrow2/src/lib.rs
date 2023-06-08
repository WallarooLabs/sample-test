use arrow2::array::Array;
use sample_std::Sample;

pub mod array;
pub mod chunk;
pub mod datatypes;
pub mod list;
pub mod primitive;
pub mod struct_;
pub mod validity;

pub use validity::{AlwaysValid, GenerateValidity, RandomValidity};

pub type ArrowSampler = Box<dyn Sample<Output = Box<dyn Array>> + Send + Sync>;
