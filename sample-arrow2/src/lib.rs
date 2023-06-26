use arrow2::{array::Array, bitmap::Bitmap};
use sample_std::{Random, Sample};

pub mod array;
pub mod chunk;
pub mod datatypes;
pub mod list;
pub mod primitive;
pub mod struct_;

pub type ArrowSampler = Box<dyn Sample<Output = Box<dyn Array>> + Send + Sync>;

pub(crate) fn generate_validity<V>(null: &Option<V>, g: &mut Random, len: usize) -> Option<Bitmap>
where
    V: Sample<Output = bool>,
{
    null.as_ref().map(|null| {
        Bitmap::from_trusted_len_iter(std::iter::repeat(()).take(len).map(|_| !null.generate(g)))
    })
}
