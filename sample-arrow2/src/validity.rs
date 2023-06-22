use arrow2::bitmap::Bitmap;
use sample_std::{Random, Sample};

pub trait GenerateValidity {
    fn generate_validity(&self, g: &mut Random, len: usize) -> Option<Bitmap>;
    fn generate_valid(&self, g: &mut Random) -> bool;
}

#[derive(Debug, Clone)]
pub struct AlwaysValid;

impl GenerateValidity for AlwaysValid {
    fn generate_validity(&self, _: &mut Random, _: usize) -> Option<Bitmap> {
        None
    }

    fn generate_valid(&self, _: &mut Random) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct RandomValidity {
    pub null_chance: f64,
    pub nullable_chance: f64,
}

pub fn generate_validity<V>(null: &Option<V>, g: &mut Random, len: usize) -> Option<Bitmap>
where
    V: Sample<Output = bool>,
{
    null.as_ref().map(|null| {
        Bitmap::from_trusted_len_iter(std::iter::repeat(()).take(len).map(|_| !null.generate(g)))
    })
}

impl GenerateValidity for RandomValidity {
    fn generate_validity(&self, g: &mut Random, len: usize) -> Option<Bitmap> {
        if g.gen_range(0.0..1.0) < self.nullable_chance {
            Some(Bitmap::from_trusted_len_iter(
                std::iter::repeat(())
                    .take(len)
                    .map(|_| self.generate_valid(g)),
            ))
        } else {
            None
        }
    }

    fn generate_valid(&self, g: &mut Random) -> bool {
        g.gen_range(0.0..1.0) > self.null_chance
    }
}
