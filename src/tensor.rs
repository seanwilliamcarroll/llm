type TensorDataUnit = f32;
type TensorData = Vec<TensorDataUnit>;

pub struct Tensor {
    pub data: TensorData,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    #[must_use]
    pub fn new(data: TensorData, shape: Vec<usize>) -> Self {
        let total_data = shape.iter().fold(1, |acc, next| acc * (*next));

        assert_eq!(total_data, data.len());

        let mut strides = vec![1_usize; shape.len()];

        for index in 0..shape.len() {
            strides[index] = shape[(index + 1)..shape.len()]
                .iter()
                .fold(1, |acc, next| acc * (*next));
        }

        Self {
            data,
            shape,
            strides,
        }
    }

    #[must_use]
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(
            vec![0.; shape.iter().fold(1, |acc, next| acc * (*next))],
            shape,
        )
    }

    pub fn arange(lower_bound: usize, upper_bound: usize) -> Self {
        assert!(upper_bound > lower_bound);
        Self::new(
            (lower_bound..upper_bound)
                .map(|elem| elem as TensorDataUnit)
                .collect::<TensorData>(),
            vec![upper_bound - lower_bound],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        assert_eq!(tensor.strides, vec![2, 1]);
    }

    #[test]
    #[should_panic]
    fn test_bad_new() {
        let _ = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 3]);
    }
}
