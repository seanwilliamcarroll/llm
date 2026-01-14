use rayon::prelude::*;

type TensorDataUnit = f32;
type TensorData = Vec<TensorDataUnit>;

pub struct Tensor {
    pub data: TensorData,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    /// # Panics
    /// Will panic on a tensor with mismatch in shape and len of data
    #[must_use]
    pub fn new(data: TensorData, shape: Vec<usize>) -> Self {
        let total_data: usize = shape.iter().product();

        assert_eq!(
            total_data,
            data.len(),
            "Shape does not match length of data!"
        );

        let mut strides = vec![1_usize; shape.len()];

        for index in 0..shape.len() {
            strides[index] = shape[(index + 1)..shape.len()].iter().product();
        }

        Self {
            data,
            shape,
            strides,
        }
    }

    #[must_use]
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(vec![0.; shape.iter().product()], shape)
    }

    /// # Panics
    /// Will panic when `lower_bound` >= `upper_bound`
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn arange(lower_bound: usize, upper_bound: usize) -> Self {
        assert!(upper_bound > lower_bound);
        Self::new(
            (lower_bound..upper_bound)
                .map(|elem| elem as TensorDataUnit)
                .collect::<TensorData>(),
            vec![upper_bound - lower_bound],
        )
    }

    #[must_use]
    pub fn get_2d(&self, row_index: usize, col_index: usize) -> TensorDataUnit {
        self.data[row_index * self.shape[1] + col_index]
    }

    /// # Panics
    /// Will panic if given a non 2D tensor
    fn validate_2d_matrix(&self) {
        assert_eq!(self.shape.len(), 2, "Expected 2D Tensors");
    }

    /// # Panics
    /// Will panic if columns of tensor A do not match rows of tensor B    
    fn validate_2d_matrix_matmul(&self, other: &Self) {
        assert_eq!(
            self.shape[1], other.shape[0],
            "Column length of first matrix must match row length of second"
        );
    }

    /// # Panics
    /// Will panic if given a non 2D tensor
    /// Will panic if columns of tensor A do not match rows of tensor B
    #[must_use]
    pub fn simple_2d_matmul(&self, other: &Self) -> Self {
        // Need to make sure that self's shape (MxN) is compatible
        // with other's shape (NxP)
        self.validate_2d_matrix();
        other.validate_2d_matrix();
        self.validate_2d_matrix_matmul(other);

        let result_row_length = self.shape[0];
        let result_col_length = other.shape[1];
        let mutual_axis_length = self.shape[1];

        let mut result = Self::zeros(vec![result_row_length, result_col_length]);

        // Naively just walk across entire matrices
        for result_row_index in 0..result_row_length {
            for result_col_index in 0..result_col_length {
                let mut sum = 0.0;
                for mutual_index in 0..mutual_axis_length {
                    sum += self.data[result_row_index * mutual_axis_length + mutual_index]
                        * other.data[mutual_index * mutual_axis_length + result_col_index];
                }
                result.data[result_row_index * result_col_length + result_col_index] = sum;
            }
        }
        result
    }

    #[must_use]
    pub fn cache_blocked_2d_matmul(&self, other: &Self) -> Self {
        const BLOCK_SIZE: usize = 8;
        // Need to make sure that self's shape (MxN) is compatible
        // with other's shape (NxP)
        self.validate_2d_matrix();
        other.validate_2d_matrix();
        self.validate_2d_matrix_matmul(other);

        let result_row_length = self.shape[0];
        let result_col_length = other.shape[1];
        let mutual_axis_length = self.shape[1];

        let mut result = Self::zeros(vec![result_row_length, result_col_length]);

        result
            .data
            .par_chunks_mut(mutual_axis_length * BLOCK_SIZE)
            .enumerate()
            .for_each(|(block_index, data)| {
                let row_index_start = block_index * BLOCK_SIZE;
                let row_index_end = (row_index_start + BLOCK_SIZE).min(result_row_length);

                // Each BLOCK_SIZE number of rows processed in parallel

                // Iterate over the columns at BLOCK_SIZE granularity
                for col_index_start in (0..result_col_length).step_by(BLOCK_SIZE) {
                    let col_index_end = (col_index_start + BLOCK_SIZE).min(result_col_length);

                    // Iterate over the mutual dimension at BLOCK_SIZE granularity
                    for mutual_index_start in (0..mutual_axis_length).step_by(BLOCK_SIZE) {
                        let mutual_index_end =
                            (mutual_index_start + BLOCK_SIZE).min(mutual_axis_length);

                        // Now actually compute the block dot products
                        for row_index in row_index_start..row_index_end {
                            for mutual_index in mutual_index_start..mutual_index_end {
                                let self_value = self.get_2d(row_index, mutual_index);
                                for col_index in col_index_start..col_index_end {
                                    data[row_index * result_col_length + col_index] +=
                                        self_value * other.get_2d(mutual_index, col_index);
                                }
                            }
                        }
                    }
                }
            });

        result
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
    #[should_panic(expected = "Shape does not match length of data!")]
    fn test_bad_new() {
        let _ = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 3]);
    }

    #[test]
    fn test_simple_2d_matmul() {
        let tensor_a = Tensor::new(
            vec![1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            vec![4, 3],
        );

        let tensor_b = Tensor::new(
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 2.0],
            vec![3, 3],
        );

        let tensor_c = tensor_a.simple_2d_matmul(&tensor_b);

        assert_eq!(tensor_c.shape, vec![4, 3]);
        assert_eq!(
            tensor_c.data,
            vec![5.0, 4.0, 3.0, 8.0, 9.0, 5.0, 6.0, 5.0, 3.0, 11.0, 9.0, 6.0]
        );
    }

    #[test]
    fn test_cache_blocked_2d_matmul() {
        let tensor_a = Tensor::new(
            vec![1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            vec![4, 3],
        );

        let tensor_b = Tensor::new(
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 2.0],
            vec![3, 3],
        );

        let tensor_c = tensor_a.cache_blocked_2d_matmul(&tensor_b);

        assert_eq!(tensor_c.shape, vec![4, 3]);
        assert_eq!(
            tensor_c.data,
            vec![5.0, 4.0, 3.0, 8.0, 9.0, 5.0, 6.0, 5.0, 3.0, 11.0, 9.0, 6.0]
        );
    }
}
