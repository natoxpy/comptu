use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};

use rand::{distributions::Uniform, Rng};

#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub axes: Vec<Vector>,
    pub shape: (usize, usize),
}

impl Vector {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn random(minmax: (f32, f32), count: usize) -> Self {
        let mut rng = rand::thread_rng();
        let range = Uniform::new_inclusive(minmax.0, minmax.1);

        let data = vec![0.; count].iter().map(|_| rng.sample(range)).collect();

        Self { data }
    }
}

impl Matrix {
    pub fn new(axes: Vec<Vector>, shape: (usize, usize)) -> Self {
        Self { axes, shape }
    }

    pub fn get_row(&self, index: usize) -> &Vector {
        self.axes.get(index).unwrap()
    }

    pub fn get_column(&self, index: usize) -> Vector {
        let mut column = vec![];

        for i in 0..self.shape.0 {
            let row = self.get_row(i);
            column.push(*row.data.get(index).unwrap());
        }

        Vector::new(column)
    }

    pub fn random(minmax: (f32, f32), shape: (usize, usize)) -> Self {
        let mut axes = vec![];

        for _ in 0..shape.0 {
            axes.push(Vector::random(minmax, shape.1))
        }

        Self { axes, shape }
    }
}

///
/// Addition and subtraction
///

impl Add<&Vector> for &Vector {
    type Output = Vector;

    fn add(self, rhs: &Vector) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!(
                "lhs and rhs vector are not the same length {} != {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        Vector::new(
            self.data
                .iter()
                .zip(rhs.data.iter())
                .map(|(l, r)| l + r)
                .collect(),
        )
    }
}

impl Sub<&Vector> for &Vector {
    type Output = Vector;

    fn sub(self, rhs: &Vector) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!(
                "lhs and rhs vector are not the same length {} != {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        Vector::new(
            self.data
                .iter()
                .zip(rhs.data.iter())
                .map(|(l, r)| l - r)
                .collect(),
        )
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        if self.shape != rhs.shape {
            panic!(
                "lhs and rhs matrix shape are not the same length {:?} != {:?}",
                self.shape, rhs.shape
            );
        }

        Matrix::new(
            self.axes
                .iter()
                .zip(rhs.axes.iter())
                .map(|(l, r)| l + r)
                .collect(),
            self.shape,
        )
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        if self.shape != rhs.shape {
            panic!(
                "lhs and rhs matrix shape are not the same length {:?} != {:?}",
                self.shape, rhs.shape
            );
        }

        Matrix::new(
            self.axes
                .iter()
                .zip(rhs.axes.iter())
                .map(|(l, r)| l - r)
                .collect(),
            self.shape,
        )
    }
}

///
/// Hadamard product
///

impl Mul<&Vector> for &Vector {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!(
                "lhs and rhs vector are not the same length {} != {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        Vector::new(
            self.data
                .iter()
                .zip(rhs.data.iter())
                .map(|(l, r)| l * r)
                .collect(),
        )
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.shape != rhs.shape {
            panic!(
                "lhs and rhs matrix shape are not the same length {:?} != {:?}",
                self.shape, rhs.shape
            );
        }

        Matrix::new(
            self.axes
                .iter()
                .zip(rhs.axes.iter())
                .map(|(left, right)| left * right)
                .collect(),
            self.shape,
        )
    }
}

///
/// Multiplication
///

impl Vector {
    pub fn dot(&self, rhs: &Vector) -> f32 {
        if self.data.len() != rhs.data.len() {
            panic!(
                "lhs and rhs vector are not the same length {} != {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(left, right)| left * right)
            .fold(0., |acc, v| acc + v)
    }
}

impl Matrix {
    #[allow(unused)]
    pub fn dot(&self, rhs: &Matrix) -> Matrix {
        if self.shape.1 != rhs.shape.0 {
            panic!(
                "lhs and rhs matrix shape are not compatible for matrix multiplication {:?} != {:?}",
                self.shape, rhs.shape
            );
        }

        let mut axes = vec![];
        let shape = (self.shape.0, rhs.shape.1);

        for row in self.axes.iter() {
            let mut vector = vec![];

            for column_index in 0..rhs.shape.1 {
                let column = rhs.get_column(column_index);

                vector.push(row.dot(&column))
            }

            axes.push(Vector::new(vector));
        }

        Matrix::new(axes, shape)
    }
}

///
/// Display
///

const DISPLAY_VECTOR_SIZE: usize = 10;
const DISPLAY_MATRIX_SIZE: usize = 10;

impl Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;

        let d = |a: &f32| format!("{:+03.6}", a);

        if self.data.len() <= DISPLAY_VECTOR_SIZE {
            for (i, data) in self.data.iter().enumerate() {
                if i == self.data.len() - 1 {
                    write!(f, "{}", d(data))?;
                } else {
                    write!(f, "{}, ", d(data))?;
                }
            }
        } else {
            let first = self.data[0..(DISPLAY_VECTOR_SIZE / 2)].to_vec();
            let last =
                self.data[self.data.len() - (DISPLAY_VECTOR_SIZE / 2)..self.data.len()].to_vec();

            for (i, data) in first.iter().enumerate() {
                if i == first.len() - 1 {
                    write!(f, "{}", d(data))?;
                } else {
                    write!(f, "{}, ", d(data))?;
                }
            }

            write!(f, " ... ")?;

            for (i, data) in last.iter().enumerate() {
                if i == last.len() - 1 {
                    write!(f, "{}", d(data))?;
                } else {
                    write!(f, "{}, ", d(data))?;
                }
            }
        }

        write!(f, "]")?;

        Ok(())
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")?;
        writeln!(f, "Matrix [{}, {}]", self.shape.0, self.shape.1)?;

        writeln!(f, "[")?;

        if self.axes.len() <= DISPLAY_MATRIX_SIZE {
            for row in self.axes.iter() {
                writeln!(f, "  {}", row)?;
            }
        } else {
            let first = self.axes[0..(DISPLAY_MATRIX_SIZE / 2)].to_vec();
            let last =
                self.axes[self.axes.len() - (DISPLAY_MATRIX_SIZE / 2)..self.axes.len()].to_vec();

            for row in first {
                writeln!(f, "  {}", row)?;
            }

            // filler
            writeln!(f)?;

            for row in last {
                writeln!(f, "  {}", row)?;
            }
        }

        write!(f, "]")?;

        Ok(())
    }
}

///
/// macro
///

#[macro_export]
macro_rules! matrix {
    ($([$($elem:expr),+]),+) => {
        {
            use self::matrix::{Vector, Matrix};

            let mut axes = Vec::new();
            let mut shape = (0, 0);

            $(
                let vector_data = vec![$($elem as f32),+];
                let vector = Vector::new(vector_data.clone());

                axes.push(vector);
                shape.0 += 1;

                if shape.1 == 0 {
                    shape.1 = vector_data.len();
                } else {
                    assert_eq!(shape.1,
                        vector_data.len(),
                        "All rows must have the same length"
                    );
                }
            )+

            Matrix::new(axes, shape)
        }
    };
}

///
/// Experimental
///

#[derive(Debug, Clone)]
pub struct Padding {
    pub row: usize,
    pub column: usize,
}

#[derive(Debug, Clone)]
pub struct MatrixBlock {
    pub matrices: Vec<Vec<Matrix>>,
    pub padding: Padding,
    pub shape: (usize, usize),
    pub matrix_shape: (usize, usize),
}

impl Padding {
    pub fn new(row: usize, column: usize) -> Self {
        Self { row, column }
    }
}

impl Matrix {
    pub fn to_array(&self) -> Vec<f32> {
        self.axes.iter().flat_map(|a| a.data.clone()).collect()
    }

    pub fn transpose(&mut self) {
        let mut rows = vec![];

        self.shape = (self.shape.1, self.shape.0);

        for column in 0..self.shape.1 {
            rows.push(self.get_column(column));
        }

        self.axes = rows;
    }

    pub fn block(&self, size: (usize, usize)) -> MatrixBlock {
        let mut rows = vec![];

        // Rows
        let rchunks = self
            .axes
            .chunks(size.0)
            .map(|v| v.to_vec())
            .collect::<Vec<Vec<Vector>>>();

        let mut padding = Padding::new(0, 0);

        let mut shape = (rchunks.len(), 0);
        let matrix_shape = size;

        for row_chunk in rchunks.iter() {
            // Columns
            let rows_chunks = row_chunk
                .iter()
                .map(|v| {
                    v.data
                        .chunks(size.1)
                        .map(|v| v.to_vec())
                        .collect::<Vec<Vec<f32>>>()
                })
                .collect::<Vec<Vec<Vec<f32>>>>();

            let last_column_chunk = rows_chunks.last().unwrap();
            let last_col = last_column_chunk.last().unwrap();
            shape.1 = last_column_chunk.len();

            padding.column = size.1 - last_col.len();

            padding.row = size.0 - rows_chunks.len();

            let mut block_column = vec![];

            for row_index in 0..last_column_chunk.len() {
                let mut mat_rows = vec![];

                for row_chunk_index in 0..size.0 {
                    let mut row;

                    let row_chunk = rows_chunks.get(row_chunk_index);

                    if row_chunk_index < rows_chunks.len() && row_index < row_chunk.unwrap().len() {
                        row = row_chunk.unwrap().get(row_index).unwrap().clone();
                    } else {
                        row = vec![0.; 1];
                    }

                    row.resize(size.1, 0.);

                    mat_rows.push(Vector::new(row));
                }

                block_column.push(Matrix::new(mat_rows, size));
            }

            rows.push(block_column);
        }

        MatrixBlock {
            matrices: rows,
            padding,
            matrix_shape,
            shape,
        }
    }
}

impl MatrixBlock {
    pub fn transpose(&mut self) {
        let mut matrices = vec![];

        self.shape = (self.shape.1, self.shape.0);

        for column in 0..self.shape.1 {
            matrices.push(self.get_column(column));
        }

        self.matrices = matrices;
    }

    pub fn inner_transpose(&mut self) {
        for matrices in self.matrices.iter_mut() {
            for matrix in matrices.iter_mut() {
                matrix.transpose();
            }
        }
    }

    pub fn from_array(
        array: Vec<f32>,
        block_size: (usize, usize),
        matrix_size: (usize, usize),
    ) -> Self {
        let mut matrices = Vec::new();

        let mut index = 0;

        for _ in 0..block_size.0 {
            let mut block_rows = Vec::new();

            for _ in 0..block_size.1 {
                let mut mat_rows = Vec::new();

                for _ in 0..matrix_size.0 {
                    let mut vector_row = Vec::new();

                    for _ in 0..matrix_size.1 {
                        let item = array[index];
                        index += 1;
                        vector_row.push(item);
                    }

                    mat_rows.push(Vector::new(vector_row));
                }

                block_rows.push(Matrix::new(mat_rows, matrix_size));
            }

            matrices.push(block_rows);
        }

        Self::new(matrices, block_size, matrix_size, Padding::new(0, 0))
    }

    /// return (data, block_size, matrix_size)
    pub fn to_array(&self) -> (Vec<f32>, (usize, usize), (usize, usize)) {
        let arrays = self
            .matrices
            .iter()
            .flat_map(|m| m.iter().flat_map(|m| m.to_array()))
            .collect();

        (arrays, self.shape, self.matrix_shape)
    }

    pub fn new(
        matrices: Vec<Vec<Matrix>>,
        shape: (usize, usize),
        matrix_shape: (usize, usize),
        padding: Padding,
    ) -> Self {
        Self {
            matrices,
            shape,
            matrix_shape,
            padding,
        }
    }

    pub fn print(&self) {
        for (i, row) in self.matrices.iter().enumerate() {
            println!("row {}", i + 1);
            for mat in row.iter() {
                println!("{}\n", mat);
            }
        }
    }

    pub fn to_matrix(&self) -> Matrix {
        let mut rows = vec![];
        let matrices_first = self.matrices.first().unwrap();

        let shape = (
            self.matrices
                .iter()
                .fold(0, |acc, m| acc + m.first().unwrap().shape.0)
                - self.padding.row,
            matrices_first.iter().fold(0, |acc, m| acc + m.shape.1) - self.padding.column,
        );

        for (i, matrices) in self.matrices.iter().enumerate() {
            let mut row_count = matrices.first().unwrap().shape.0;

            if i == self.matrices.len() - 1 {
                row_count -= self.padding.row;
            }

            for r in 0..row_count {
                let mut row = vec![];

                for matrix in matrices {
                    let mat_row = matrix.get_row(r);
                    let mut mat_row_data = mat_row.clone();

                    mat_row_data
                        .data
                        .truncate(mat_row.data.len() - self.padding.column);

                    row.push(mat_row_data);
                }

                rows.push(Vector::new(
                    row.iter().flat_map(|r| r.data.clone()).collect(),
                ));
            }
        }

        Matrix::new(rows, shape)
    }

    pub fn get_row(&self, index: usize) -> &Vec<Matrix> {
        self.matrices.get(index).unwrap()
    }

    pub fn get_column(&self, index: usize) -> Vec<Matrix> {
        let mut column = vec![];

        for i in 0..self.shape.0 {
            let row = self.get_row(i);

            column.push(row.get(index).unwrap().clone());
        }

        column
    }

    pub fn dot(&self, rhs: &MatrixBlock) -> MatrixBlock {
        if self.shape.1 != rhs.shape.0 {
            panic!(
                "lhs and rhs matrix shape are not compatible for block matrix multiplication {:?} != {:?}",
                self.shape, rhs.shape
            );
        }

        let mut matrices = vec![];
        let shape = (self.shape.0, rhs.shape.1);
        let matrix_shape = self.matrices.first().unwrap().first().unwrap().shape;

        for row in self.matrices.iter() {
            let mut vector = vec![];

            for column_index in 0..rhs.shape.1 {
                let column = rhs.get_column(column_index);

                let di = row
                    .iter()
                    .zip(column)
                    .map(|(left, right)| left.dot(&right))
                    .collect::<Vec<Matrix>>();

                let mut n = di.first().unwrap().clone();

                for d in di[1..].iter() {
                    n = &n + d;
                }

                vector.push(n.clone())
            }

            matrices.push(vector);
        }

        MatrixBlock::new(
            matrices,
            matrix_shape,
            shape,
            Padding::new(self.padding.row, rhs.padding.column),
        )
    }
}
