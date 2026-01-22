Quiz title: Python 3 NumPy Quiz
Quiz description: A quiz covering NumPy arrays, indexing, slicing, reshaping, broadcasting, and array operations.

Title: ndarray Meaning
Points: 1
1. What is the output of `type(numpy.array([1, 2]))`
a) `<class 'numpy.array'>`
*b)`<class 'numpy.ndarray'>`
c) `<class 'numpy.matrix'>`
d) `<class 'numpy.tensor'>`

Title: Vector vs Matrix
Points: 1
2. In NumPy, what is the difference between a vector and a matrix?
a) Vectors use integers, matrices use floats
*b) A vector is 1-dimensional, a matrix is 2-dimensional
c) Vectors are mutable, matrices are immutable
d) There is no difference

Title: NumPy Indexing Start
Points: 1
3. What does indexing in NumPy start at?
*a) 0
b) 1
c) -1
d) It depends on the array type

Title: Array Concatenation
Points: 1
4. Which function is used to join two arrays along an existing axis?
a) np.join()
b) np.merge()
*c) np.concatenate()
d) np.append()

Title: Reshape Function
Points: 1
5. What does `arr.reshape()` do?
a) Changes the data type of the array
*b) Gives a new shape to an array without changing its data
c) Sorts the array elements
d) Removes duplicate elements

Title: Reshape Requirement
Points: 1
6. When reshaping an array, what requirement must be satisfied?
a) The new shape must have fewer elements
*b) The total number of elements must remain the same
c) The new shape must have more dimensions
d) The array must be 1-dimensional first

Title: np.newaxis Purpose
Points: 1
7. What does `np.newaxis` do?
*a) Increases the dimensions of an array by one
b) Creates a new array
c) Adds new elements to an array
d) Creates a copy of the array

Title: np.newaxis Implementation
Points: 1
8. What is `np.newaxis` implemented as?
a) 0
b) -1
*c) None
d) Ellipsis

Title: Squeeze Method
Points: 1
9. What does the `.squeeze()` method do?
a) Compresses the data values
*b) Removes dimensions of size 1
c) Reduces the array to 1D
d) Removes duplicate elements

Title: Boolean Indexing
Points: 1
10. Given `a = np.array([1, 2, 3, 4, 5])`, what does `a[a < 3]` return?
a) [True, True, False, False, False]
*b) [1, 2]
c) [3, 4, 5]
d) An error

Title: Logical Operators
Points: 1
11. What do the operators `&` and `|` do when used with NumPy boolean arrays?
a) Bitwise operations on integers
*b) Element-wise AND and OR operations
c) Concatenate arrays
d) Compare array shapes

Title: np.nonzero Function
Points: 1
12. What does `np.nonzero()` return?
a) All zero elements in an array
*b) The indices of non-zero elements
c) A count of non-zero elements
d) An array with zeros removed

Title: Ellipsis in Indexing
Points: 1
13. What does `...` (Ellipsis) do in NumPy indexing?
*a) Expands to the number of colons needed to index all dimensions
b) Selects the last element
c) Creates a copy of the array
d) Raises an error

Title: View vs Copy
Points: 1
14. What is the difference between a view and a copy in NumPy?
*a) A view shares data with the original array; a copy has its own data
b) A view is slower; a copy is faster
c) A view is immutable; a copy is mutable
d) There is no difference

Title: Modifying a View
Points: 1
15. What happens when you modify a view of a NumPy array?
a) Only the view is modified
*b) The original array is also modified
c) An error is raised
d) A new array is created

Title: Deep Copy Method
Points: 1
16. Which method creates a deep copy of a NumPy array?
a) arr.view()
b) arr.clone()
*c) arr.copy()
d) np.duplicate(arr)

Title: Broadcasting Definition
Points: 1
17. What is broadcasting in NumPy?
a) Sending arrays to multiple processes
*b) A mechanism that allows operations on arrays of different shapes
c) Converting arrays to different data types
d) Distributing computations across GPUs

Title: Sum Axis 0
Points: 1
18. What does `np.sum(arr, axis=0)` do for a 2D array?
*a) Sums along columns (returns one value per column)
b) Sums along rows (returns one value per row)
c) Returns the total sum
d) Returns the cumulative sum

Title: Sum Axis 1
Points: 1
19. What does `np.sum(arr, axis=1)` do for a 2D array?
a) Sums along columns
*b) Sums along rows (returns one value per row)
c) Returns the total sum
d) Flattens and sums

Title: Flatten vs Ravel
Points: 1
20. What is the difference between `.flatten()` and `.ravel()`?
a) They are identical
b) flatten() is faster
*c) flatten() returns a copy; ravel() returns a view when possible
d) ravel() returns a copy; flatten() returns a view

Title: Transpose Property
Points: 1
21. What property gives you the transpose of a NumPy array?
a) arr.transpose
*b) arr.T
c) arr.trans
d) arr.flip

Title: vstack Function
Points: 1
22. What does `np.vstack()` do?
a) Validates the stack
*b) Stacks arrays vertically (row-wise)
c) Creates a vector stack
d) Stacks arrays diagonally

Title: hstack Function
Points: 1
23. What does `np.hstack()` do?
*a) Stacks arrays horizontally (column-wise)
b) Creates a hash stack
c) Stacks arrays vertically
d) Creates a heap stack

Title: Zeros Array
Points: 1
24. How do you create an array of all zeros with shape (3, 4)?
a) np.array(0, (3, 4))
*b) np.zeros((3, 4))
c) np.zeros(3, 4)
d) np.zero((3, 4))

Title: Ones Array
Points: 1
25. How do you create an array of all ones with shape (2, 3)?
a) np.array(1, (2, 3))
*b) np.ones((2, 3))
c) np.ones(2, 3)
d) np.one((2, 3))

Title: Memory Order
Points: 1
26. What determines whether NumPy uses C-order or Fortran-order for array storage?
*a) How indices correspond to the order the array is stored in memory
b) The data type of the array
c) The number of dimensions
d) The total size of the array

Title: C-Order Index
Points: 1
27. In C-order (row-major), which index changes most rapidly when traversing memory?
a) The first index
*b) The last index
c) The middle index
d) All indices change at the same rate

Title: hsplit Function
Points: 1
28. What does `np.hsplit(arr, 3)` do?
*a) Splits the array into 3 equal parts horizontally
b) Splits the array into 3 equal parts vertically
c) Removes every 3rd element
d) Creates 3 copies of the array

Title: Flatten Result
Points: 1
29. Given `arr = np.array([[1, 2], [3, 4]])`, what does `arr.flatten()` return?
a) [[1, 2, 3, 4]]
*b) [1, 2, 3, 4]
c) [[1], [2], [3], [4]]
d) [[1, 2], [3, 4]]

Title: Standard Deviation
Points: 1
30. What aggregation function computes the standard deviation in NumPy?
a) np.stddev()
*b) np.std()
c) np.stdev()
d) np.standard_deviation()
