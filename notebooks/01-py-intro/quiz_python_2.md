Quiz title: Python 2 Quiz
Quiz description: A quiz covering Python exceptions, data model, operator overloading, iterators, scopes, and classes.

Title: ZeroDivisionError
Points: 1
1. What type of error occurs when you try to execute `10 * (1/0)`?
a) TypeError
b) ValueError
*c) ZeroDivisionError
d) ArithmeticError

Title: NameError
Points: 1
2. What type of error occurs when you try to use an undefined variable like `spam` in an expression?
a) TypeError
b) ValueError
*c) NameError
d) ReferenceError

Title: TypeError
Points: 1
3. What type of error occurs when you try to execute `'2' + 2`?
a) ValueError
b) SyntaxError
*c) TypeError
d) CastError

Title: Except Clause Execution
Points: 1
4. In a try/except block, when is the except clause executed?
a) Always, regardless of whether an exception occurs
*b) Only when an exception matching the specified type occurs in the try clause
c) Before the try clause is executed
d) Only when the try clause completes successfully

Title: Raise Statement
Points: 1
5. What does the `raise` statement do in Python?
a) Catches an exception
b) Ignores an exception
*c) Forces a specified exception to occur
d) Logs an exception to a file

Title: With Statement Purpose
Points: 1
6. What is the purpose of the `with` statement in Python?
a) To define a while loop
b) To create a new variable scope
*c) To ensure proper cleanup of resources like files
d) To catch exceptions silently

Title: File Resource Leak
Points: 1
7. What happens if you open a file without using `with` and an exception occurs before closing it?
a) Python automatically closes the file
*b) The file may remain open, causing resource leaks
c) The program crashes immediately
d) The file is deleted

Title: __len__ Method
Points: 1
8. What special method should you implement to make `len()` work on your custom class?
a) __length__
*b) __len__
c) __size__
d) length

Title: __add__ Method
Points: 1
9. Which special method is called when you use the `+` operator on objects?
*a) `__add__`
b) `__plus__`
c) `__sum__`
d) `__concat__`

Title: __matmul__ Method
Points: 1
10. What special method is used for the matrix multiplication operator `@`?
a) `__mul__`
b) `__mmul__`
*c) `__matmul__`
d) `__matrix__`

Title: __str__ vs __repr__
Points: 1
11. What is the difference between `__str__` and `__repr__`?
a) They are identical in purpose
*b) `__str__` is for human-readable output; `__repr__` is for unambiguous representation
c) `__str__` is for integers; `__repr__` is for strings
d) `__str__` is deprecated; only `__repr__` should be used

Title: Iterator Methods
Points: 1
12. To make a custom class iterable, which two special methods must be implemented?
a) `__loop__` and `__next__`
*b) `__iter__` and `__next__`
c) `__for__` and `__in__`
d) `__iterate__` and `__step__`

Title: StopIteration
Points: 1
13. What exception should be raised to signal the end of iteration?
a) EndIteration
b) IterationComplete
*c) StopIteration
d) NoMoreItems

Title: Namespace Definition
Points: 1
14. In Python, what is a namespace?
a) A directory containing Python files
*b) A mapping from names to objects
c) A special type of dictionary
d) A module's file path

Title: Nonlocal Keyword
Points: 1
15. What does the `nonlocal` keyword do?
a) Declares a variable as global
*b) Allows modification of a variable in an enclosing (non-global) scope
c) Creates a new local variable
d) Prevents a variable from being modified

Title: Global Keyword
Points: 1
16. What does the `global` keyword do inside a function?
*a) Allows the function to modify a variable in the module's global scope
b) Creates a new global variable only accessible in that function
c) Imports a global module
d) Declares a constant that cannot be changed

Title: Scope Search Order
Points: 1
17. What are the three nested scopes searched (in order) when looking up a name in Python?
a) Global, module, local
*b) Local, enclosing functions, global (then built-in)
c) Built-in, global, local
d) Module, class, function

Title: Data Attribute
Points: 1
18. In a class, what is a data attribute?
*a) An instance variable that stores data
b) A method that returns data
c) A class-level constant
d) A type annotation

Title: Method vs Function
Points: 1
19. What is the difference between a method and a regular function in Python?
a) Methods can only return None
*b) Methods are functions that belong to an object and receive `self` as the first parameter
c) Methods cannot have parameters
d) Methods are always faster than functions

Title: Method Access
Points: 1
20. Given a class with `def f(self):`, how can you access this method from an instance `x`?
*a) x.f()
b) f(x)
c) x->f()
d) x::f()
