# PERFORMANCE CHECKLIST
This is a list of things to consider when trying to optimize a program. It is assumed that the person optimizing already has selected and implemented and algorithm.

## Attempt to find hardware specifications for the machine
Try to find information pertaining to the CPU's cache levels and the processor instructions it has available.

## Identify which constants affect performance
Find which variables in the program affect its performance and "centralize" them using macros or constants that you can easily find in the source code. If possible, find how these variables relate to performance.

## Identify optimization opportunities
### Cache usage
Identify what parts of your algorithm involve many memory read-writes with different variables or memory chunks. Bearing the desired functionality in mind, you can try to think of a different memory layout that allows you to make better use of the cache. You can think of several layouts and try them all out if possible.

In single-threaded programs this might mean trying to make sure your program re-uses as much of the fetched cachelines as possible, so as to avoid an expensive RAM read/write operation. This same line of reasoning works for multi-threaded programs when looking at a single thread, but cache-line sloshing should be avoided by adding padding between the chunks of memory that correspond to different threads. Some concrete elements of the program which allow for better cache usage are:
- *Loops*: Specifically the way the pointer moves and the order in which they are nested.
- *Indirection*: Avoid structures that require you to de-reference multiple layers of pointers when possible.
- *Copies*: Sometimes, copying a chunk of data to a different, contiguous location in memory might impove performance if the chunk is going to be re-used often for a period of time.

### Data types
Certain numeric types might perform better in some machines. For example, in my experience, the `int` operations in my machine usually run somewhat faster than all other numeric types.

### Compilation flags
Adding compilations flags can greatly improve efficiency. Look up your compiler speficiations, some common flags are:
- `-O3`
- `-ffast-math` (breaks [some things](https://stackoverflow.com/questions/7420665/what-does-gccs-ffast-math-actually-do))

### Consider different data structures
Changing data structures for subtitutes which allow for the same operations might improve performance. For example, changing a map for an array in an algorithm that requries sequential access to memory positions or that have a limited number of possible indexes.

### SIMD instructions
Consider using SIMD instructions when possible. Try different SIMD libraries and hardware depending on the machine that you're on and the algorithm. [This guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#) might be useful.

### Other optimization opportunities
Other opportunities might arise from the specific algorithm and might have an unknown relation to performance. You can experiment by adjusting the values of these variables and trying to find the best combination.



 
