Data Structures
====================

### Arrays

* contiguous areas of memory, arranges in row-major / col-major form
* element access - `O(1)`
* element add / remove at end - `O(1)`
* element add/remove at arbitrary location - **`O(n)`** --> hence use Linked-Lists

### Linked Lists

* constant time for insert / remove

**Operation**	- **Singly**		-	**Doubly**
PushFront(key)	- O(1)
TopFront()	- O(1)
PopFront()	- O(1)
PushBack(Key)	- O(n); O(1) with tail
TopBack()	- O(n); O(1) with tail
PopBack()	- O(n)			-	O(1)
FindKey(key)	- O(n)
Erase(Key)	- O(n)
Empty()		- O(1) - check if head != NULL
AddBefore(node,key) - O(n)		-	O(1)
AddAfter(node, key) - O(1)

### Stacks

* Abstract datatype
* *similar to a stack of books!*
* **LIFO**
* Supports:
	- Push(Key) = PushFront()
	- Key Top() = TopFront() + PopFront()
	- Key Pop() = PopFront()
	- Boolean Empty()

* Can be implemented using array (fixed size) / LinkedList (dynamic-size)

### Queues

* Abstract datatype
* *Similar to a queue of people!*
* **FIFO**
* Supports:
	- Enqueue(Key) = PushBack(Key)
	- Key Dequeue() = TopFront() + PopFront()

### Trees

* Contains:
	- top key - parent
	- list of child nodes - child
	- ancestor - upwards
	- descendatns - downwards
	- siblings - similar level
	- leaf - no child
	- `LEVEL = 1 + (# edges to root)`
	- `HEIGHT = 1 + (# edges to farthest leaf)` - can be calculated 
	   recursively down to up

#### Binary Tree

* Contains
	- key
	- left node
	- right node

### Tree Traversal

* Some common techniques include:
- *Depth-first* : traverse a subtree, before exploring its siblings - `In/Pre/Post-OrderTraversal()`
- Pre : node then children
- Post : children then node
- *Breadth-first* : traverse all nodes at a level, before exploring next level - 

