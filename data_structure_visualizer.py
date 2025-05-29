import collections
import random
import hashlib

# --- Linear Data Structures ---

# Doubly Linked List
class DoublyLinkedListNode:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

    def __repr__(self):
        return f"Node({self.value})"

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, value):
        new_node = DoublyLinkedListNode(value)
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.size += 1
        print(f"Appended {value}. List state:")
        self.display()

    def prepend(self, value):
        new_node = DoublyLinkedListNode(value)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1
        print(f"Prepended {value}. List state:")
        self.display()

    def delete_by_value(self, value):
        if not self.head:
            print("List is empty. Cannot delete.")
            return

        current = self.head
        found = False
        while current:
            if current.value == value:
                found = True
                if current == self.head:
                    self.head = current.next
                    if self.head:
                        self.head.prev = None
                    else:
                        self.tail = None
                elif current == self.tail:
                    self.tail = current.prev
                    self.tail.next = None
                else:
                    current.prev.next = current.next
                    current.next.prev = current.prev
                self.size -= 1
                print(f"Deleted {value}. List state:")
                break
            current = current.next
        if not found:
            print(f"Value {value} not found for deletion. List state:")
        self.display()

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(str(current.value))
            current = current.next
        if not elements:
            print("  (empty)")
        else:
            print(f"  Head <-> {' <-> '.join(elements)} <-> Tail (Size: {self.size})")
        print("-" * 30)

# Circular Linked List
class CircularLinkedListNode:
    def __init__(self, value):
        self.value = value
        self.next = None

    def __repr__(self):
        return f"Node({self.value})"

class CircularSinglyLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def append(self, value):
        new_node = CircularLinkedListNode(value)
        if not self.head:
            self.head = new_node
            new_node.next = self.head
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            current.next = new_node
            new_node.next = self.head
        self.size += 1
        print(f"Appended {value}. List state:")
        self.display()

    def prepend(self, value):
        new_node = CircularLinkedListNode(value)
        if not self.head:
            self.head = new_node
            new_node.next = self.head
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            new_node.next = self.head
            current.next = new_node
            self.head = new_node
        self.size += 1
        print(f"Prepended {value}. List state:")
        self.display()

    def delete_by_value(self, value):
        if not self.head:
            print("List is empty. Cannot delete.")
            return

        if self.head.value == value and self.head.next == self.head:
            self.head = None
            self.size -= 1
            print(f"Deleted {value}. List state:")
            self.display()
            return

        prev = None
        current = self.head
        found = False
        while True:
            if current.value == value:
                found = True
                if current == self.head:
                    temp = self.head
                    while temp.next != self.head:
                        temp = temp.next
                    self.head = self.head.next
                    temp.next = self.head
                else:
                    prev.next = current.next
                self.size -= 1
                print(f"Deleted {value}. List state:")
                break
            prev = current
            current = current.next
            if current == self.head:
                break

        if not found:
            print(f"Value {value} not found for deletion. List state:")
        self.display()

    def display(self):
        if not self.head:
            print("  (empty)")
            print("-" * 30)
            return

        elements = []
        current = self.head
        while True:
            elements.append(str(current.value))
            current = current.next
            if current == self.head:
                break
        print(f"  {' -> '.join(elements)} -> ... (back to {self.head.value}) (Size: {self.size})")
        print("-" * 30)

# Simple Queue (List-based for conceptual clarity)
class SimpleQueue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)
        print(f"Enqueued {item}. Queue state:")
        self.display()

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
            return None
        item = self.queue.pop(0) # O(N) operation for lists
        print(f"Dequeued {item}. Queue state:")
        self.display()
        return item

    def front(self):
        if self.is_empty():
            print("Queue is empty. No front element.")
            return None
        return self.queue[0]

    def is_empty(self):
        return len(self.queue) == 0

    def display(self):
        if self.is_empty():
            print("  (empty)")
        else:
            print(f"  Front <-- {' <-- '.join(map(str, self.queue))} <-- Rear (Size: {len(self.queue)})")
        print("-" * 30)

# Circular Queue
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def enqueue(self, item):
        if self.size == self.capacity:
            print("Queue is full. Cannot enqueue.")
            return False
        self.queue[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1
        print(f"Enqueued {item}. Queue state:")
        self.display()
        return True

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
            return None
        item = self.queue[self.head]
        self.queue[self.head] = None
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        print(f"Dequeued {item}. Queue state:")
        self.display()
        return item

    def front(self):
        if self.is_empty():
            return None
        return self.queue[self.head]

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity

    def display(self):
        if self.is_empty():
            print(f"  Array: {self.queue} | (empty)")
        else:
            display_arr = []
            for i in range(self.size):
                idx = (self.head + i) % self.capacity
                display_arr.append(str(self.queue[idx]))
            print(f"  Array: {self.queue} | Head: {self.head}, Tail: {self.tail}, Size: {self.size}")
            print(f"  Conceptual: Front ({self.queue[self.head]}) <-- {' <-- '.join(display_arr)} <-- Rear ({self.queue[(self.tail - 1 + self.capacity) % self.capacity]})")
        print("-" * 30)


# --- Non-Linear Data Structures ---

# Tree (General N-ary Tree)
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def __repr__(self):
        return f"Node({self.value})"

def add_child_to_tree(parent_node, child_value):
    child_node = TreeNode(child_value)
    parent_node.children.append(child_node)
    return child_node

def visualize_tree(node, level=0, prefix=""):
    if node is not None:
        indent = "    " * level
        print(f"{indent}{prefix}{node.value}")
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            child_prefix = "└── " if is_last_child else "├── "
            visualize_tree(child, level + 1, child_prefix)

# Heap (Min-Heap, Max-Heap using heapq)
import heapq # Python's built-in min-heap

# Trie (Prefix Tree)
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        print(f"Inserted '{word}'.")
        self.display()

    def search(self, word):
        node = self.root
        path = []
        for char in word:
            if char not in node.children:
                print(f"Search for '{word}': NOT FOUND. Path: {' -> '.join(path)}")
                return False
            path.append(char)
            node = node.children[char]
        result = node.is_end_of_word
        print(f"Search for '{word}': {'FOUND' if result else 'NOT FOUND (prefix exists)'}. Path: {' -> '.join(path)}")
        return result

    def starts_with(self, prefix):
        node = self.root
        path = []
        for char in prefix:
            if char not in node.children:
                print(f"Prefix '{prefix}': NO WORDS. Path: {' -> '.join(path)}")
                return False
            path.append(char)
            node = node.children[char]
        print(f"Prefix '{prefix}': WORDS EXIST. Path: {' -> '.join(path)}")
        return True

    def _display_trie(self, node, prefix="", level=0):
        indent = "  " * level
        status = "(End of Word)" if node.is_end_of_word else ""
        print(f"{indent}{prefix}{status}")
        for char, child_node in sorted(node.children.items()):
            self._display_trie(child_node, f"[{char}]", level + 1)

    def display(self):
        print("Current Trie Structure (simplified):")
        self._display_trie(self.root, "ROOT")
        print("-" * 30)

# Graph (Adjacency List Representation)
class Graph:
    def __init__(self, is_directed=False):
        self.graph = collections.defaultdict(list) # Adjacency list
        self.is_directed = is_directed
        self.vertices = set()

    def add_vertex(self, vertex):
        self.vertices.add(vertex)
        if vertex not in self.graph:
            self.graph[vertex] = []
        print(f"Added vertex: {vertex}")
        self.display()

    def add_edge(self, u, v, weight=1):
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append((v, weight))
        if not self.is_directed:
            self.graph[v].append((u, weight))
        print(f"Added edge: {u} {'--' if not self.is_directed else '->'} {v} (Weight: {weight})")
        self.display()

    def remove_vertex(self, vertex):
        if vertex not in self.vertices:
            print(f"Vertex {vertex} not found.")
            return

        self.vertices.discard(vertex)
        for vtx in list(self.graph.keys()):
            self.graph[vtx] = [(neighbor, w) for neighbor, w in self.graph[vtx] if neighbor != vertex]
        if vertex in self.graph:
            del self.graph[vertex]
        print(f"Removed vertex: {vertex}")
        self.display()

    def remove_edge(self, u, v):
        if u not in self.graph or v not in self.vertices:
            print(f"Edge {u} -> {v} not found (one or both vertices missing).")
            return

        self.graph[u] = [(neighbor, w) for neighbor, w in self.graph[u] if neighbor != v]
        if not self.is_directed:
            self.graph[v] = [(neighbor, w) for neighbor, w in self.graph[v] if neighbor != u]
        print(f"Removed edge: {u} {'--' if not self.is_directed else '->'} {v}")
        self.display()

    def display(self):
        print("Current Graph (Adjacency List):")
        if not self.vertices:
            print("  (empty)")
        else:
            sorted_vertices = sorted(list(self.vertices))
            for vertex in sorted_vertices:
                neighbors = self.graph.get(vertex, [])
                formatted_neighbors = []
                for neighbor, weight in sorted(neighbors):
                    if weight == 1:
                        formatted_neighbors.append(str(neighbor))
                    else:
                        formatted_neighbors.append(f"{neighbor}({weight})")

                print(f"  {vertex}: [{', '.join(formatted_neighbors)}]")
        print("-" * 40)


# --- Specialized or Advanced Data Structures ---

# Disjoint Set / Union-Find
class DisjointSet:
    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}
        print(f"Initialized Disjoint Set with elements: {list(elements)}")
        self.display()

    def find(self, i):
        path_taken = []
        current = i
        while current != self.parent[current]:
            path_taken.append(current)
            current = self.parent[current]
        
        root = current
        for node in path_taken:
            self.parent[node] = root
        
        print(f"Find({i}): Root is {root}. Path compressed: {path_taken}")
        self.display()
        return root

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                print(f"Union({i}, {j}): {root_i} attached under {root_j} (rank).")
            elif self.rank[root_j] < self.rank[root_i]:
                self.parent[root_j] = root_i
                print(f"Union({i}, {j}): {root_j} attached under {root_i} (rank).")
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
                print(f"Union({i}, {j}): {root_j} attached under {root_i} (equal rank, rank_of_{root_i} incremented).")
            self.display()
            return True
        else:
            print(f"Union({i}, {j}): {i} and {j} are already in the same set.")
            self.display()
            return False

    def display(self):
        print("Current Disjoint Set State:")
        for elem in sorted(self.parent.keys()):
            representative = elem
            while representative != self.parent[representative]:
                representative = self.parent[representative]
            print(f"  Element '{elem}' belongs to set represented by '{representative}' (Parent: {self.parent[elem]}, Rank: {self.rank.get(elem, 'N/A')})")
        
        sets = collections.defaultdict(list)
        for elem in self.parent:
            sets[self.find(elem)].append(elem)
        
        print("  Current Sets (Conceptual):")
        for rep, elems in sets.items():
            print(f"    Set of {rep}: {sorted(elems)}")
        print("-" * 40)

# Bloom Filter
class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size
        print(f"Initialized Bloom Filter: Size={size}, Num Hashes={num_hashes}")
        self.display()

    def _hash(self, item, seed):
        h = hashlib.md5(f"{item}-{seed}".encode()).hexdigest()
        return int(h, 16) % self.size

    def add(self, item):
        print(f"Adding '{item}':")
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            self.bit_array[index] = 1
            print(f"  Hash {i+1} maps to index {index}. Set bit_array[{index}] to 1.")
        print(f"'{item}' added.")
        self.display()

    def contains(self, item):
        print(f"Checking if '{item}' exists:")
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                print(f"  Hash {i+1} maps to index {index}. bit_array[{index}] is 0. '{item}' is DEFINITELY NOT present.")
                return False
            else:
                print(f"  Hash {i+1} maps to index {index}. bit_array[{index}] is 1 (match).")
        print(f"'{item}' MIGHT BE present (all bits set).")
        return True

    def display(self):
        print("Current Bloom Filter Bit Array:")
        print(f"  [{' '.join(map(str, self.bit_array))}]")
        print("-" * 40)

# Skip List
class SkipListNode:
    def __init__(self, value, level):
        self.value = value
        self.next = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=4, p=0.5):
        self.max_level = max_level
        self.p = p
        self.head = SkipListNode(float('-inf'), max_level)
        self.level = 0
        print(f"Initialized Skip List (Max Level: {max_level}, Probability p: {p})")
        self.display()

    def _random_level(self):
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def insert(self, value):
        print(f"Inserting {value}:")
        update = [None] * (self.max_level + 1)
        current = self.head

        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].value < value:
                current = current.next[i]
            update[i] = current

        current = current.next[0]

        if current is None or current.value != value:
            new_level = self._random_level()
            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.head
                self.level = new_level

            new_node = SkipListNode(value, new_level)
            for i in range(new_level + 1):
                new_node.next[i] = update[i].next[i]
                update[i].next[i] = new_node
            print(f"  Inserted {value} at level {new_level}.")
            self.display()
        else:
            print(f"  Value {value} already exists.")
            self.display()

    def search(self, value):
        print(f"Searching for {value}:")
        current = self.head
        path = []
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].value < value:
                path.append(f"Level {i}: {current.value} -> {current.next[i].value}")
                current = current.next[i]
            path.append(f"Level {i}: Current at {current.value}")
        
        current = current.next[0]
        
        if current and current.value == value:
            print(f"  Found {value}. Path: {path}")
            return True
        else:
            print(f"  {value} not found. Path: {path}")
            return False

    def delete(self, value):
        print(f"Deleting {value}:")
        update = [None] * (self.max_level + 1)
        current = self.head

        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].value < value:
                current = current.next[i]
            update[i] = current

        current = current.next[0]

        if current and current.value == value:
            for i in range(self.level + 1):
                if update[i].next[i] != current:
                    continue
                update[i].next[i] = current.next[i]
            
            while self.level > 0 and self.head.next[self.level] is None:
                self.level -= 1
            print(f"  Deleted {value}.")
            self.display()
            return True
        else:
            print(f"  Value {value} not found for deletion.")
            self.display()
            return False

    def display(self):
        print("Current Skip List Structure:")
        if self.head.next[0] is None:
            print("  (empty)")
        else:
            for i in range(self.level, -1, -1):
                level_str = f"Level {i:2d}: Head "
                current = self.head.next[i]
                while current:
                    level_str += f"-> {current.value} "
                    current = current.next[i]
                print(level_str + "-> None")
        print("-" * 40)

# LRU Cache (Least Recently Used)
class LRUNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

    def __repr__(self):
        return f"({self.key}: {self.value})"

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = LRUNode(0, 0)
        self.tail = LRUNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        print(f"Initialized LRU Cache with capacity: {capacity}")
        self.display()

    def _add_node(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _move_to_front(self, node):
        self._remove_node(node)
        self._add_node(node)

    def get(self, key):
        print(f"Getting key '{key}':")
        if key in self.cache:
            node = self.cache[key]
            self._move_to_front(node)
            print(f"  Found '{key}': {node.value}. Moved to front.")
            self.display()
            return node.value
        else:
            print(f"  Key '{key}' not found.")
            self.display()
            return -1

    def put(self, key, value):
        print(f"Putting ({key}: {value}):")
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
            print(f"  Key '{key}' updated and moved to front.")
        else:
            new_node = LRUNode(key, value)
            self.cache[key] = new_node
            self._add_node(new_node)

            if len(self.cache) > self.capacity:
                lru_node = self.tail.prev
                self._remove_node(lru_node)
                del self.cache[lru_node.key]
                print(f"  Cache full. Removed LRU item: ({lru_node.key}: {lru_node.value}).")
            print(f"  New item ({key}: {value}) added.")
        self.display()

    def display(self):
        print("Current LRU Cache State:")
        items = []
        current = self.head.next
        while current != self.tail:
            items.append(f"({current.key}:{current.value})")
            current = current.next
        
        print(f"  Cache (MRU -> LRU): {' <-> '.join(items)}")
        print(f"  Size: {len(self.cache)} / Capacity: {self.capacity}")
        print("-" * 40)


# --- Example Usage and Output ---

print("=" * 50)
print("--- DEMONSTRATING DATA STRUCTURES ---")
print("=" * 50)
print("\n")

print("--- Doubly Linked List ---")
dll = DoublyLinkedList()
dll.append(10)
dll.append(20)
dll.prepend(5)
dll.append(30)
dll.delete_by_value(20)
dll.delete_by_value(5)
dll.delete_by_value(100) # Not found
dll.delete_by_value(30)
dll.delete_by_value(10)
dll.delete_by_value(50) # Empty list
print("\n")

print("--- Circular Singly Linked List ---")
cll = CircularSinglyLinkedList()
cll.append(10)
cll.append(20)
cll.prepend(5)
cll.append(30)
cll.delete_by_value(20)
cll.delete_by_value(5)
cll.delete_by_value(100) # Not found
cll.delete_by_value(30)
cll.delete_by_value(10)
print("\n")

print("--- Simple Queue (List-based) ---")
sq = SimpleQueue()
sq.enqueue(10)
sq.enqueue(20)
sq.dequeue()
sq.enqueue(30)
sq.dequeue()
sq.dequeue()
sq.dequeue() # Empty
print(f"Front element: {sq.front()}")
print("\n")

print("--- Circular Queue ---")
cq = CircularQueue(5)
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)
cq.dequeue()
cq.enqueue(4)
cq.enqueue(5)
cq.enqueue(6) # Full
cq.dequeue()
cq.dequeue()
cq.enqueue(7)
cq.enqueue(8)
cq.dequeue()
cq.dequeue()
cq.dequeue()
cq.dequeue() # Empty
print("\n")

print("--- General Tree (N-ary Tree) ---")
root = TreeNode("A")
b = add_child_to_tree(root, "B")
c = add_child_to_tree(root, "C")
d = add_child_to_tree(root, "D")

e = add_child_to_tree(b, "E")
f = add_child_to_tree(b, "F")

g = add_child_to_tree(c, "G")

h = add_child_to_tree(d, "H")
i = add_child_to_tree(d, "I")

print("Tree Structure:")
visualize_tree(root)
print("-" * 30)
print("\n")

print("--- Min-Heap (using heapq) ---")
min_heap = []
print("Inserting elements:")
elements_to_add_min = [30, 10, 50, 5, 20, 40]
for elem in elements_to_add_min:
    heapq.heappush(min_heap, elem)
    print(f"  Inserted {elem}. Heap: {min_heap}")
print("Final Heap Array (Min-Heap property ensured):", min_heap)
print("-" * 30)

print("Extracting min elements:")
while min_heap:
    min_val = heapq.heappop(min_heap)
    print(f"  Extracted min: {min_val}. Heap: {min_heap}")
print("Heap after all extractions:", min_heap)
print("-" * 30)

print("\n--- Max-Heap (conceptual, using heapq with negation) ---")
max_heap = []
elements_to_add_max = [30, 10, 50, 5, 20, 40]
print("Inserting elements (as negatives to simulate max-heap):")
for elem in elements_to_add_max:
    heapq.heappush(max_heap, -elem)
    print(f"  Inserted {elem}. Internal Heap: {max_heap}")
print("Final Internal Heap Array:", max_heap)
print("-" * 30)

print("Extracting max elements:")
while max_heap:
    max_val = -heapq.heappop(max_heap)
    print(f"  Extracted max: {max_val}. Internal Heap: {max_heap}")
print("Internal Heap after all extractions:", max_heap)
print("-" * 30)
print("\n")

print("--- Trie (Prefix Tree) ---")
trie = Trie()
trie.insert("apple")
trie.insert("apricot")
trie.insert("apply")
trie.insert("banana")
trie.search("apple")
trie.search("app")
trie.search("orange")
trie.starts_with("app")
trie.starts_with("ban")
trie.starts_with("grape")
print("\n")

print("--- Undirected, Unweighted Graph ---")
g_undirected = Graph(is_directed=False)
g_undirected.add_vertex("A")
g_undirected.add_vertex("B")
g_undirected.add_vertex("C")
g_undirected.add_edge("A", "B")
g_undirected.add_edge("B", "C")
g_undirected.add_edge("A", "C")
g_undirected.remove_edge("B", "C")
g_undirected.remove_vertex("A")
print("\n")

print("--- Directed, Weighted Graph ---")
g_directed = Graph(is_directed=True)
g_directed.add_edge("X", "Y", 5)
g_directed.add_edge("X", "Z", 2)
g_directed.add_edge("Y", "Z", 3)
g_directed.add_edge("Z", "X", 1)
g_directed.remove_edge("X", "Z")
print("\n")

print("--- Disjoint Set (Union-Find) ---")
elements = ['A', 'B', 'C', 'D', 'E', 'F']
ds = DisjointSet(elements)
ds.union('A', 'B')
ds.union('C', 'D')
ds.union('E', 'F')
ds.union('B', 'D')
ds.union('A', 'C')
ds.find('F')
ds.find('A')
ds.union('D', 'F')
print("\n")

print("--- Bloom Filter ---")
bf = BloomFilter(20, 3)
bf.add("apple")
bf.add("banana")
bf.contains("apple")
bf.contains("grape")
bf.contains("orange")
print("\n")

print("--- Skip List ---")
sl = SkipList(max_level=4, p=0.5)
sl.insert(30)
sl.insert(10)
sl.insert(50)
sl.insert(20)
sl.insert(45)
sl.insert(5)
sl.search(20)
sl.search(100)
sl.delete(10)
sl.delete(50)
sl.delete(100)
sl.insert(15)
sl.search(15)
print("\n")

print("--- LRU Cache ---")
lru = LRUCache(2)
lru.put(1, 10)
lru.put(2, 20)
lru.get(1)
lru.put(3, 30)
lru.get(2)
lru.put(4, 40)
lru.get(1)
lru.get(3)
lru.put(2, 200)
lru.get(4)
print("\n")

print("--- Map / Dictionary (Python dict) ---")
my_dict = {}
print(f"Initial: {my_dict}")
my_dict["apple"] = 5
print(f"Added 'apple': {my_dict}")
my_dict["banana"] = 2
print(f"Added 'banana': {my_dict}")
print(f"Value of 'apple': {my_dict.get('apple')}")
my_dict["apple"] = 7
print(f"Updated 'apple': {my_dict}")
del my_dict["banana"]
print(f"Deleted 'banana': {my_dict}")
print(f"'orange' in dict? {'orange' in my_dict}")
print("-" * 40)
print("\n")

print("--- Set (Python set) ---")
my_set = set()
print(f"Initial: {my_set}")
my_set.add(10)
print(f"Added 10: {my_set}")
my_set.add(20)
my_set.add(10)
print(f"Added 20, 10 (again): {my_set}")
my_set.remove(20)
print(f"Removed 20: {my_set}")
print(f"10 in set? {10 in my_set}")
print("-" * 40)

print("\n--- Multiset / Bag (Python collections.Counter) ---")
from collections import Counter
my_multiset = Counter()
print(f"Initial: {my_multiset}")
my_multiset.update([1, 2, 3, 2, 1, 4])
print(f"Added elements: {my_multiset}")
my_multiset.update([1])
print(f"Added another 1: {my_multiset}")
print(f"Count of 2: {my_multiset[2]}")
my_multiset.subtract([1, 2])
print(f"Subtracted one 1 and one 2: {my_multiset}")
print("-" * 40)
print("\n")

print("=" * 50)
print("--- END OF DEMONSTRATION ---")
print("=" * 50)
