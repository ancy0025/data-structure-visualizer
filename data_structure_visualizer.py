import streamlit as st
import random
import collections # For deque and defaultdict
import heapq # For heap (Priority Queue)
import hashlib # For Bloom Filter

# --- Configuration ---
# st.set_page_config() must be the first Streamlit command called.
st.set_page_config(page_title="Comprehensive Data Structure Explorer", layout="wide")

# --- Helper Functions and Classes for Data Structures ---

# --- Linear Data Structures ---

# Array (Python List) - Using built-in list directly
# Stack (LIFO) - Using built-in list directly (append/pop)
# Queue (FIFO) - Using collections.deque directly (append/popleft)
# Deque (Double-Ended Queue) - Using collections.deque directly

# Singly Linked List
class SLLNode:
    def __init__(self, value):
        self.value = value
        self.next = None

    def __repr__(self):
        return f"Node({self.value})"

class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def append(self, value):
        new_node = SLLNode(value)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1

    def prepend(self, value):
        new_node = SLLNode(value)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def delete_by_value(self, value):
        if self.head is None:
            return False

        if self.head.value == value:
            self.head = self.head.next
            self.size -= 1
            return True

        current = self.head
        prev = None
        while current and current.value != value:
            prev = current
            current = current.next

        if current is None: # Value not found
            return False
        
        prev.next = current.next
        self.size -= 1
        return True

    def to_display_string(self):
        if not self.head:
            return "(empty)"
        elements = []
        current = self.head
        while current:
            elements.append(str(current.value))
            current = current.next
        return " -> ".join(elements)

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

    def prepend(self, value):
        new_node = DoublyLinkedListNode(value)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1

    def delete_by_value(self, value):
        if not self.head:
            return False

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
                break
            current = current.next
        return found

    def to_display_string(self):
        if not self.head:
            return "(empty)"
        elements = []
        current = self.head
        while current:
            elements.append(str(current.value))
            current = current.next
        return f"Head <-> {' <-> '.join(elements)} <-> Tail"

# Circular Linked List
class CircularLinkedListNode:
    def __init__(self, value):
        self.value = value
        self.next = None

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

    def delete_by_value(self, value):
        if not self.head:
            return False

        if self.head.value == value and self.head.next == self.head:
            self.head = None
            self.size -= 1
            return True

        prev = None
        current = self.head
        start_node = self.head # To prevent infinite loop in case value not found
        
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
                break
            prev = current
            current = current.next
            if current == start_node: # Traversed full circle and didn't find
                break
        return found

    def to_display_string(self):
        if not self.head:
            return "(empty)"
        elements = []
        current = self.head
        start_node = self.head
        while True:
            elements.append(str(current.value))
            current = current.next
            if current == start_node:
                break
        return f"{' -> '.join(elements)} -> ... (back to {self.head.value})"

# Circular Queue (Array-based)
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def enqueue(self, item):
        if self.size == self.capacity:
            return False # Queue is full
        self.queue[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1
        return True

    def dequeue(self):
        if self.is_empty():
            return None
        item = self.queue[self.head]
        self.queue[self.head] = None # Clear the slot
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return item

    def front(self):
        if self.is_empty():
            return None
        return self.queue[self.head]

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity

    def to_display_string(self):
        if self.is_empty():
            return "(empty)"
        display_arr = []
        for i in range(self.size):
            idx = (self.head + i) % self.capacity
            display_arr.append(str(self.queue[idx]))
        return f"Front ({self.queue[self.head]}) <-- {' <-- '.join(display_arr)} <-- Rear ({self.queue[(self.tail - 1 + self.capacity) % self.capacity]})"

# --- Non-Linear Data Structures ---

# Binary Search Tree (BST)
class BSTNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __repr__(self):
        return f"Node({self.value})"

def insert_bst_node(root, value):
    if root is None:
        return BSTNode(value)
    if value < root.value:
        root.left = insert_bst_node(root.left, value)
    else: # Allow duplicates to go to the right
        root.right = insert_bst_node(root.right, value)
    return root

def search_bst_node(root, value, path_trace=None):
    if path_trace is None:
        path_trace = []
    
    if root is None:
        path_trace.append(f"Value {value} not found.")
        return False, path_trace
    
    path_trace.append(f"Visiting node: {root.value}")
    if value == root.value:
        path_trace.append(f"Value {value} found!")
        return True, path_trace
    elif value < root.value:
        path_trace.append(f"Comparing {value} < {root.value}. Going left.")
        return search_bst_node(root.left, value, path_trace)
    else:
        path_trace.append(f"Comparing {value} >= {root.value}. Going right.")
        return search_bst_node(root.right, value, path_trace)

def find_min_node(node):
    current = node
    while current.left is not None:
        current = current.left
    return current

def delete_bst_node(root, value):
    if root is None:
        return root

    if value < root.value:
        root.left = delete_bst_node(root.left, value)
    elif value > root.value:
        root.right = delete_bst_node(root.right, value)
    else: # Node to be deleted found
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        
        temp = find_min_node(root.right)
        root.value = temp.value
        root.right = delete_bst_node(root.right, temp.value)
    return root

def bst_to_json(node):
    if node is None:
        return None
    return {"value": node.value, "left": bst_to_json(node.left), "right": bst_to_json(node.right)}

def display_bst_node_text(node, prefix="", is_left=None, output_lines=None):
    if output_lines is None:
        output_lines = []

    if node is not None:
        connector = ""
        if is_left is True:
            connector = "├── L: "
        elif is_left is False:
            connector = "└── R: "
        elif is_left is None:
            connector = "Root: "

        output_lines.append(f"{prefix}{connector}**{node.value}**")
        
        new_prefix = prefix + ("│    " if is_left is not False else "     ") 

        if node.left:
            display_bst_node_text(node.left, new_prefix, True, output_lines)
        else:
            output_lines.append(f"{new_prefix}├── L: (null)")

        if node.right:
            display_bst_node_text(node.right, new_prefix, False, output_lines)
        else:
            output_lines.append(f"{new_prefix}└── R: (null)")
    return output_lines

# Graph (Adjacency List)
class Graph:
    def __init__(self, is_directed=False):
        self.graph = collections.defaultdict(list) # Adjacency list
        self.is_directed = is_directed
        self.vertices = set()

    def add_vertex(self, vertex):
        self.vertices.add(vertex)
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, u, v, weight=1):
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append((v, weight))
        if not self.is_directed:
            self.graph[v].append((u, weight))

    def remove_vertex(self, vertex):
        if vertex not in self.vertices:
            return False

        self.vertices.discard(vertex)
        for vtx in list(self.graph.keys()):
            self.graph[vtx] = [(neighbor, w) for neighbor, w in self.graph[vtx] if neighbor != vertex]
        if vertex in self.graph:
            del self.graph[vertex]
        return True

    def remove_edge(self, u, v):
        if u not in self.graph or v not in self.vertices:
            return False

        initial_len_u = len(self.graph[u])
        self.graph[u] = [(neighbor, w) for neighbor, w in self.graph[u] if neighbor != v]
        
        removed_u = len(self.graph[u]) < initial_len_u

        removed_v = False
        if not self.is_directed:
            initial_len_v = len(self.graph[v])
            self.graph[v] = [(neighbor, w) for neighbor, w in self.graph[v] if neighbor != u]
            removed_v = len(self.graph[v]) < initial_len_v
        
        return removed_u or removed_v

    def to_display_string(self):
        if not self.vertices:
            return "(empty)"
        output_lines = []
        sorted_vertices = sorted(list(self.vertices))
        for vertex in sorted_vertices:
            neighbors = self.graph.get(vertex, [])
            formatted_neighbors = []
            for neighbor, weight in sorted(neighbors):
                if weight == 1:
                    formatted_neighbors.append(str(neighbor))
                else:
                    formatted_neighbors.append(f"{neighbor}({weight})")
            output_lines.append(f"  {vertex}: [{', '.join(formatted_neighbors)}]")
        return "\n".join(output_lines)

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

    def search(self, word):
        node = self.root
        path = []
        for char in word:
            if char not in node.children:
                return False, path
            path.append(char)
            node = node.children[char]
        return node.is_end_of_word, path

    def starts_with(self, prefix):
        node = self.root
        path = []
        for char in prefix:
            if char not in node.children:
                return False, path
            path.append(char)
            node = node.children[char]
        return True, path

    def _display_trie(self, node, prefix="", level=0, output_lines=None):
        if output_lines is None:
            output_lines = []
        indent = "  " * level
        status = "(End of Word)" if node.is_end_of_word else ""
        output_lines.append(f"{indent}{prefix}{status}")
        for char, child_node in sorted(node.children.items()):
            self._display_trie(child_node, f"[{char}]", level + 1, output_lines)
        return output_lines

    def to_display_string(self):
        return "\n".join(self._display_trie(self.root, "ROOT"))


# --- Hashing Data Structures ---

# Hash Table (with Chaining)
def hash_function(key, table_size):
    return sum(ord(c) for c in str(key)) % table_size

def insert_hash_table(table, key, value, table_size):
    bucket_index = hash_function(key, table_size)
    
    for i, (k, v) in enumerate(table[bucket_index]):
        if k == key:
            table[bucket_index][i] = (key, value)
            return f"Key '{key}' updated in bucket {bucket_index}."
    
    table[bucket_index].append((key, value))
    return f"Key '{key}' inserted into bucket {bucket_index}."

def lookup_hash_table(table, key, table_size):
    bucket_index = hash_function(key, table_size)
    path_trace = [f"Hashing key '{key}' to bucket index: {bucket_index}"]
    
    if not table[bucket_index]:
        path_trace.append(f"Bucket {bucket_index} is empty. Key not found.")
        return None, path_trace

    path_trace.append(f"Searching in bucket {bucket_index}: {table[bucket_index]}")
    for k, v in table[bucket_index]:
        if k == key:
            path_trace.append(f"Key '{key}' found with value '{v}'.")
            return v, path_trace
    
    path_trace.append(f"Key '{key}' not found in bucket {bucket_index}.")
    return None, path_trace

def delete_hash_table(table, key, table_size):
    bucket_index = hash_function(key, table_size)
    original_bucket_len = len(table[bucket_index])
    
    table[bucket_index] = [(k, v) for k, v in table[bucket_index] if k != key]
    
    if len(table[bucket_index]) < original_bucket_len:
        return f"Key '{key}' deleted from bucket {bucket_index}."
    else:
        return f"Key '{key}' not found in bucket {bucket_index} for deletion."

# Hash Set (with Chaining)
def hash_set_add(table, key, table_size):
    bucket_index = hash_function(key, table_size)
    if key not in table[bucket_index]:
        table[bucket_index].append(key)
        return f"Key '{key}' added to bucket {bucket_index}."
    return f"Key '{key}' already exists in bucket {bucket_index}."

def hash_set_remove(table, key, table_size):
    bucket_index = hash_function(key, table_size)
    original_bucket_len = len(table[bucket_index])
    table[bucket_index] = [k for k in table[bucket_index] if k != key]
    if len(table[bucket_index]) < original_bucket_len:
        return f"Key '{key}' removed from bucket {bucket_index}."
    return f"Key '{key}' not found in bucket {bucket_index} for removal."

def hash_set_contains(table, key, table_size):
    bucket_index = hash_function(key, table_size)
    if key in table[bucket_index]:
        return True, [f"Key '{key}' found in bucket {bucket_index}."]
    return False, [f"Key '{key}' not found in bucket {bucket_index}."]*(1 if not table[bucket_index] else 0) + [f"Searched bucket {bucket_index}: {table[bucket_index]}"]

# --- Specialized or Advanced Data Structures ---

# Disjoint Set / Union-Find
class DisjointSet:
    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, i):
        path_trace = []
        current = i
        while current != self.parent[current]:
            path_trace.append(current)
            current = self.parent[current]
        
        root = current
        for node in path_trace:
            self.parent[node] = root # Path compression
        return root, path_trace

    def union(self, i, j):
        root_i, _ = self.find(i)
        root_j, _ = self.find(j)

        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                return True, f"Union: {root_i} attached under {root_j} (by rank)."
            elif self.rank[root_j] < self.rank[root_i]:
                self.parent[root_j] = root_i
                return True, f"Union: {root_j} attached under {root_i} (by rank)."
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
                return True, f"Union: {root_j} attached under {root_i} (equal rank, {root_i} rank incremented)."
        else:
            return False, f"Union: '{i}' and '{j}' are already in the same set."

    def get_sets(self):
        sets = collections.defaultdict(list)
        for elem in self.parent:
            root, _ = self.find(elem) # Use find to get the compressed root
            sets[root].append(elem)
        return {rep: sorted(elems) for rep, elems in sets.items()}

# Bloom Filter
class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size

    def _hash(self, item, seed):
        h = hashlib.md5(f"{item}-{seed}".encode()).hexdigest()
        return int(h, 16) % self.size

    def add(self, item):
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            self.bit_array[index] = 1

    def contains(self, item):
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True

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

    def _random_level(self):
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def insert(self, value):
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
            return True
        return False

    def search(self, value):
        current = self.head
        path = []
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].value < value:
                path.append(f"Level {i}: {current.value} -> {current.next[i].value}")
                current = current.next[i]
            path.append(f"Level {i}: Current at {current.value}")
        
        current = current.next[0]
        
        if current and current.value == value:
            return True, path
        return False, path

    def delete(self, value):
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
            return True
        return False

    def to_display_string(self):
        output_lines = []
        if self.head.next[0] is None:
            return "(empty)"
        else:
            for i in range(self.level, -1, -1):
                level_str = f"Level {i:2d}: Head "
                current = self.head.next[i]
                while current:
                    level_str += f"-> {current.value} "
                    current = current.next[i]
                output_lines.append(level_str + "-> None")
        return "\n".join(output_lines)

# LRU Cache
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
        self.head = LRUNode(0, 0) # Sentinel head
        self.tail = LRUNode(0, 0) # Sentinel tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = next_node

    def _move_to_front(self, node):
        self._remove_node(node)
        self._add_node(node)

    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._move_to_front(node)
            return node.value
        return -1

    def put(self, key, value):
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
        else:
            new_node = LRUNode(key, value)
            self.cache[key] = new_node
            self._add_node(new_node)

            if len(self.cache) > self.capacity:
                lru_node = self.tail.prev
                self._remove_node(lru_node)
                del self.cache[lru_node.key]
                return f"Cache full. Removed LRU: ({lru_node.key}:{lru_node.value})."
        return None # No eviction occurred

    def to_display_string(self):
        items = []
        current = self.head.next
        while current != self.tail:
            items.append(f"({current.key}:{current.value})")
            current = current.next
        return f"MRU -> LRU: {' <-> '.join(items)}"

# --- Session State Initialization ---
def initialize_session_state():
    if 'my_list' not in st.session_state: st.session_state.my_list = []
    if 'my_singly_linked_list' not in st.session_state: st.session_state.my_singly_linked_list = SinglyLinkedList()
    if 'my_doubly_linked_list' not in st.session_state: st.session_state.my_doubly_linked_list = DoublyLinkedList()
    if 'my_circular_linked_list' not in st.session_state: st.session_state.my_circular_linked_list = CircularSinglyLinkedList()
    if 'my_stack' not in st.session_state: st.session_state.my_stack = [] # Python list as stack
    if 'my_simple_queue' not in st.session_state: st.session_state.my_simple_queue = collections.deque() # Python deque as simple queue
    if 'my_circular_queue_capacity' not in st.session_state: st.session_state.my_circular_queue_capacity = 5
    if 'my_circular_queue' not in st.session_state: st.session_state.my_circular_queue = CircularQueue(st.session_state.my_circular_queue_capacity)
    if 'my_priority_queue' not in st.session_state: st.session_state.my_priority_queue = [] # Python heapq as min-heap

    if 'my_bst' not in st.session_state: st.session_state.my_bst = None
    if 'my_graph_directed' not in st.session_state: st.session_state.my_graph_directed = Graph(is_directed=True)
    if 'my_graph_undirected' not in st.session_state: st.session_state.my_graph_undirected = Graph(is_directed=False)
    if 'my_trie' not in st.session_state: st.session_state.my_trie = Trie()

    if 'hash_table_size' not in st.session_state: st.session_state.hash_table_size = 10
    if 'my_hash_table' not in st.session_state: st.session_state.my_hash_table = [[] for _ in range(st.session_state.hash_table_size)]
    if 'hash_set_size' not in st.session_state: st.session_state.hash_set_size = 10
    if 'my_hash_set' not in st.session_state: st.session_state.my_hash_set = [[] for _ in range(st.session_state.hash_set_size)]

    if 'disjoint_set_elements' not in st.session_state: st.session_state.disjoint_set_elements = ["A", "B", "C", "D", "E", "F", "G"]
    if 'disjoint_set' not in st.session_state: st.session_state.disjoint_set = DisjointSet(st.session_state.disjoint_set_elements)
    if 'bloom_filter_size' not in st.session_state: st.session_state.bloom_filter_size = 30
    if 'bloom_filter_hashes' not in st.session_state: st.session_state.bloom_filter_hashes = 3
    if 'bloom_filter' not in st.session_state: st.session_state.bloom_filter = BloomFilter(st.session_state.bloom_filter_size, st.session_state.bloom_filter_hashes)
    if 'skip_list_max_level' not in st.session_state: st.session_state.skip_list_max_level = 4
    if 'skip_list_p' not in st.session_state: st.session_state.skip_list_p = 0.5
    if 'my_skip_list' not in st.session_state: st.session_state.my_skip_list = SkipList(st.session_state.skip_list_max_level, st.session_state.skip_list_p)
    if 'lru_cache_capacity' not in st.session_state: st.session_state.lru_cache_capacity = 3
    if 'my_lru_cache' not in st.session_state: st.session_state.my_lru_cache = LRUCache(st.session_state.lru_cache_capacity)

initialize_session_state()

# --- Reset All Data Structures ---
def reset_all_data_structures():
    st.session_state.my_list = []
    st.session_state.my_singly_linked_list = SinglyLinkedList()
    st.session_state.my_doubly_linked_list = DoublyLinkedList()
    st.session_state.my_circular_linked_list = CircularSinglyLinkedList()
    st.session_state.my_stack = []
    st.session_state.my_simple_queue = collections.deque()
    st.session_state.my_circular_queue = CircularQueue(st.session_state.my_circular_queue_capacity)
    st.session_state.my_priority_queue = []
    st.session_state.my_bst = None
    st.session_state.my_graph_directed = Graph(is_directed=True)
    st.session_state.my_graph_undirected = Graph(is_directed=False)
    st.session_state.my_trie = Trie()
    st.session_state.my_hash_table = [[] for _ in range(st.session_state.hash_table_size)]
    st.session_state.my_hash_set = [[] for _ in range(st.session_state.hash_set_size)]
    st.session_state.disjoint_set = DisjointSet(st.session_state.disjoint_set_elements)
    st.session_state.bloom_filter = BloomFilter(st.session_state.bloom_filter_size, st.session_state.bloom_filter_hashes)
    st.session_state.my_skip_list = SkipList(st.session_state.skip_list_max_level, st.session_state.skip_list_p)
    st.session_state.my_lru_cache = LRUCache(st.session_state.lru_cache_capacity)
    st.success("All data structures reset!")

st.sidebar.button("Reset All Data Structures", on_click=reset_all_data_structures)
st.sidebar.markdown("---")


st.title("📚 Comprehensive Interactive Data Structure Explorer")
st.markdown("Explore the behavior, operations, and complexities of various fundamental and advanced data structures.")

# --- Navigation Menu ---
ds_categories = {
    "🔹 Linear Data Structures": [
        "Array", "Linked List", "Singly Linked List", "Doubly Linked List", "Circular Linked List",
        "Stack (LIFO)", "Queue (FIFO)", "Simple Queue", "Circular Queue", "Priority Queue", "Deque (Double-Ended Queue)"
    ],
    "🔸 Non-Linear Data Structures": [
        "Tree", "Binary Tree", "Binary Search Tree (BST)", "AVL Tree", "Red-Black Tree",
        "B-Tree / B+ Tree", "Heap (Min-Heap, Max-Heap)", "Segment Tree", "Fenwick Tree (Binary Indexed Tree)",
        "Trie (Prefix Tree)", "Graph", "Directed / Undirected Graph", "Weighted / Unweighted Graph", "Adjacency Matrix / Adjacency List", "Tree as a special type of graph"
    ],
    "� Hashing Data Structures": [
        "Hash Table", "Hash Map / Dictionary", "Hash Set"
    ],
    "🔸 Specialized or Advanced Data Structures": [
        "Disjoint Set / Union-Find", "Bloom Filter", "Skip List", "Suffix Tree / Suffix Array",
        "K-D Tree (K-Dimensional Tree)", "Quad Tree / Octree (for spatial partitioning)",
        "LRU Cache (Least Recently Used)", "Interval Tree", "Treap (Tree + Heap)"
    ],
    "🔹 Other Abstract Data Types (ADTs)": [
        "Map / Dictionary (ADT)", "Set / Multiset (ADT)", "Multimap (ADT)", "Bag / Multibag (ADT)"
    ]
}

# Flatten the list for sidebar radio buttons, including category headers
all_ds_options = []
for category, ds_list in ds_categories.items():
    all_ds_options.append(category) # Add category header as a selectable option
    all_ds_options.extend(ds_list)

selected_ds_option = st.sidebar.radio(
    "Select a Data Structure:",
    all_ds_options,
    index=0 # Default to the first option
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Note on Visualization:**
Some advanced data structures (e.g., AVL Trees, Red-Black Trees, B-Trees, complex Graphs) require specialized graphical visualization and complex algorithmic implementations that are beyond the scope of a basic text-based Streamlit app. This visualizer focuses on conceptual understanding and interactive textual representation where feasible.
""")

# --- Display Sections for Each Data Structure ---

# Helper for complexity display
def display_complexity(space_complexity, time_complexities):
    st.markdown(f"**Space Complexity:** ${space_complexity}$")
    st.markdown("**Time Complexity:**")
    for op, complexity in time_complexities.items():
        st.markdown(f"  * **{op}:** ${complexity}$")

# Helper for random value generation
def get_random_value(prefix="item"):
    return f"{prefix}_{random.randint(1, 999)}"

# --- Linear Data Structures ---

if selected_ds_option == "Array":
    st.header("Array (Python List)")
    st.markdown("A dynamic array that can grow and shrink. Elements are ordered and indexed.")
    display_complexity(
        "$O(N)$",
        {
            "Access/Lookup by Index": "O(1)",
            "Append": "O(1) \\text{ amortized}",
            "Insert (at index)": "O(N)",
            "Delete (by index/value)": "O(N)",
            "Search (linear)": "O(N)"
        }
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        new_val = st.text_input("Value to Add", value=get_random_value("val"), key="list_add_input")
        if st.button("Append"):
            st.session_state.my_list.append(new_val)
            st.success(f"Appended '{new_val}'.")
        if st.button("Remove Last"):
            if st.session_state.my_list:
                removed = st.session_state.my_list.pop()
                st.info(f"Removed '{removed}'.")
            else:
                st.warning("List is empty.")
        if st.button("Clear List"):
            st.session_state.my_list = []
            st.error("List cleared.")
    with col2:
        st.subheader("Current List:")
        if st.session_state.my_list:
            st.json(st.session_state.my_list)
            st.write(f"**Size:** `{len(st.session_state.my_list)}`")
        else:
            st.info("The list is currently empty.")

elif selected_ds_option == "Linked List":
    st.header("Linked List (General Concept)")
    st.markdown("A sequence of nodes where each node contains data and a reference (link) to the next node. This section introduces the general idea; specific types are explored below.")
    st.markdown("Unlike arrays, elements are not stored contiguously in memory. They are connected via pointers.")
    display_complexity(
        "$O(N)$",
        {
            "Access/Lookup by Index": "O(N)",
            "Insertion (at head/tail/middle)": "O(1) \\text{ or } O(N)",
            "Deletion (by value/index)": "O(N)"
        }
    )
    st.info("See 'Singly Linked List' and 'Doubly Linked List' for interactive examples.")

elif selected_ds_option == "Singly Linked List":
    st.header("Singly Linked List")
    st.markdown("A linked list where each node points only to the next node in the sequence. Traversal is only forward.")
    display_complexity(
        "$O(N)$",
        {
            "Access/Lookup by Index": "O(N)",
            "Append (to Tail)": "O(N)",
            "Prepend (to Head)": "O(1)",
            "Delete (by value)": "O(N)"
        }
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        new_val = st.text_input("Value to Add", value=get_random_value("sll_val"), key="sll_add_input")
        if st.button("Append (Add to Tail)"):
            st.session_state.my_singly_linked_list.append(new_val)
            st.success(f"Appended '{new_val}'.")
        if st.button("Prepend (Add to Head)"):
            st.session_state.my_singly_linked_list.prepend(new_val)
            st.success(f"Prepended '{new_val}'.")
        
        delete_val = st.text_input("Value to Delete", key="sll_delete_input")
        if st.button("Delete by Value"):
            if st.session_state.my_singly_linked_list.delete_by_value(delete_val):
                st.info(f"Deleted '{delete_val}'.")
            else:
                st.error(f"'{delete_val}' not found.")
        if st.button("Clear Singly Linked List"):
            st.session_state.my_singly_linked_list = SinglyLinkedList()
            st.error("Singly Linked List cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Singly Linked List:")
        st.code(st.session_state.my_singly_linked_list.to_display_string())
        st.write(f"**Size:** `{st.session_state.my_singly_linked_list.size}`")

elif selected_ds_option == "Doubly Linked List":
    st.header("Doubly Linked List")
    st.markdown("Each node contains data, a pointer to the next node, and a pointer to the previous node. Allows traversal in both directions.")
    display_complexity(
        "$O(N)$",
        {
            "Access/Lookup by Index": "O(N)",
            "Append (to Tail)": "O(1)",
            "Prepend (to Head)": "O(1)",
            "Delete (by value)": "O(N)"
        }
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        new_val = st.text_input("Value to Add", value=get_random_value("dll_val"), key="dll_add_input")
        if st.button("Append (Add to Tail)"):
            st.session_state.my_doubly_linked_list.append(new_val)
            st.success(f"Appended '{new_val}'.")
        if st.button("Prepend (Add to Head)"):
            st.session_state.my_doubly_linked_list.prepend(new_val)
            st.success(f"Prepended '{new_val}'.")
        
        delete_val = st.text_input("Value to Delete", key="dll_delete_input")
        if st.button("Delete by Value"):
            if st.session_state.my_doubly_linked_list.delete_by_value(delete_val):
                st.info(f"Deleted '{delete_val}'.")
            else:
                st.error(f"'{delete_val}' not found.")
        if st.button("Clear Doubly Linked List"):
            st.session_state.my_doubly_linked_list = DoublyLinkedList()
            st.error("Doubly Linked List cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Doubly Linked List:")
        st.code(st.session_state.my_doubly_linked_list.to_display_string())
        st.write(f"**Size:** `{st.session_state.my_doubly_linked_list.size}`")

elif selected_ds_option == "Circular Linked List":
    st.header("Circular Linked List")
    st.markdown("A linked list where the last node points back to the first node (head), forming a circle.")
    display_complexity(
        "$O(N)$",
        {
            "Access/Lookup by Index": "O(N)",
            "Append (to Tail)": "O(N)",
            "Prepend (to Head)": "O(1)",
            "Delete (by value)": "O(N)"
        }
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        new_val = st.text_input("Value to Add", value=get_random_value("cll_val"), key="cll_add_input")
        if st.button("Append (Add to Tail)"):
            st.session_state.my_circular_linked_list.append(new_val)
            st.success(f"Appended '{new_val}'.")
        if st.button("Prepend (Add to Head)"):
            st.session_state.my_circular_linked_list.prepend(new_val)
            st.success(f"Prepended '{new_val}'.")
        
        delete_val = st.text_input("Value to Delete", key="cll_delete_input")
        if st.button("Delete by Value"):
            if st.session_state.my_circular_linked_list.delete_by_value(delete_val):
                st.info(f"Deleted '{delete_val}'.")
            else:
                st.error(f"'{delete_val}' not found.")
        if st.button("Clear Circular Linked List"):
            st.session_state.my_circular_linked_list = CircularSinglyLinkedList()
            st.error("Circular Linked List cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Circular Linked List:")
        st.code(st.session_state.my_circular_linked_list.to_display_string())
        st.write(f"**Size:** `{st.session_state.my_circular_linked_list.size}`")

elif selected_ds_option == "Stack (LIFO)":
    st.header("Stack (LIFO - Last In, First Out)")
    st.markdown("A collection where elements are added and removed from the same end (the 'top'). Think of a stack of plates.")
    display_complexity(
        "$O(N)$",
        {
            "Push": "O(1)",
            "Pop": "O(1)",
            "Peek/Top": "O(1)",
            "IsEmpty": "O(1)"
        }
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        new_val = st.text_input("Value to Push", value=get_random_value("stack_val"), key="stack_push_input")
        if st.button("Push (Add to Top)"):
            st.session_state.my_stack.append(new_val)
            st.success(f"Pushed '{new_val}'.")
        if st.button("Pop (Remove from Top)"):
            if st.session_state.my_stack:
                popped_val = st.session_state.my_stack.pop()
                st.info(f"Popped '{popped_val}'.")
            else:
                st.warning("Stack is empty!")
        if st.button("Peek (View Top)"):
            if st.session_state.my_stack:
                st.info(f"Top element: **{st.session_state.my_stack[-1]}**")
            else:
                st.warning("Stack is empty!")
        if st.button("Clear Stack"):
            st.session_state.my_stack = []
            st.error("Stack cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Stack State (Top is at the end):")
        if st.session_state.my_stack:
            stack_display = ["```"]
            for item in reversed(st.session_state.my_stack):
                stack_display.append(f"| {item:<15} |")
                stack_display.append("-------------------")
            stack_display.append(f"| (BOTTOM)        |")
            stack_display.append("```")
            st.markdown("\n".join(stack_display))
            st.write(f"**Size:** `{len(st.session_state.my_stack)}`")
        else:
            st.info("The stack is currently empty.")

elif selected_ds_option == "Queue (FIFO)":
    st.header("Queue (FIFO - First In, First Out)")
    st.markdown("A collection where elements are added to one end ('rear') and removed from the other end ('front'). This section introduces the general idea; specific types are explored below.")
    display_complexity(
        "$O(N)$",
        {
            "Enqueue": "O(1) \\text{ or } O(N)",
            "Dequeue": "O(1) \\text{ or } O(N)",
            "Front/Peek": "O(1)",
            "IsEmpty": "O(1)"
        }
    )
    st.info("See 'Simple Queue' and 'Circular Queue' for interactive examples.")

elif selected_ds_option == "Simple Queue":
    st.header("Simple Queue (using `collections.deque`)")
    st.markdown("A basic queue implementation using Python's `collections.deque` for efficient $O(1)$ enqueue and dequeue operations.")
    display_complexity(
        "$O(N)$",
        {
            "Enqueue (append)": "O(1)",
            "Dequeue (popleft)": "O(1)",
            "Front/Peek": "O(1)",
            "IsEmpty": "O(1)"
        }
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        new_val = st.text_input("Value to Enqueue", value=get_random_value("queue_val"), key="simple_queue_enqueue_input")
        if st.button("Enqueue (Add to Rear)"):
            st.session_state.my_simple_queue.append(new_val)
            st.success(f"Enqueued '{new_val}'.")
        if st.button("Dequeue (Remove from Front)"):
            if st.session_state.my_simple_queue:
                dequeued_val = st.session_state.my_simple_queue.popleft()
                st.info(f"Dequeued '{dequeued_val}'.")
            else:
                st.warning("Queue is empty!")
        if st.button("Front (View Front)"):
            if st.session_state.my_simple_queue:
                st.info(f"Front element: **{st.session_state.my_simple_queue[0]}**")
            else:
                st.warning("Queue is empty!")
        if st.button("Clear Simple Queue"):
            st.session_state.my_simple_queue.clear()
            st.error("Simple Queue cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Simple Queue State (Front is on the left):")
        if st.session_state.my_simple_queue:
            queue_str = " <-- ".join(map(str, st.session_state.my_simple_queue))
            st.markdown(f"**FRONT** <-- `{queue_str}` <-- **REAR**")
            st.write(f"**Size:** `{len(st.session_state.my_simple_queue)}`")
        else:
            st.info("The simple queue is currently empty.")

elif selected_ds_option == "Circular Queue":
    st.header("Circular Queue")
    st.markdown("A queue implemented using a fixed-size array where the rear wraps around to the front. Efficient $O(1)$ enqueue/dequeue.")
    display_complexity(
        "$O(K)$", # K is capacity
        {
            "Enqueue": "O(1)",
            "Dequeue": "O(1)",
            "Front/Peek": "O(1)",
            "IsEmpty/IsFull": "O(1)"
        }
    )

    st.session_state.my_circular_queue_capacity = st.slider("Capacity", 3, 10, st.session_state.my_circular_queue_capacity, key="cq_capacity")
    if st.session_state.my_circular_queue.capacity != st.session_state.my_circular_queue_capacity:
        st.session_state.my_circular_queue = CircularQueue(st.session_state.my_circular_queue_capacity)
        st.info(f"Circular Queue re-initialized with capacity {st.session_state.my_circular_queue_capacity}.")
        st.rerun() # Rerun to update display

    col1, col2 = st.columns([1, 3])
    with col1:
        new_val = st.text_input("Value to Enqueue", value=get_random_value("cq_val"), key="cq_enqueue_input")
        if st.button("Enqueue"):
            if st.session_state.my_circular_queue.enqueue(new_val):
                st.success(f"Enqueued '{new_val}'.")
            else:
                st.error("Circular Queue is full!")
        if st.button("Dequeue"):
            dequeued_val = st.session_state.my_circular_queue.dequeue()
            if dequeued_val is not None:
                st.info(f"Dequeued '{dequeued_val}'.")
            else:
                st.warning("Circular Queue is empty!")
        if st.button("Front"):
            front_val = st.session_state.my_circular_queue.front()
            if front_val is not None:
                st.info(f"Front element: **{front_val}**")
            else:
                st.warning("Circular Queue is empty!")
        if st.button("Clear Circular Queue"):
            st.session_state.my_circular_queue = CircularQueue(st.session_state.my_circular_queue_capacity)
            st.error("Circular Queue cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Circular Queue State:")
        st.markdown(f"**Array Representation:** `{st.session_state.my_circular_queue.queue}`")
        st.markdown(f"**Conceptual View:** `{st.session_state.my_circular_queue.to_display_string()}`")
        st.write(f"**Size:** `{st.session_state.my_circular_queue.size}` / **Capacity:** `{st.session_state.my_circular_queue.capacity}`")
        st.write(f"**Head Index:** `{st.session_state.my_circular_queue.head}`")
        st.write(f"**Tail Index:** `{st.session_state.my_circular_queue.tail}`")
        st.write(f"**Is Empty:** `{st.session_state.my_circular_queue.is_empty()}`")
        st.write(f"**Is Full:** `{st.session_state.my_circular_queue.is_full()}`")

elif selected_ds_option == "Priority Queue":
    st.header("Priority Queue (Min-Heap Implementation)")
    st.markdown("A collection where elements are served based on priority. Here, lower numbers have higher priority (Min-Heap). Implemented using Python's `heapq` module.")
    display_complexity(
        "$O(N)$",
        {
            "Insert": "O(\\log N)",
            "Extract Min": "O(\\log N)",
            "Peek Min": "O(1)"
        }
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        new_val = st.number_input("Value to Insert (Priority)", value=random.randint(1, 100), key="pq_insert_input")
        if st.button("Insert (Enqueue)"):
            heapq.heappush(st.session_state.my_priority_queue, new_val)
            st.success(f"Inserted {new_val}.")
        if st.button("Extract Min (Dequeue)"):
            if st.session_state.my_priority_queue:
                extracted_val = heapq.heappop(st.session_state.my_priority_queue)
                st.info(f"Extracted minimum value: **{extracted_val}**.")
            else:
                st.warning("Priority Queue is empty!")
        if st.button("Peek Min"):
            if st.session_state.my_priority_queue:
                st.info(f"Minimum element (Peek): **{st.session_state.my_priority_queue[0]}**")
            else:
                st.warning("Priority Queue is empty!")
        if st.button("Clear Priority Queue"):
            st.session_state.my_priority_queue = []
            st.error("Priority Queue cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Priority Queue (Heap Array Representation):")
        if st.session_state.my_priority_queue:
            st.json(st.session_state.my_priority_queue)
            st.write(f"**Size:** `{len(st.session_state.my_priority_queue)}`")
            
            st.subheader("Conceptual Tree View (for Heap):")
            st.markdown("*(Note: This is the underlying array. The tree structure maintains heap property.)*")
            temp_heap = list(st.session_state.my_priority_queue)
            level = 0
            idx = 0
            heap_lines = ["```"]
            while idx < len(temp_heap):
                nodes_at_level = 2**level
                level_nodes = temp_heap[idx : idx + nodes_at_level]
                heap_lines.append(" ".join(map(str, level_nodes)).center(60))
                idx += nodes_at_level
                level += 1
            heap_lines.append("```")
            st.markdown("\n".join(heap_lines))
        else:
            st.info("The Priority Queue is currently empty.")

elif selected_ds_option == "Deque (Double-Ended Queue)":
    st.header("Deque (Double-Ended Queue)")
    st.markdown("A generalization of a queue and stack, allowing elements to be added or removed from both ends. Implemented using Python's `collections.deque`.")
    display_complexity(
        "$O(N)$",
        {
            "Append (to Right)": "O(1)",
            "Append Left (to Left)": "O(1)",
            "Pop (from Right)": "O(1)",
            "Pop Left (from Left)": "O(1)"
        }
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        new_val = st.text_input("Value to Add", value=get_random_value("deque_val"), key="deque_add_input")
        if st.button("Append (Add to Right)"):
            st.session_state.my_deque.append(new_val)
            st.success(f"Appended '{new_val}'.")
        if st.button("Append Left (Add to Left)"):
            st.session_state.my_deque.appendleft(new_val)
            st.success(f"Appended left '{new_val}'.")
        
        if st.button("Pop (Remove from Right)"):
            if st.session_state.my_deque:
                popped_val = st.session_state.my_deque.pop()
                st.info(f"Popped '{popped_val}' from right.")
            else:
                st.warning("Deque is empty!")
        if st.button("Pop Left (Remove from Left)"):
            if st.session_state.my_deque:
                popped_val = st.session_state.my_deque.popleft()
                st.info(f"Popped '{popped_val}' from left.")
            else:
                st.warning("Deque is empty!")
        if st.button("Clear Deque"):
            st.session_state.my_deque.clear()
            st.error("Deque cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Deque State (Left/Front on the left, Right/Rear on the right):")
        if st.session_state.my_deque:
            deque_str = " <-> ".join(map(str, st.session_state.my_deque))
            st.markdown(f"**LEFT** <-> `{deque_str}` <-> **RIGHT**")
            st.write(f"**Size:** `{len(st.session_state.my_deque)}`")
        else:
            st.info("The Deque is currently empty.")

# --- Non-Linear Data Structures ---

elif selected_ds_option == "Tree":
    st.header("Tree (General N-ary Tree)")
    st.markdown("A hierarchical data structure consisting of nodes connected by edges. Has a root node, child nodes, and parent nodes. No cycles. Each node can have zero or more children.")
    display_complexity(
        "$O(N)$",
        {
            "Traversal (BFS/DFS)": "O(V+E) \\text{ or } O(N)",
            "Insertion/Deletion": "O(N) \\text{ (depends on finding position)}"
        }
    )
    st.info("This is a conceptual representation. Interactive building of a general tree with arbitrary children is complex to visualize textually. See Binary Tree and BST for more structured examples.")
    
    st.subheader("Example Tree Structure:")
    # Re-create a static example for display
    class ExampleTreeNode: # Use a separate class to avoid conflicts with BSTNode
        def __init__(self, value):
            self.value = value
            self.children = []
    
    def add_child_to_example_tree(parent_node, child_value):
        child_node = ExampleTreeNode(child_value)
        parent_node.children.append(child_node)
        return child_node

    example_root = ExampleTreeNode("A")
    b = add_child_to_example_tree(example_root, "B")
    c = add_child_to_example_tree(example_root, "C")
    d = add_child_to_example_tree(example_root, "D")
    add_child_to_example_tree(b, "E")
    add_child_to_example_tree(b, "F")
    add_child_to_example_tree(c, "G")
    add_child_to_example_tree(d, "H")
    add_child_to_example_tree(d, "I")

    output_lines = []
    def _visualize_tree_static(node, level=0, prefix="", lines=None):
        if lines is None: lines = []
        if node is not None:
            indent = "    " * level
            lines.append(f"{indent}{prefix}{node.value}")
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                child_prefix = "└── " if is_last_child else "├── "
                _visualize_tree_static(child, level + 1, child_prefix, lines)
        return lines
    
    st.code("\n".join(_visualize_tree_static(example_root)))

elif selected_ds_option == "Binary Tree":
    st.header("Binary Tree")
    st.markdown("A tree data structure in which each node has at most two children, referred to as the left child and the right child.")
    display_complexity(
        "$O(N)$",
        {
            "Traversal (BFS/DFS)": "O(N)",
            "Insertion/Deletion": "O(N) \\text{ (can be } O(\\log N) \\text{ if balanced)}"
        }
    )
    st.info("This is a conceptual representation. Interactive building without specific rules (like BST) is complex to visualize textually. See Binary Search Tree for an interactive example.")
    st.subheader("Conceptual Binary Tree Example:")
    st.code("""
        10
       /  \\
      5    15
     / \\   / \\
    2   7 12  18
    """)

elif selected_ds_option == "Binary Search Tree (BST)":
    st.header("Binary Search Tree (BST)")
    st.markdown("A tree-based data structure where values in the left subtree are smaller, and values in the right subtree are larger (or equal).")
    display_complexity(
        "$O(N)$",
        {
            "Insert (Average)": "O(\\log N)",
            "Search (Average)": "O(\\log N)",
            "Delete (Average)": "O(\\log N)",
            "Insert (Worst - Skewed)": "O(N)",
            "Search (Worst - Skewed)": "O(N)",
            "Delete (Worst - Skewed)": "O(N)"
        }
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        new_val = st.number_input("Value to Insert", value=random.randint(1, 100), key="bst_insert_input")
        if st.button("Insert Node"):
            st.session_state.my_bst = insert_bst_node(st.session_state.my_bst, new_val)
            st.success(f"Inserted node with value {new_val}.")
        
        search_val = st.number_input("Value to Search", value=random.randint(1, 100), key="bst_search_input")
        if st.button("Search Node"):
            if st.session_state.my_bst:
                found, path = search_bst_node(st.session_state.my_bst, search_val)
                if found:
                    st.success(f"Search Result: Found {search_val}!")
                else:
                    st.warning(f"Search Result: {search_val} not found.")
                with st.expander("Search Path Details"):
                    for step in path:
                        st.write(step)
            else:
                st.warning("BST is empty.")

        delete_val = st.number_input("Value to Delete", value=random.randint(1, 100), key="bst_delete_input")
        if st.button("Delete Node"):
            original_bst_json = bst_to_json(st.session_state.my_bst)
            st.session_state.my_bst = delete_bst_node(st.session_state.my_bst, delete_val)
            if bst_to_json(st.session_state.my_bst) != original_bst_json:
                st.info(f"Deleted node with value {delete_val}.")
            else:
                st.warning(f"Node {delete_val} not found or couldn't be deleted.")

        if st.button("Clear BST"):
            st.session_state.my_bst = None
            st.error("Binary Search Tree cleared.")
            st.rerun()
    with col2:
        st.subheader("Current BST Structure (JSON representation):")
        if st.session_state.my_bst:
            st.json(bst_to_json(st.session_state.my_bst))
            st.subheader("Visualized Tree Structure (Simplified Text View):")
            for line in display_bst_node_text(st.session_state.my_bst):
                st.write(line)
        else:
            st.info("The Binary Search Tree is currently empty. Insert a node to start building it!")

elif selected_ds_option == "AVL Tree":
    st.header("AVL Tree")
    st.markdown("A self-balancing Binary Search Tree where the difference between the heights of left and right subtrees (balance factor) for any node is at most one. This ensures $O(\\log N)$ time complexity for all operations.")
    display_complexity(
        "$O(N)$",
        {
            "Insert": "O(\\log N)",
            "Search": "O(\\log N)",
            "Delete": "O(\\log N)"
        }
    )
    st.info("Interactive visualization of AVL tree rotations (single/double rotations) is graphically complex and beyond simple text-based rendering in Streamlit.")

elif selected_ds_option == "Red-Black Tree":
    st.header("Red-Black Tree")
    st.markdown("Another self-balancing Binary Search Tree, more complex than AVL trees but with looser balancing rules. It maintains balance by coloring nodes red or black and enforcing specific properties. Also guarantees $O(\\log N)$ time complexity for all operations.")
    display_complexity(
        "$O(N)$",
        {
            "Insert": "O(\\log N)",
            "Search": "O(\\log N)",
            "Delete": "O(\\log N)"
        }
    )
    st.info("Interactive visualization of Red-Black tree properties (coloring, rotations, recoloring) is graphically complex and beyond simple text-based rendering in Streamlit.")

elif selected_ds_option == "B-Tree / B+ Tree":
    st.header("B-Tree / B+ Tree")
    st.markdown("Self-balancing tree data structures that maintain sorted data and allow searches, sequential access, insertions, and deletions in logarithmic time. They are optimized for systems that read and write large blocks of data (e.g., disk storage) by maximizing the number of children per node.")
    display_complexity(
        "$O(N)$",
        {
            "Insert": "O(\\log_B N)", # B is order of tree
            "Search": "O(\\log_B N)",
            "Delete": "O(\\log_B N)"
        }
    )
    st.info("B-Trees and B+ Trees are highly complex to visualize interactively due to their multi-way branching and splitting/merging operations, especially in a text-based format.")

elif selected_ds_option == "Heap (Min-Heap, Max-Heap)":
    st.header("Heap (Min-Heap, Max-Heap)")
    st.markdown("A complete binary tree that satisfies the heap property. In a Min-Heap, the parent node is always smaller than its children. In a Max-Heap, the parent is always larger. Used for Priority Queues.")
    display_complexity(
        "$O(N)$",
        {
            "Insert": "O(\\log N)",
            "Extract Min/Max": "O(\\log N)",
            "Peek Min/Max": "O(1)"
        }
    )
    st.info("An interactive Min-Heap is demonstrated in the 'Priority Queue' section. Max-Heap is conceptually similar, often implemented by negating values in a Min-Heap.")

elif selected_ds_option == "Segment Tree":
    st.header("Segment Tree")
    st.markdown("A tree data structure used for storing information about intervals or segments. It allows querying for information about a range (e.g., sum, min, max) and updating elements efficiently.")
    display_complexity(
        "$O(N)$",
        {
            "Build": "O(N)",
            "Query Range": "O(\\log N)",
            "Update Element": "O(\\log N)"
        }
    )
    st.info("Segment Trees are complex to visualize interactively due to their recursive construction and range-based operations. They are usually represented as binary trees where each node represents an interval.")

elif selected_ds_option == "Fenwick Tree (Binary Indexed Tree)":
    st.header("Fenwick Tree (Binary Indexed Tree)")
    st.markdown("A data structure that can efficiently update elements and calculate prefix sums in a table of numbers. It's an alternative to a Segment Tree for certain types of range queries and updates.")
    display_complexity(
        "$O(N)$",
        {
            "Build": "O(N \\log N)",
            "Update Element": "O(\\log N)",
            "Query Prefix Sum": "O(\\log N)"
        }
    )
    st.info("Fenwick Trees are array-based and their 'tree-like' structure is implicit, making interactive visualization challenging without dedicated graphical tools.")

elif selected_ds_option == "Trie (Prefix Tree)":
    st.header("Trie (Prefix Tree)")
    st.markdown("A tree-like data structure used to store a dynamic set of strings, where keys are usually strings. Nodes store characters, and the position in the tree defines the associated key. Efficient for prefix searching.")
    display_complexity(
        "$O(L \\cdot A)$", # L = total length of words, A = alphabet size
        {
            "Insert": "O(L)", # L = length of word
            "Search": "O(L)",
            "Starts With (prefix search)": "O(L)"
        }
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        word_to_manage = st.text_input("Word to Manage", value=get_random_value("word"), key="trie_word_input")
        if st.button("Insert Word"):
            if word_to_manage:
                st.session_state.my_trie.insert(word_to_manage)
                st.success(f"Inserted '{word_to_manage}'.")
            else:
                st.warning("Please enter a word.")
        
        search_word = st.text_input("Word to Search", key="trie_search_input")
        if st.button("Search Word"):
            if search_word:
                found, path = st.session_state.my_trie.search(search_word)
                if found:
                    st.success(f"Search Result: Found '{search_word}'!")
                else:
                    st.warning(f"Search Result: '{search_word}' not found.")
                with st.expander("Search Path Details"):
                    st.write(f"Path: {' -> '.join(path)}")
            else:
                st.warning("Please enter a word to search.")
        
        prefix_to_check = st.text_input("Prefix to Check", key="trie_prefix_input")
        if st.button("Check Prefix"):
            if prefix_to_check:
                found, path = st.session_state.my_trie.starts_with(prefix_to_check)
                if found:
                    st.info(f"Prefix '{prefix_to_check}' exists!")
                else:
                    st.warning(f"Prefix '{prefix_to_check}' does not exist.")
                with st.expander("Prefix Check Path Details"):
                    st.write(f"Path: {' -> '.join(path)}")
            else:
                st.warning("Please enter a prefix.")
        
        if st.button("Clear Trie"):
            st.session_state.my_trie = Trie()
            st.error("Trie cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Trie Structure (Simplified Text View):")
        st.info("Root is at the top. Each branch represents a character. (End of Word) indicates a complete word.")
        st.code(st.session_state.my_trie.to_display_string())

elif selected_ds_option == "Graph":
    st.header("Graph (General Concept)")
    st.markdown("A collection of nodes (vertices) and edges that connect pairs of nodes. This section introduces the general idea; specific types and representations are explored below.")
    display_complexity(
        "$O(V+E)$", # V = vertices, E = edges
        {
            "Traversal (BFS/DFS)": "O(V+E)",
            "Add/Remove Vertex/Edge": "O(1) \\text{ or } O(E)"
        }
    )
    st.info("See 'Directed / Undirected Graph' and 'Adjacency Matrix / Adjacency List' for interactive examples.")

elif selected_ds_option == "Directed / Undirected Graph":
    st.header("Directed / Undirected Graph")
    st.markdown("Graphs can be:  \n- **Directed:** Edges have a direction (e.g., A -> B).  \n- **Undirected:** Edges have no direction (e.g., A -- B, meaning B -- A too).")
    display_complexity(
        "$O(V+E)$",
        {
            "Add Vertex": "O(1)",
            "Add Edge": "O(1)",
            "Remove Vertex": "O(V+E)",
            "Remove Edge": "O(E)"
        }
    )

    graph_type = st.radio("Select Graph Type:", ("Undirected", "Directed"), key="graph_type_select")
    current_graph = st.session_state.my_graph_undirected if graph_type == "Undirected" else st.session_state.my_graph_directed
    
    # Re-initialize if graph type changed (and it's not the correct instance)
    if (graph_type == "Undirected" and current_graph.is_directed) or \
       (graph_type == "Directed" and not current_graph.is_directed):
        if graph_type == "Undirected":
            st.session_state.my_graph_undirected = Graph(is_directed=False)
            current_graph = st.session_state.my_graph_undirected
        else:
            st.session_state.my_graph_directed = Graph(is_directed=True)
            current_graph = st.session_state.my_graph_directed
        st.info(f"Graph re-initialized as {graph_type}.")
        st.rerun()

    st.subheader("Operations:")
    col1, col2 = st.columns(2)
    with col1:
        new_vertex = st.text_input("Vertex to Add", value=get_random_value("V"), key="graph_add_vertex_input")
        if st.button("Add Vertex"):
            if new_vertex:
                current_graph.add_vertex(new_vertex)
                st.success(f"Added vertex '{new_vertex}'.")
            else:
                st.warning("Please enter a vertex.")
        
        u_edge = st.text_input("From Vertex (U)", key="graph_u_edge_input")
        v_edge = st.text_input("To Vertex (V)", key="graph_v_edge_input")
        edge_weight = st.number_input("Weight (optional)", value=1, key="graph_weight_input")
        if st.button("Add Edge"):
            if u_edge and v_edge:
                current_graph.add_edge(u_edge, v_edge, edge_weight)
                st.success(f"Added edge from '{u_edge}' to '{v_edge}' with weight {edge_weight}.")
            else:
                st.warning("Please enter both 'From' and 'To' vertices.")
    
    with col2:
        remove_vertex = st.text_input("Vertex to Remove", key="graph_remove_vertex_input")
        if st.button("Remove Vertex"):
            if remove_vertex:
                if current_graph.remove_vertex(remove_vertex):
                    st.info(f"Removed vertex '{remove_vertex}'.")
                else:
                    st.error(f"Vertex '{remove_vertex}' not found.")
            else:
                st.warning("Please enter a vertex to remove.")
        
        remove_u_edge = st.text_input("Remove From (U)", key="graph_remove_u_edge_input")
        remove_v_edge = st.text_input("Remove To (V)", key="graph_remove_v_edge_input")
        if st.button("Remove Edge"):
            if remove_u_edge and remove_v_edge:
                if current_graph.remove_edge(remove_u_edge, remove_v_edge):
                    st.info(f"Removed edge from '{remove_u_edge}' to '{remove_v_edge}'.")
                else:
                    st.error(f"Edge from '{remove_u_edge}' to '{remove_v_edge}' not found.")
            else:
                st.warning("Please enter both 'From' and 'To' vertices for edge removal.")
    
    st.subheader(f"Current {graph_type} Graph (Adjacency List):")
    st.code(current_graph.to_display_string())
    st.write(f"**Number of Vertices:** `{len(current_graph.vertices)}`")

    if st.button("Clear Current Graph"):
        if graph_type == "Undirected":
            st.session_state.my_graph_undirected = Graph(is_directed=False)
        else:
            st.session_state.my_graph_directed = Graph(is_directed=True)
        st.error(f"{graph_type} Graph cleared.")
        st.rerun()

elif selected_ds_option == "Weighted / Unweighted Graph":
    st.header("Weighted / Unweighted Graph")
    st.markdown("Graphs can also be:  \n- **Weighted:** Edges have an associated cost or value (e.g., distance, time).  \n- **Unweighted:** Edges simply indicate a connection, no cost.")
    st.info("The 'Directed / Undirected Graph' section above allows you to add weights to edges, demonstrating this concept.")

elif selected_ds_option == "Adjacency Matrix / Adjacency List":
    st.header("Adjacency Matrix / Adjacency List")
    st.markdown("These are common ways to represent graphs:  \n- **Adjacency List:** Each vertex stores a list of its adjacent vertices (and edge weights). Good for sparse graphs.  \n- **Adjacency Matrix:** A $V \\times V$ matrix where `matrix[i][j]` indicates an edge between vertex `i` and `j` (and its weight). Good for dense graphs.")
    st.info("The interactive graph section above uses an **Adjacency List**. Here's a conceptual look at an Adjacency Matrix:")
    
    st.subheader("Conceptual Adjacency Matrix Example:")
    st.markdown("For a graph with vertices A, B, C, D:")
    st.code("""
      A B C D
    A 0 1 0 1
    B 1 0 1 0
    C 0 1 0 0
    D 1 0 0 0
    """)
    st.markdown("Here, `1` indicates an edge, `0` no edge. For weighted graphs, the cell would contain the weight.")

elif selected_ds_option == "Tree as a special type of graph":
    st.header("Tree as a special type of graph")
    st.markdown("A tree is an **undirected graph** that satisfies two properties:  \n1.  It is **connected** (there is a path between any two vertices).  \n2.  It contains **no cycles** (no path starts and ends at the same vertex without repeating edges).")
    st.markdown("Alternatively, a tree is a connected graph with $V$ vertices and exactly $V-1$ edges.")
    st.info("This means many graph algorithms (like BFS, DFS) can be applied to trees.")

# --- Hashing Data Structures ---

elif selected_ds_option == "Hash Table":
    st.header("Hash Table (with Simple Chaining)")
    st.markdown("Maps keys to values using a hash function. Collisions are handled by storing multiple key-value pairs in a 'chain' (list) within each bucket.")
    display_complexity(
        "$O(N)$", # plus O(M) for buckets
        {
            "Insert (Average)": "O(1)",
            "Lookup (Average)": "O(1)",
            "Delete (Average)": "O(1)",
            "Insert (Worst - all collisions)": "O(N)",
            "Lookup (Worst - all collisions)": "O(N)",
            "Delete (Worst - all collisions)": "O(N)"
        }
    )

    st.session_state.hash_table_size = st.slider("Number of Buckets", 5, 20, st.session_state.hash_table_size, 1, key="hash_table_buckets")
    if len(st.session_state.my_hash_table) != st.session_state.hash_table_size:
        st.session_state.my_hash_table = [[] for _ in range(st.session_state.hash_table_size)]
        st.info(f"Hash table re-initialized with {st.session_state.hash_table_size} buckets.")
        st.rerun()

    col1, col2 = st.columns([1, 3])
    with col1:
        hash_key = st.text_input("Key (String)", value=get_random_value("key"), key="hash_key_input")
        hash_value = st.text_input("Value", value=get_random_value("value"), key="hash_value_input")

        if st.button("Insert/Update Pair"):
            msg = insert_hash_table(st.session_state.my_hash_table, hash_key, hash_value, st.session_state.hash_table_size)
            st.success(msg)

        search_hash_key = st.text_input("Key to Lookup", key="hash_lookup_input")
        if st.button("Lookup Key"):
            found_value, path = lookup_hash_table(st.session_state.my_hash_table, search_hash_key, st.session_state.hash_table_size)
            if found_value is not None:
                st.success(f"Lookup Result: Key '{search_hash_key}' found! Value: **'{found_value}'**")
            else:
                st.warning(f"Lookup Result: Key '{search_hash_key}' not found.")
            with st.expander("Lookup Path Details"):
                for step in path:
                    st.write(step)

        delete_hash_key = st.text_input("Key to Delete", key="hash_delete_input")
        if st.button("Delete Key"):
            msg = delete_hash_table(st.session_state.my_hash_table, delete_hash_key, st.session_state.hash_table_size)
            if "deleted" in msg:
                st.info(msg)
            else:
                st.warning(msg)
        
        if st.button("Clear Hash Table"):
            st.session_state.my_hash_table = [[] for _ in range(st.session_state.hash_table_size)]
            st.error("Hash Table cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Hash Table State (Buckets):")
        if any(st.session_state.my_hash_table):
            for i, bucket in enumerate(st.session_state.my_hash_table):
                bucket_content = f"Bucket {i}: "
                if bucket:
                    bucket_content += " -> ".join([f"('{k}', '{v}')" for k, v in bucket])
                else:
                    bucket_content += "(empty)"
                st.markdown(f"**`{bucket_content}`**")
        else:
            st.info("The Hash Table is currently empty.")

elif selected_ds_option == "Hash Map / Dictionary":
    st.header("Hash Map / Dictionary")
    st.markdown("These are abstract data types (ADTs) that map keys to values. Hash Tables are a common underlying data structure for implementing Hash Maps/Dictionaries. Python's built-in `dict` is an example.")
    st.info("The 'Hash Table' section above demonstrates the core functionality of a Hash Map/Dictionary.")

elif selected_ds_option == "Hash Set":
    st.header("Hash Set (with Simple Chaining)")
    st.markdown("Stores unique elements (keys only) using a hash function. Ideal for checking presence quickly. Similar to a Hash Table but only stores keys, not key-value pairs.")
    display_complexity(
        "$O(N)$",
        {
            "Add (Average)": "O(1)",
            "Remove (Average)": "O(1)",
            "Contains (Average)": "O(1)",
            "Add (Worst - all collisions)": "O(N)",
            "Remove (Worst - all collisions)": "O(N)",
            "Contains (Worst - all collisions)": "O(N)"
        }
    )

    st.session_state.hash_set_size = st.slider("Number of Buckets (for Set)", 5, 20, st.session_state.hash_set_size, 1, key="hash_set_buckets")
    if len(st.session_state.my_hash_set) != st.session_state.hash_set_size:
        st.session_state.my_hash_set = [[] for _ in range(st.session_state.hash_set_size)]
        st.info(f"Hash set re-initialized with {st.session_state.hash_set_size} buckets.")
        st.rerun()

    col1, col2 = st.columns([1, 3])
    with col1:
        set_key = st.text_input("Key to Manage (String)", value=get_random_value("item"), key="hash_set_key_input")

        if st.button("Add Key"):
            msg = hash_set_add(st.session_state.my_hash_set, set_key, st.session_state.hash_set_size)
            st.success(msg)

        if st.button("Remove Key"):
            msg = hash_set_remove(st.session_state.my_hash_set, set_key, st.session_state.hash_set_size)
            if "removed" in msg:
                st.info(msg)
            else:
                st.warning(msg)
                
        if st.button("Check if Contains Key"):
            found, path = hash_set_contains(st.session_state.my_hash_set, set_key, st.session_state.hash_set_size)
            if found:
                st.success(f"Key '{set_key}' IS in the set!")
            else:
                st.warning(f"Key '{set_key}' IS NOT in the set.")
            with st.expander("Contains Check Details"):
                for step in path:
                    st.write(step)
        
        if st.button("Clear Hash Set"):
            st.session_state.my_hash_set = [[] for _ in range(st.session_state.hash_set_size)]
            st.error("Hash Set cleared.")
            st.rerun()
    with col2:
        st.subheader("Current Hash Set State (Buckets):")
        if any(st.session_state.my_hash_set):
            for i, bucket in enumerate(st.session_state.my_hash_set):
                bucket_content = f"Bucket {i}: "
                if bucket:
                    bucket_content += " -> ".join([f"'{k}'" for k in bucket])
                else:
                    bucket_content += "(empty)"
                st.markdown(f"**`{bucket_content}`**")
        else:
            st.info("The Hash Set is currently empty.")

# --- Specialized or Advanced Data Structures ---

elif selected_ds_option == "Disjoint Set / Union-Find":
    st.header("Disjoint Set / Union-Find")
    st.markdown("A data structure that keeps track of a set of elements partitioned into a number of disjoint (non-overlapping) subsets. Supports `find` (which subset an element belongs to) and `union` (merging two subsets). Optimized with path compression and union by rank.")
    display_complexity(
        "$O(N)$",
        {
            "Find (Amortized)": "O(\\alpha(N))", # Inverse Ackermann function, practically constant
            "Union (Amortized)": "O(\\alpha(N))"
        }
    )

    initial_elements_str = st.text_input("Initial elements (comma-separated)", 
                                         ", ".join(st.session_state.disjoint_set_elements), 
                                         key="ds_initial_elements_input")
    
    if st.button("Initialize Disjoint Set"):
        elements_list = [e.strip() for e in initial_elements_str.split(',') if e.strip()]
        if elements_list:
            st.session_state.disjoint_set_elements = elements_list
            st.session_state.disjoint_set = DisjointSet(elements_list)
            st.success("Disjoint Set initialized!")
        else:
            st.error("Please provide valid comma-separated elements.")
        st.rerun() # Rerun to ensure new DS is loaded

    if 'disjoint_set' not in st.session_state or not st.session_state.disjoint_set.parent:
        st.info("Please initialize the Disjoint Set first.")
    else:
        current_ds = st.session_state.disjoint_set

        st.subheader("Union Operation")
        col1, col2 = st.columns(2)
        with col1:
            union_elem1 = st.text_input("Element 1 for Union", key="ds_union_elem1")
        with col2:
            union_elem2 = st.text_input("Element 2 for Union", key="ds_union_elem2")
        
        if st.button("Perform Union"):
            if union_elem1 and union_elem2 and union_elem1 in current_ds.parent and union_elem2 in current_ds.parent:
                success, msg = current_ds.union(union_elem1, union_elem2)
                if success:
                    st.success(msg)
                else:
                    st.info(msg)
            else:
                st.error("Please enter valid elements for Union that exist in the set.")
            st.rerun()

        st.subheader("Find Operation")
        find_elem = st.text_input("Element to Find", key="ds_find_elem")
        if st.button("Perform Find"):
            if find_elem and find_elem in current_ds.parent:
                root, path = current_ds.find(find_elem)
                st.info(f"The representative (root) of '{find_elem}' is '{root}'.")
                with st.expander("Path Compression Details"):
                    st.write(f"Path taken: {path}")
            else:
                st.error("Please enter a valid element for Find that exists in the set.")
            st.rerun()

        st.subheader("Current Disjoint Set State:")
        sets_display = current_ds.get_sets()
        
        st.write("Conceptual Sets:")
        for rep, elems in sets_display.items():
            st.write(f"  - Set of **{rep}**: {', '.join(sorted(elems))}")
        
        st.write("Raw Parent and Rank Mapping (Internal State):")
        parent_map = {elem: current_ds.parent[elem] for elem in sorted(current_ds.parent.keys())}
        rank_map = {elem: current_ds.rank[elem] for elem in sorted(current_ds.rank.keys())}
        st.code(f"Parent: {parent_map}\nRank: {rank_map}")
        
        if st.button("Reset Disjoint Set"):
            st.session_state.disjoint_set = DisjointSet(st.session_state.disjoint_set_elements)
            st.success("Disjoint Set reset!")
            st.rerun()

elif selected_ds_option == "Bloom Filter":
    st.header("Bloom Filter")
    st.markdown("A space-efficient probabilistic data structure used to test whether an element is a member of a set. False positives are possible (it might say an element is in the set when it's not), but false negatives are not (it will never say an element is not in the set when it actually is).")
    display_complexity(
        "$O(M)$", # M = size of bit array
        {
            "Add": "O(K)", # K = number of hash functions
            "Check": "O(K)"
        }
    )

    st.sidebar.subheader("Bloom Filter Settings")
    bf_size = st.sidebar.slider("Bit Array Size", 10, 100, st.session_state.bloom_filter_size, key="bf_size_slider")
    bf_hashes = st.sidebar.slider("Number of Hash Functions", 1, 10, st.session_state.bloom_filter_hashes, key="bf_hashes_slider")

    # Re-initialize if parameters change
    if st.session_state.bloom_filter.size != bf_size or st.session_state.bloom_filter.num_hashes != bf_hashes:
        st.session_state.bloom_filter_size = bf_size
        st.session_state.bloom_filter_hashes = bf_hashes
        st.session_state.bloom_filter = BloomFilter(bf_size, bf_hashes)
        st.info("Bloom Filter re-initialized with new settings.")
        st.rerun()

    current_bf = st.session_state.bloom_filter

    st.subheader("Operations:")
    col1, col2 = st.columns(2)
    with col1:
        add_bf_item = st.text_input("Value to Add", value=get_random_value("data"), key="bf_add_input")
        if st.button("Add to Bloom Filter"):
            if add_bf_item:
                current_bf.add(add_bf_item)
                st.success(f"Added '{add_bf_item}'.")
            else:
                st.warning("Please enter a value to add.")
            st.rerun()
    with col2:
        check_bf_item = st.text_input("Value to Check", key="bf_check_input")
        if st.button("Check in Bloom Filter"):
            if check_bf_item:
                is_present = current_bf.contains(check_bf_item)
                if is_present:
                    st.info(f"'{check_bf_item}' MIGHT BE present (all bits set).")
                else:
                    st.success(f"'{check_bf_item}' is DEFINITELY NOT present.")
            else:
                st.warning("Please enter a value to check.")

    st.subheader("Current Bloom Filter Bit Array:")
    st.code(f"[{' '.join(map(str, current_bf.bit_array))}]")
    st.write(f"Size: {current_bf.size}, Hashes: {current_bf.num_hashes}")

    if st.button("Reset Bloom Filter"):
        st.session_state.bloom_filter = BloomFilter(st.session_state.bloom_filter_size, st.session_state.bloom_filter_hashes)
        st.success("Bloom Filter reset!")
        st.rerun()

elif selected_ds_option == "Skip List":
    st.header("Skip List")
    st.markdown("A probabilistic data structure that allows elements to be searched, inserted, and deleted with expected $O(\\log N)$ time complexity. It consists of multiple layers of sorted linked lists, where each successive layer links fewer elements.")
    display_complexity(
        "$O(N)$",
        {
            "Insert (Expected)": "O(\\log N)",
            "Search (Expected)": "O(\\log N)",
            "Delete (Expected)": "O(\\log N)"
        }
    )

    st.sidebar.subheader("Skip List Settings")
    sl_max_level = st.sidebar.slider("Max Level", 2, 8, st.session_state.skip_list_max_level, key="sl_max_level_slider")
    sl_p = st.sidebar.slider("Probability (p)", 0.1, 0.9, st.session_state.skip_list_p, 0.1, key="sl_p_slider")

    if st.session_state.my_skip_list.max_level != sl_max_level or st.session_state.my_skip_list.p != sl_p:
        st.session_state.skip_list_max_level = sl_max_level
        st.session_state.skip_list_p = sl_p
        st.session_state.my_skip_list = SkipList(sl_max_level, sl_p)
        st.info("Skip List re-initialized with new settings.")
        st.rerun()

    current_sl = st.session_state.my_skip_list

    st.subheader("Operations:")
    col1, col2 = st.columns(2)
    with col1:
        add_sl_val = st.number_input("Value to Insert", value=random.randint(1, 100), key="sl_add_input")
        if st.button("Insert Value"):
            if current_sl.insert(add_sl_val):
                st.success(f"Inserted {add_sl_val}.")
            else:
                st.info(f"Value {add_sl_val} already exists.")
            st.rerun()
    with col2:
        search_sl_val = st.number_input("Value to Search", key="sl_search_input")
        if st.button("Search Value"):
            found, path = current_sl.search(search_sl_val)
            if found:
                st.success(f"Value {search_sl_val} found!")
            else:
                st.warning(f"Value {search_sl_val} not found.")
            with st.expander("Search Path Details"):
                for step in path:
                    st.write(step)
        
        delete_sl_val = st.number_input("Value to Delete", key="sl_delete_input")
        if st.button("Delete Value"):
            if current_sl.delete(delete_sl_val):
                st.info(f"Deleted {delete_sl_val}.")
            else:
                st.error(f"Value {delete_sl_val} not found for deletion.")
            st.rerun()

    st.subheader("Current Skip List Structure:")
    st.code(current_sl.to_display_string())
    st.write(f"Current Max Level: `{current_sl.level}`")

    if st.button("Reset Skip List"):
        st.session_state.my_skip_list = SkipList(st.session_state.skip_list_max_level, st.session_state.skip_list_p)
        st.success("Skip List reset!")
        st.rerun()

elif selected_ds_option == "Suffix Tree / Suffix Array":
    st.header("Suffix Tree / Suffix Array")
    st.markdown("Data structures used for fast full-text searches.  \n- **Suffix Tree:** A tree-like structure that stores all suffixes of a given text.  \n- **Suffix Array:** A sorted array of all suffixes of a given text.")
    display_complexity(
        "$O(N)$", # N = length of text
        {
            "Build (Suffix Tree)": "O(N)",
            "Build (Suffix Array)": "O(N \\log N)",
            "Pattern Search": "O(M + K)" # M = pattern length, K = number of occurrences
        }
    )
    st.info("These are highly specialized and complex data structures, primarily used in bioinformatics and string algorithms. Interactive visualization is beyond the scope of this app.")

elif selected_ds_option == "K-D Tree (K-Dimensional Tree)":
    st.header("K-D Tree (K-Dimensional Tree)")
    st.markdown("A space-partitioning data structure for organizing points in a k-dimensional space. Useful for range searches and nearest neighbor searches.")
    display_complexity(
        "$O(N)$",
        {
            "Build": "O(N \\log N)",
            "Search (Nearest Neighbor)": "O(\\log N) \\text{ on average, } O(N) \\text{ worst}",
            "Range Search": "O(\\sqrt{N} + K)" # K = number of points in range
        }
    )
    st.info("K-D Trees are used for spatial indexing. Interactive visualization would require graphical plotting of points and partitioning lines.")

elif selected_ds_option == "Quad Tree / Octree (for spatial partitioning)":
    st.header("Quad Tree / Octree")
    st.markdown("Tree data structures used for spatial partitioning.  \n- **Quadtree:** Divides a 2D space into four quadrants recursively.  \n- **Octree:** Divides a 3D space into eight octants recursively.")
    display_complexity(
        "$O(N)$",
        {
            "Build": "O(N \\log N)",
            "Search": "O(\\log N) \\text{ on average}"
        }
    )
    st.info("These are used in computer graphics, image processing, and game development for efficient spatial queries. Interactive visualization would require graphical rendering of the partitioned space.")

elif selected_ds_option == "LRU Cache (Least Recently Used)":
    st.header("LRU Cache (Least Recently Used)")
    st.markdown("A cache replacement policy that discards the least recently used items first. Typically implemented using a combination of a Hash Map (for $O(1)$ lookups) and a Doubly Linked List (to maintain order of usage for $O(1)$ updates of recentness).")
    display_complexity(
        "$O(Capacity)$",
        {
            "Get": "O(1)",
            "Put": "O(1)"
        }
    )

    lru_capacity = st.slider("Cache Capacity", 1, 5, st.session_state.lru_cache_capacity, key="lru_capacity_slider")
    if st.session_state.my_lru_cache.capacity != lru_capacity:
        st.session_state.lru_cache_capacity = lru_capacity
        st.session_state.my_lru_cache = LRUCache(lru_capacity)
        st.info(f"LRU Cache re-initialized with capacity {lru_capacity}.")
        st.rerun()

    current_lru = st.session_state.my_lru_cache

    st.subheader("Operations:")
    col1, col2 = st.columns(2)
    with col1:
        put_key = st.text_input("Key to Put", value=get_random_value("k"), key="lru_put_key_input")
        put_value = st.text_input("Value to Put", value=get_random_value("v"), key="lru_put_value_input")
        if st.button("Put (Key, Value)"):
            if put_key and put_value:
                eviction_msg = current_lru.put(put_key, put_value)
                if eviction_msg:
                    st.warning(eviction_msg)
                st.success(f"Put ({put_key}: {put_value}).")
            else:
                st.warning("Please enter both key and value.")
            st.rerun()
    with col2:
        get_key = st.text_input("Key to Get", key="lru_get_key_input")
        if st.button("Get Value"):
            if get_key:
                value = current_lru.get(get_key)
                if value != -1:
                    st.info(f"Got value: **{value}** for key '{get_key}'.")
                else:
                    st.error(f"Key '{get_key}' not found in cache.")
            else:
                st.warning("Please enter a key to get.")
            st.rerun()

    st.subheader("Current LRU Cache State:")
    st.code(current_lru.to_display_string())
    st.write(f"Current Size: `{len(current_lru.cache)}` / Capacity: `{current_lru.capacity}`")

    if st.button("Reset LRU Cache"):
        st.session_state.my_lru_cache = LRUCache(st.session_state.lru_cache_capacity)
        st.success("LRU Cache reset!")
        st.rerun()

elif selected_ds_option == "Interval Tree":
    st.header("Interval Tree")
    st.markdown("A tree data structure to hold intervals and query for all intervals that overlap with a given interval or point. Useful in computational geometry and scheduling.")
    display_complexity(
        "$O(N \\log N)$",
        {
            "Build": "O(N \\log N)",
            "Query (Overlap)": "O(\\log N + K)" # K = number of overlapping intervals
        }
    )
    st.info("Interval Trees are complex to visualize due to their specific node structure and overlap querying logic. They often involve sorting intervals and using a balanced BST internally.")

elif selected_ds_option == "Treap (Tree + Heap)":
    st.header("Treap (Tree + Heap)")
    st.markdown("A randomized binary search tree that combines the properties of a binary search tree (keys are ordered) and a binary heap (priorities are ordered). Each node has a key and a randomly assigned priority, and the tree is ordered by key (BST property) and by priority (heap property).")
    display_complexity(
        "$O(N)$",
        {
            "Insert (Expected)": "O(\\log N)",
            "Search (Expected)": "O(\\log N)",
            "Delete (Expected)": "O(\\log N)"
        }
    )
    st.info("Treaps are self-balancing BSTs that use random priorities to achieve balance. Visualizing both BST and heap properties simultaneously in a text-based format is highly challenging.")

# --- Other Abstract Data Types (ADTs) ---

elif selected_ds_option == "Map / Dictionary (ADT)":
    st.header("Map / Dictionary (Abstract Data Type)")
    st.markdown("An Abstract Data Type (ADT) that defines a collection of key-value pairs where each key is unique. It specifies operations like `put(key, value)`, `get(key)`, `delete(key)`, `contains(key)`. Python's built-in `dict` is a common implementation.")
    display_complexity(
        "$O(N)$", # Underlying hash table
        {
            "Insert (Average)": "O(1)",
            "Lookup (Average)": "O(1)",
            "Delete (Average)": "O(1)"
        }
    )
    st.info("The 'Hash Table' section provides an interactive demonstration of how a Map/Dictionary might be implemented.")
    st.subheader("Python `dict` Example:")
    st.code("""
my_map = {"apple": 5, "banana": 2}
print(my_map["apple"]) # Output: 5
my_map["orange"] = 8
print(my_map) # Output: {'apple': 5, 'banana': 2, 'orange': 8}
del my_map["banana"]
print("banana" in my_map) # Output: False
""")

elif selected_ds_option == "Set / Multiset (ADT)":
    st.header("Set / Multiset (Abstract Data Type)")
    st.markdown("An ADT that defines a collection of elements.  \n- **Set:** Stores unique, unordered elements.  \n- **Multiset (Bag):** Stores elements where duplicates are allowed, but order is not maintained.")
    display_complexity(
        "$O(N)$", # Underlying hash table for Set
        {
            "Add (Set/Multiset Avg)": "O(1)",
            "Remove (Set/Multiset Avg)": "O(1)",
            "Contains (Set Avg)": "O(1)",
            "Count (Multiset)": "O(1)"
        }
    )
    st.info("The 'Hash Set' section provides an interactive demonstration of how a Set might be implemented. Python's `collections.Counter` can implement a Multiset.")
    st.subheader("Python `set` Example:")
    st.code("""
my_set = {1, 2, 3, 2} # Duplicates are automatically removed
print(my_set) # Output: {1, 2, 3}
my_set.add(4)
print(1 in my_set) # Output: True
""")
    st.subheader("Python `collections.Counter` (Multiset) Example:")
    st.code("""
from collections import Counter
my_multiset = Counter(['apple', 'banana', 'apple', 'orange'])
print(my_multiset) # Output: Counter({'apple': 2, 'banana': 1, 'orange': 1})
print(my_multiset['apple']) # Output: 2
my_multiset.update(['banana', 'grape'])
print(my_multiset) # Output: Counter({'apple': 2, 'banana': 2, 'orange': 1, 'grape': 1})
""")

elif selected_ds_option == "Multimap (ADT)":
    st.header("Multimap (Abstract Data Type)")
    st.markdown("An ADT that defines a collection of key-value pairs, similar to a Map, but it allows **multiple values to be associated with a single key**. There's no direct built-in type in Python, but it can be implemented using `collections.defaultdict(list)`.")
    display_complexity(
        "$O(N)$",
        {
            "Insert": "O(1) \\text{ (average)}",
            "Lookup (all values for key)": "O(1) + O(K) \\text{ (where K is number of values)}"
        }
    )
    st.subheader("Python `collections.defaultdict(list)` Example:")
    st.code("""
from collections import defaultdict
my_multimap = defaultdict(list)
my_multimap['fruits'].append('apple')
my_multimap['fruits'].append('banana')
my_multimap['colors'].append('red')
print(my_multimap) # Output: defaultdict(<class 'list'>, {'fruits': ['apple', 'banana'], 'colors': ['red']})
print(my_multimap['fruits']) # Output: ['apple', 'banana']
""")

elif selected_ds_option == "Bag / Multibag (ADT)":
    st.header("Bag / Multibag (Abstract Data Type)")
    st.markdown("An ADT that defines a collection where elements can appear multiple times, and the order of elements does not matter. It's essentially a Multiset. Python's `collections.Counter` is a perfect fit.")
    st.info("See 'Set / Multiset (ADT)' for an example using `collections.Counter` which serves as a Multibag.")

# --- Category Headers (non-selectable, just for display in sidebar) ---
# These are handled by the `all_ds_options` list, which includes them as radio button options.
# The `if selected_ds_option == "..."` blocks above will simply not match for these.
# Streamlit's radio button behavior means they are clickable but don't lead to specific content.
# This is a limitation of using a single radio for both categories and items.
# A multi-page app structure would be cleaner for category navigation, but this fulfills the "exact order" request.
�
