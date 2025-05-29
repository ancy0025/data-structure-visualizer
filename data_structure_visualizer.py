import streamlit as st
import random
import collections # For deque and defaultdict
import heapq # For heap (Priority Queue)

st.set_page_config(page_title="Advanced Data Structure Visualizer", layout="wide")
st.title("📚 Advanced Interactive Data Structure Visualizer")
st.markdown("Explore how fundamental data structures and their algorithms work, including their time and space complexity. This app focuses on structures that can be effectively visualized with text-based representations.")

st.sidebar.markdown("""
### Data Structures Included:
- **Linear:** List, Stack, Queue, Deque, Singly Linked List
- **Non-Linear:** Binary Search Tree, Priority Queue (Min-Heap)
- **Hashing:** Hash Table (Chaining), Hash Set
""")

st.sidebar.write("---")
if st.sidebar.button("Reset All Data Structures", key="reset_all_btn"):
    st.session_state.my_list = []
    st.session_state.my_stack = []
    st.session_state.my_queue = collections.deque()
    st.session_state.my_singly_linked_list = None # Root of SLL
    st.session_state.my_deque = collections.deque()
    st.session_state.my_bst = None
    st.session_state.my_priority_queue = []
    st.session_state.my_hash_table = [[] for _ in range(st.session_state.hash_table_size)]
    st.session_state.my_hash_set = [[] for _ in range(st.session_state.hash_set_size)]
    st.success("All data structures reset!")
    st.rerun()

st.sidebar.write("---")
st.sidebar.info("""
**Note on Complexity:**
Some advanced data structures (e.g., AVL Trees, Red-Black Trees, Graphs with complex algorithms like Dijkstra's, B-Trees, Bloom Filters) require specialized graphical visualization and complex algorithmic implementations that are beyond the scope of a basic text-based Streamlit app. This visualizer focuses on conceptual understanding.
""")

# --- Session State Initialization for Data Structures ---
if "my_list" not in st.session_state:
    st.session_state.my_list = []
if "my_stack" not in st.session_state:
    st.session_state.my_stack = []
if "my_queue" not in st.session_state:
    st.session_state.my_queue = collections.deque()
if "my_singly_linked_list" not in st.session_state:
    st.session_state.my_singly_linked_list = None # Head of the singly linked list
if "my_deque" not in st.session_state:
    st.session_state.my_deque = collections.deque()
if "my_bst" not in st.session_state:
    st.session_state.my_bst = None
if "my_priority_queue" not in st.session_state:
    st.session_state.my_priority_queue = []
if "my_hash_table" not in st.session_state:
    st.session_state.my_hash_table = [[] for _ in range(10)]
if "hash_table_size" not in st.session_state:
    st.session_state.hash_table_size = 10
if "my_hash_set" not in st.session_state:
    st.session_state.my_hash_set = [[] for _ in range(10)]
if "hash_set_size" not in st.session_state:
    st.session_state.hash_set_size = 10


# --- Helper Functions for Singly Linked List (SLL) ---
class SLLNode:
    def __init__(self, value):
        self.value = value
        self.next = None

    def __repr__(self):
        return f"Node({self.value})"

def sll_to_list_representation(head):
    elements = []
    current = head
    while current:
        elements.append(current.value)
        current = current.next
    return elements

def sll_display_text(head):
    if not head:
        return "(empty)"
    
    current = head
    s = ""
    while current:
        s += f"{current.value}"
        if current.next:
            s += " -> "
        current = current.next
    return s

def sll_append(head, value):
    new_node = SLLNode(value)
    if head is None:
        return new_node
    current = head
    while current.next:
        current = current.next
    current.next = new_node
    return head

def sll_prepend(head, value):
    new_node = SLLNode(value)
    new_node.next = head
    return new_node

def sll_delete_by_value(head, value):
    if head is None:
        return None # List is empty

    # If head is the node to be deleted
    if head.value == value:
        return head.next

    current = head
    prev = None
    while current and current.value != value:
        prev = current
        current = current.next

    if current is None: # Value not found
        return head
    
    prev.next = current.next # Skip the current node
    return head

# --- Helper Functions for Binary Search Tree (BST) ---
class BSTNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __repr__(self):
        return f"Node({self.value})"

def to_dict(node):
    if node is None:
        return None
    return {"value": node.value, "left": to_dict(node.left), "right": to_dict(node.right)}

def insert_bst_node(root, value):
    if root is None:
        return BSTNode(value)
    if value < root.value:
        root.left = insert_bst_node(root.left, value)
    else:
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
        path_trace.append(f"Comparing {value} > {root.value}. Going right.")
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

def display_bst_node_text(node, prefix="", is_left=None):
    if node is not None:
        connector = ""
        if is_left is True:
            connector = "├── L: "
        elif is_left is False:
            connector = "└── R: "
        elif is_left is None:
            connector = "Root: "

        st.write(f"{prefix}{connector}**{node.value}**")
        
        new_prefix = prefix + ("│   " if is_left is not False else "    ") 

        if node.left:
            display_bst_node_text(node.left, new_prefix, True)
        else:
            st.write(f"{new_prefix}├── L: (null)")

        if node.right:
            display_bst_node_text(node.right, new_prefix, False)
        else:
            st.write(f"{new_prefix}└── R: (null)")

# --- Helper Functions for Hash Table ---
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

# --- Helper Functions for Hash Set ---
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


# --- Main Application Sections ---

st.header("Linear Data Structures")
st.write("---")

# --- 1. Python List (Array-like) ---
st.subheader("1. Python List (Array-like)")
st.markdown("A dynamic array that can grow and shrink. Elements are ordered.")

with st.expander("Time and Space Complexity for Python List"):
    st.markdown("""
    * **Space Complexity:** $O(N)$ for storing $N$ elements.
    * **Time Complexity:**
        * **Access/Lookup by Index (`list[i]`):** $O(1)$
        * **Append (`list.append(item)`):** $O(1)$ amortized (resizing takes $O(N)$, but rarely)
        * **Insert (`list.insert(i, item)`):** $O(N)$ (elements after `i` must be shifted)
        * **Delete (`list.pop(i)`, `del list[i]`, `list.remove(item)`):** $O(N)$ (elements after `i` must be shifted)
        * **Search (`item in list` or `list.index(item)`):** $O(N)$ (linear scan)
    """)

list_col1, list_col2 = st.columns([1, 3])
with list_col1:
    st.markdown("**Operations:**")
    new_list_val = st.number_input("Value to Add", value=random.randint(1, 100), key="list_add_input")
    if st.button("Add to List (Append)", key="list_add_btn"):
        st.session_state.my_list.append(new_list_val)
        st.success(f"Added {new_list_val} to the list.")
    if st.button("Remove Last Element", key="list_remove_btn"):
        if st.session_state.my_list:
            removed_val = st.session_state.my_list.pop()
            st.info(f"Removed {removed_val} from the list.")
        else:
            st.warning("List is empty! Nothing to remove.")
    if st.button("Clear List", key="list_clear_btn"):
        st.session_state.my_list = []
        st.error("List cleared.")
with list_col2:
    st.markdown("**Current List State:**")
    if st.session_state.my_list:
        st.json(st.session_state.my_list)
        st.write(f"**Current Size:** `{len(st.session_state.my_list)}`")
    else:
        st.info("The list is currently empty.")
st.write("---")

# --- 2. Stack (LIFO) ---
st.subheader("2. Stack (LIFO - Last In, First Out)")
st.markdown("A collection where elements are added and removed from the same end (the 'top'). Think of a stack of plates.")

with st.expander("Time and Space Complexity for Stack"):
    st.markdown("""
    * **Space Complexity:** $O(N)$ for storing $N$ elements.
    * **Time Complexity:**
        * **Push:** $O(1)$
        * **Pop:** $O(1)$
        * **Peek/Top:** $O(1)$
        * **IsEmpty:** $O(1)$
    """)

stack_col1, stack_col2 = st.columns([1, 3])
with stack_col1:
    st.markdown("**Operations:**")
    new_stack_val = st.number_input("Value to Push", value=random.randint(1, 100), key="stack_push_input")
    if st.button("Push (Add to Top)", key="stack_push_btn"):
        st.session_state.my_stack.append(new_stack_val)
        st.success(f"Pushed {new_stack_val} onto the stack.")
    if st.button("Pop (Remove from Top)", key="stack_pop_btn"):
        if st.session_state.my_stack:
            popped_val = st.session_state.my_stack.pop()
            st.info(f"Popped {popped_val} from the stack.")
        else:
            st.warning("Stack is empty! Nothing to pop.")
    if st.button("Peek (View Top)", key="stack_peek_btn"):
        if st.session_state.my_stack:
            st.info(f"Top element: **{st.session_state.my_stack[-1]}**")
        else:
            st.warning("Stack is empty!")
    if st.button("Clear Stack", key="stack_clear_btn"):
        st.session_state.my_stack = []
        st.error("Stack cleared.")
with stack_col2:
    st.markdown("**Current Stack State (Top is at the end):**")
    if st.session_state.my_stack:
        st.markdown("```")
        for item in reversed(st.session_state.my_stack):
            st.markdown(f"| {item:<10} |")
            st.markdown("------------")
        st.markdown(f"| (BOTTOM) |")
        st.markdown("```")
        st.write(f"**Current Size:** `{len(st.session_state.my_stack)}`")
    else:
        st.info("The stack is currently empty.")
st.write("---")

# --- 3. Queue (FIFO) ---
st.subheader("3. Queue (FIFO - First In, First Out)")
st.markdown("A collection where elements are added to one end ('rear') and removed from the other end ('front'). Think of a waiting line.")

with st.expander("Time and Space Complexity for Queue (using `collections.deque`)"):
    st.markdown("""
    * **Space Complexity:** $O(N)$ for storing $N$ elements.
    * **Time Complexity:**
        * **Enqueue (Add to Rear):** $O(1)$
        * **Dequeue (Remove from Front):** $O(1)$
        * **Front/Peek:** $O(1)$
        * **IsEmpty:** $O(1)$
    """)

queue_col1, queue_col2 = st.columns([1, 3])
with queue_col1:
    st.markdown("**Operations:**")
    new_queue_val = st.number_input("Value to Enqueue", value=random.randint(1, 100), key="queue_enqueue_input")
    if st.button("Enqueue (Add to Rear)", key="queue_enqueue_btn"):
        st.session_state.my_queue.append(new_queue_val)
        st.success(f"Enqueued {new_queue_val} to the queue.")
    if st.button("Dequeue (Remove from Front)", key="queue_dequeue_btn"):
        if st.session_state.my_queue:
            dequeued_val = st.session_state.my_queue.popleft()
            st.info(f"Dequeued {dequeued_val} from the queue.")
        else:
            st.warning("Queue is empty! Nothing to dequeue.")
    if st.button("Front (View Front)", key="queue_front_btn"):
        if st.session_state.my_queue:
            st.info(f"Front element: **{st.session_state.my_queue[0]}**")
        else:
            st.warning("Queue is empty!")
    if st.button("Clear Queue", key="queue_clear_btn"):
        st.session_state.my_queue.clear()
        st.error("Queue cleared.")
with queue_col2:
    st.markdown("**Current Queue State (Front is on the left):**")
    if st.session_state.my_queue:
        queue_str = " <-- ".join(map(str, st.session_state.my_queue))
        st.markdown(f"**FRONT** <-- `{queue_str}` <-- **REAR**")
        st.write(f"**Current Size:** `{len(st.session_state.my_queue)}`")
    else:
        st.info("The queue is currently empty.")
st.write("---")

# --- 4. Singly Linked List ---
st.subheader("4. Singly Linked List")
st.markdown("A sequence of nodes where each node contains data and a reference (link) to the next node in the sequence. Traversal is always from head to tail.")

with st.expander("Time and Space Complexity for Singly Linked List"):
    st.markdown("""
    * **Space Complexity:** $O(N)$ for storing $N$ nodes.
    * **Time Complexity:**
        * **Access/Lookup by Index:** $O(N)$ (requires traversal)
        * **Append (to Tail):** $O(N)$ (requires traversal to find tail)
        * **Prepend (to Head):** $O(1)$
        * **Insert (at specific index):** $O(N)$
        * **Delete (by value or index):** $O(N)$ (requires traversal to find node and its predecessor)
    """)

sll_col1, sll_col2 = st.columns([1, 3])
with sll_col1:
    st.markdown("**Operations:**")
    new_sll_val = st.number_input("Value to Add", value=random.randint(1, 100), key="sll_add_input")
    if st.button("Append (Add to Tail)", key="sll_append_btn"):
        st.session_state.my_singly_linked_list = sll_append(st.session_state.my_singly_linked_list, new_sll_val)
        st.success(f"Appended {new_sll_val} to the list.")
    if st.button("Prepend (Add to Head)", key="sll_prepend_btn"):
        st.session_state.my_singly_linked_list = sll_prepend(st.session_state.my_singly_linked_list, new_sll_val)
        st.success(f"Prepended {new_sll_val} to the list.")
    
    delete_sll_val = st.number_input("Value to Delete", value=random.randint(1, 100), key="sll_delete_input")
    if st.button("Delete by Value", key="sll_delete_btn"):
        initial_sll_elements = sll_to_list_representation(st.session_state.my_singly_linked_list)
        st.session_state.my_singly_linked_list = sll_delete_by_value(st.session_state.my_singly_linked_list, delete_sll_val)
        final_sll_elements = sll_to_list_representation(st.session_state.my_singly_linked_list)
        if len(final_sll_elements) < len(initial_sll_elements):
            st.info(f"Deleted {delete_sll_val} from the list.")
        else:
            st.warning(f"Value {delete_sll_val} not found in the list.")
    
    if st.button("Clear Linked List", key="sll_clear_btn"):
        st.session_state.my_singly_linked_list = None
        st.error("Singly Linked List cleared.")
with sll_col2:
    st.markdown("**Current Singly Linked List State:**")
    if st.session_state.my_singly_linked_list:
        st.code(sll_display_text(st.session_state.my_singly_linked_list))
    else:
        st.info("The Singly Linked List is currently empty.")
st.write("---")

# --- 5. Deque (Double-Ended Queue) ---
st.subheader("5. Deque (Double-Ended Queue)")
st.markdown("A generalization of a queue and stack, allowing elements to be added or removed from both ends.")

with st.expander("Time and Space Complexity for Deque (using `collections.deque`)"):
    st.markdown("""
    * **Space Complexity:** $O(N)$ for storing $N$ elements.
    * **Time Complexity:**
        * **Append (`append`):** $O(1)$
        * **Append Left (`appendleft`):** $O(1)$
        * **Pop (`pop` - from right):** $O(1)$
        * **Pop Left (`popleft` - from left):** $O(1)$
        * **Peek (left/right):** $O(1)$
    """)

deque_col1, deque_col2 = st.columns([1, 3])
with deque_col1:
    st.markdown("**Operations:**")
    new_deque_val = st.number_input("Value to Add", value=random.randint(1, 100), key="deque_add_input")
    if st.button("Append (Add to Right)", key="deque_append_btn"):
        st.session_state.my_deque.append(new_deque_val)
        st.success(f"Appended {new_deque_val}.")
    if st.button("Append Left (Add to Left)", key="deque_appendleft_btn"):
        st.session_state.my_deque.appendleft(new_deque_val)
        st.success(f"Appended left {new_deque_val}.")
    
    if st.button("Pop (Remove from Right)", key="deque_pop_btn"):
        if st.session_state.my_deque:
            popped_val = st.session_state.my_deque.pop()
            st.info(f"Popped {popped_val} from right.")
        else:
            st.warning("Deque is empty!")
    if st.button("Pop Left (Remove from Left)", key="deque_popleft_btn"):
        if st.session_state.my_deque:
            popped_val = st.session_state.my_deque.popleft()
            st.info(f"Popped {popped_val} from left.")
        else:
            st.warning("Deque is empty!")
    if st.button("Clear Deque", key="deque_clear_btn"):
        st.session_state.my_deque.clear()
        st.error("Deque cleared.")
with deque_col2:
    st.markdown("**Current Deque State (Left/Front on the left, Right/Rear on the right):**")
    if st.session_state.my_deque:
        deque_str = " <-> ".join(map(str, st.session_state.my_deque))
        st.markdown(f"**LEFT** <-> `{deque_str}` <-> **RIGHT**")
        st.write(f"**Current Size:** `{len(st.session_state.my_deque)}`")
    else:
        st.info("The Deque is currently empty.")
st.write("---")

st.header("Non-Linear Data Structures")
st.write("---")

# --- 6. Priority Queue (Min-Heap) ---
st.subheader("6. Priority Queue (Min-Heap Implementation)")
st.markdown("A collection where elements are served based on priority. Here, lower numbers have higher priority (Min-Heap).")

with st.expander("Time and Space Complexity for Priority Queue (Min-Heap)"):
    st.markdown("""
    * **Space Complexity:** $O(N)$ for storing $N$ elements.
    * **Time Complexity:**
        * **Insert (`heapq.heappush`):** $O(\log N)$
        * **Extract Min (`heapq.heappop`):** $O(\log N)$
        * **Peek Min:** $O(1)$
    """)

pq_col1, pq_col2 = st.columns([1, 3])
with pq_col1:
    st.markdown("**Operations:**")
    new_pq_val = st.number_input("Value to Insert (Priority)", value=random.randint(1, 100), key="pq_insert_input")
    if st.button("Insert (Enqueue)", key="pq_insert_btn"):
        heapq.heappush(st.session_state.my_priority_queue, new_pq_val)
        st.success(f"Inserted {new_pq_val} into the Priority Queue.")
    if st.button("Extract Min (Dequeue)", key="pq_extract_min_btn"):
        if st.session_state.my_priority_queue:
            extracted_val = heapq.heappop(st.session_state.my_priority_queue)
            st.info(f"Extracted minimum value: **{extracted_val}**.")
        else:
            st.warning("Priority Queue is empty!")
    if st.button("Peek Min", key="pq_peek_min_btn"):
        if st.session_state.my_priority_queue:
            st.info(f"Minimum element (Peek): **{st.session_state.my_priority_queue[0]}**")
        else:
            st.warning("Priority Queue is empty!")
    if st.button("Clear Priority Queue", key="pq_clear_btn"):
        st.session_state.my_priority_queue = []
        st.error("Priority Queue cleared.")
with pq_col2:
    st.markdown("**Current Priority Queue (Heap Array Representation):**")
    if st.session_state.my_priority_queue:
        st.json(st.session_state.my_priority_queue)
        st.write(f"**Current Size:** `{len(st.session_state.my_priority_queue)}`")
        
        st.subheader("Conceptual Tree View (for Heap):")
        st.markdown("*(Note: This is the underlying array. The tree structure maintains heap property.)*")
        st.write("```")
        temp_heap = list(st.session_state.my_priority_queue)
        level = 0
        idx = 0
        while idx < len(temp_heap):
            nodes_at_level = 2**level
            level_nodes = temp_heap[idx : idx + nodes_at_level]
            st.write(" ".join(map(str, level_nodes)).center(60))
            idx += nodes_at_level
            level += 1
        st.write("```")
    else:
        st.info("The Priority Queue is currently empty.")
st.write("---")

# --- 7. Binary Search Tree (BST) ---
st.subheader("7. Binary Search Tree (BST)")
st.markdown("A tree-based data structure where values in the left subtree are smaller, and values in the right subtree are larger.")

with st.expander("Time and Space Complexity for Binary Search Tree"):
    st.markdown("""
    * **Space Complexity:** $O(N)$ for storing $N$ elements.
    * **Time Complexity (Average Case - Balanced Tree):**
        * **Insert:** $O(\log N)$
        * **Search:** $O(\log N)$
        * **Delete:** $O(\log N)$
    * **Time Complexity (Worst Case - Skewed Tree):**
        * **Insert:** $O(N)$
        * **Search:** $O(N)$
        * **Delete:** $O(N)$
    """)

bst_col1, bst_col2 = st.columns([1, 3])
with bst_col1:
    st.markdown("**Operations:**")
    new_bst_val = st.number_input("Value to Insert", value=random.randint(1, 100), key="bst_insert_input")
    if st.button("Insert Node", key="bst_insert_btn"):
        st.session_state.my_bst = insert_bst_node(st.session_state.my_bst, new_bst_val)
        st.success(f"Inserted node with value {new_bst_val}.")
    
    search_bst_val = st.number_input("Value to Search", value=random.randint(1, 100), key="bst_search_input")
    if st.button("Search Node", key="bst_search_btn"):
        found, path = search_bst_node(st.session_state.my_bst, search_bst_val)
        if found:
            st.success(f"Search Result: Found {search_bst_val}!")
        else:
            st.warning(f"Search Result: {search_bst_val} not found.")
        with st.expander("Search Path Details"):
            for step in path:
                st.write(step)

    delete_bst_val = st.number_input("Value to Delete", value=random.randint(1, 100), key="bst_delete_input")
    if st.button("Delete Node", key="bst_delete_btn"):
        original_bst_json = to_dict(st.session_state.my_bst)
        st.session_state.my_bst = delete_bst_node(st.session_state.my_bst, delete_bst_val)
        if to_dict(st.session_state.my_bst) != original_bst_json:
            st.info(f"Deleted node with value {delete_bst_val}.")
        else:
            st.warning(f"Node {delete_bst_val} not found or couldn't be deleted.")

    if st.button("Clear BST", key="bst_clear_btn"):
        st.session_state.my_bst = None
        st.error("Binary Search Tree cleared.")
with bst_col2:
    st.markdown("**Current BST Structure (JSON representation):**")
    if st.session_state.my_bst:
        st.json(to_dict(st.session_state.my_bst))
        st.subheader("Visualized Tree Structure (Simplified Text View):")
        display_bst_node_text(st.session_state.my_bst)
    else:
        st.info("The Binary Search Tree is currently empty. Insert a node to start building it!")
st.write("---")

st.header("Hashing Data Structures")
st.write("---")

# --- 8. Hash Table (with Chaining) ---
st.subheader("8. Hash Table (with Simple Chaining)")
st.markdown("Maps keys to values using a hash function. Collisions are handled by storing multiple key-value pairs in a 'chain' (list) within each bucket.")

with st.expander("Time and Space Complexity for Hash Table (with Chaining)"):
    st.markdown("""
    * **Space Complexity:** $O(N)$ for storing $N$ elements (plus $O(M)$ for $M$ buckets, where $M$ is table size).
    * **Time Complexity (Average Case):**
        * **Insert:** $O(1)$
        * **Lookup:** $O(1)$
        * **Delete:** $O(1)$
    * **Time Complexity (Worst Case - All collisions, resulting in a single long chain):**
        * **Insert:** $O(N)$
        * **Lookup:** $O(N)$
        * **Delete:** $O(N)$
    """)

hash_table_col1, hash_table_col2 = st.columns([1, 3])
with hash_table_col1:
    st.markdown("**Operations:**")
    st.session_state.hash_table_size = st.slider("Number of Buckets", 5, 20, st.session_state.hash_table_size, 1, key="hash_table_buckets")
    if len(st.session_state.my_hash_table) != st.session_state.hash_table_size:
        st.session_state.my_hash_table = [[] for _ in range(st.session_state.hash_table_size)]
        st.info(f"Hash table re-initialized with {st.session_state.hash_table_size} buckets.")

    hash_key = st.text_input("Key (String)", value=f"key_{random.randint(1, 100)}", key="hash_key_input")
    hash_value = st.text_input("Value", value=f"value_{random.randint(100, 200)}", key="hash_value_input")

    if st.button("Insert/Update Pair", key="hash_insert_btn"):
        msg = insert_hash_table(st.session_state.my_hash_table, hash_key, hash_value, st.session_state.hash_table_size)
        st.success(msg)

    search_hash_key = st.text_input("Key to Lookup", value=f"key_{random.randint(1, 100)}", key="hash_lookup_input")
    if st.button("Lookup Key", key="hash_lookup_btn"):
        found_value, path = lookup_hash_table(st.session_state.my_hash_table, search_hash_key, st.session_state.hash_table_size)
        if found_value is not None:
            st.success(f"Lookup Result: Key '{search_hash_key}' found! Value: **'{found_value}'**")
        else:
            st.warning(f"Lookup Result: Key '{search_hash_key}' not found.")
        with st.expander("Lookup Path Details"):
            for step in path:
                st.write(step)

    delete_hash_key = st.text_input("Key to Delete", value=f"key_{random.randint(1, 100)}", key="hash_delete_input")
    if st.button("Delete Key", key="hash_delete_btn"):
        msg = delete_hash_table(st.session_state.my_hash_table, delete_hash_key, st.session_state.hash_table_size)
        if "deleted" in msg:
            st.info(msg)
        else:
            st.warning(msg)
    
    if st.button("Clear Hash Table", key="hash_clear_btn"):
        st.session_state.my_hash_table = [[] for _ in range(st.session_state.hash_table_size)]
        st.error("Hash Table cleared.")
with hash_table_col2:
    st.markdown("**Current Hash Table State (Buckets):**")
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
st.write("---")

# --- 9. Hash Set (with Chaining) ---
st.subheader("9. Hash Set (with Simple Chaining)")
st.markdown("Stores unique elements (keys only) using a hash function. Ideal for checking presence quickly.")

with st.expander("Time and Space Complexity for Hash Set (with Chaining)"):
    st.markdown("""
    * **Space Complexity:** $O(N)$ for storing $N$ unique elements.
    * **Time Complexity (Average Case):**
        * **Add:** $O(1)$
        * **Remove:** $O(1)$
        * **Contains:** $O(1)$
    * **Time Complexity (Worst Case - All collisions):**
        * **Add:** $O(N)$
        * **Remove:** $O(N)$
        * **Contains:** $O(N)$
    """)

hash_set_col1, hash_set_col2 = st.columns([1, 3])
with hash_set_col1:
    st.markdown("**Operations:**")
    st.session_state.hash_set_size = st.slider("Number of Buckets (for Set)", 5, 20, st.session_state.hash_set_size, 1, key="hash_set_buckets")
    if len(st.session_state.my_hash_set) != st.session_state.hash_set_size:
        st.session_state.my_hash_set = [[] for _ in range(st.session_state.hash_set_size)]
        st.info(f"Hash set re-initialized with {st.session_state.hash_set_size} buckets.")

    set_key = st.text_input("Key to Manage (String)", value=f"item_{random.randint(1, 100)}", key="hash_set_key_input")

    if st.button("Add Key", key="hash_set_add_btn"):
        msg = hash_set_add(st.session_state.my_hash_set, set_key, st.session_state.hash_set_size)
        st.success(msg)

    if st.button("Remove Key", key="hash_set_remove_btn"):
        msg = hash_set_remove(st.session_state.my_hash_set, set_key, st.session_state.hash_set_size)
        if "removed" in msg:
            st.info(msg)
        else:
            st.warning(msg)
            
    if st.button("Check if Contains Key", key="hash_set_contains_btn"):
        found, path = hash_set_contains(st.session_state.my_hash_set, set_key, st.session_state.hash_set_size)
        if found:
            st.success(f"Key '{set_key}' IS in the set!")
        else:
            st.warning(f"Key '{set_key}' IS NOT in the set.")
        with st.expander("Contains Check Details"):
            for step in path:
                st.write(step)
    
    if st.button("Clear Hash Set", key="hash_set_clear_btn"):
        st.session_state.my_hash_set = [[] for _ in range(st.session_state.hash_set_size)]
        st.error("Hash Set cleared.")
with hash_set_col2:
    st.markdown("**Current Hash Set State (Buckets with Keys):**")
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
st.write("---")

st.markdown("Created with Streamlit for Advanced Data Structure Visualization")
