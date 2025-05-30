import streamlit as st
import time
import random

# --- Data Structures and Their Representations ---
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None
        self.left = None
        self.right = None

def create_linked_list(values):
    head = None
    tail = None
    for value in values:
        new_node = Node(value)
        if not head:
            head = new_node
            tail = new_node
        else:
            tail.next = new_node
            tail = new_node
    return head

def create_doubly_linked_list(values):
    head = None
    tail = None
    for value in values:
        new_node = Node(value)
        if not head:
            head = new_node
            tail = new_node
        else:
            tail.next = new_node
            new_node.prev = tail
            tail = new_node
    return head

def create_binary_tree(values):
    if not values:
        return None
    root = Node(values[0])
    nodes = [root]
    i = 1
    while i < len(values):
        node = nodes.pop(0)
        if i < len(values):
            if values[i] is not None:
                node.left = Node(values[i])
                nodes.append(node.left)
            i += 1
        if i < len(values):
            if values[i] is not None:
                 node.right = Node(values[i])
                 nodes.append(node.right)
            i += 1
    return root

def display_linked_list(head):
    display = ""
    current = head
    while current:
        display += f"{current.data} -> "
        current = current.next
    return display + "None"

def display_doubly_linked_list(head):
    display = ""
    current = head
    while current:
        display += f"{current.data} <-> "
        current = current.next
    return display + "None"

def display_tree(root):
    if not root:
        return "Empty Tree"
    def traverse(node, level=0, prefix="Root: "):
        if node:
            right_str = traverse(node.right, level + 1, "Right: ") if node.right else ""
            node_str = "  " * level + prefix + str(node.data) + "\n"
            left_str = traverse(node.left, level + 1, "Left: ") if node.left else ""
            return right_str + node_str + left_str
        return ""
    return traverse(root)

def enqueue(queue, item):
    queue.append(item)

def dequeue(queue):
    if queue:
        return queue.pop(0)
    return None

def push(stack, item):
    stack.append(item)

def pop(stack):
    if stack:
        return stack.pop()
    return None

def insert_bst(root, key):
    if not root:
        return Node(key)
    if key < root.data:
        root.left = insert_bst(root.left, key)
    else:
        root.right = insert_bst(root.right, key)
    return root

def search_bst(root, key):
    if not root or root.data == key:
        return root
    if key < root.data:
        return search_bst(root.left, key)
    return search_bst(root.right, key)

def hash_function(key, size):
    return key % size

def insert_hash_table(table, key, value):
    index = hash_function(key, len(table))
    table[index].append((key, value))

def search_hash_table(table, key):
    index = hash_function(key, len(table))
    for k, v in table[index]:
        if k == key:
            return v
    return None

def add_to_bloom_filter(bloom_filter, item, hash_functions):
    for hash_func in hash_functions:
        index = hash_func(item, len(bloom_filter))
        bloom_filter[index] = True

def check_bloom_filter(bloom_filter, item, hash_functions):
    for hash_func in hash_functions:
        index = hash_func(item, len(bloom_filter))
        if not bloom_filter[index]:
            return False
    return True

def insert_skip_list(skip_list, key, value, max_level):
    level = 0
    while random.random() < 0.5 and level < max_level:
        level += 1
    new_node = [None] * (level + 1)
    new_node[0] = (key, value)
    if not skip_list:
        skip_list.append(new_node)
        return
    current = skip_list[0]
    update = [None] * (max_level + 1)
    for i in range(len(current) - 1, -1, -1):
        while current[i] and current[i][0] < key:
            current = current[i]
        update[i] = current
    if level >= len(skip_list):
        skip_list.append(new_node)
    else:
        skip_list[level] = new_node
    for i in range(level + 1):
        new_node[i] = update[i][i] if update[i] else None
        if update[i]:
            update[i][i] = new_node

def search_skip_list(skip_list, key):
    if not skip_list:
        return None
    current = skip_list[0]
    for i in range(len(current) - 1, -1, -1):
        while current[i] and current[i][0] < key:
            current = current[i]
    if current[0] == key:
        return current[0][1]
    return None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.queue = []

    def get(self, key):
        if key in self.cache:
            self.queue.remove(key)
            self.queue.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.queue.remove(key)
        elif len(self.queue) >= self.capacity:
            oldest = self.queue.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.queue.append(key)

def find_set(parent, i):
    if parent[i] == i:
        return i
    return find_set(parent, parent[i])

def union_sets(parent, rank, x, y):
    x_root = find_set(parent, x)
    y_root = find_set(parent, y)
    if x_root != y_root:
        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        elif rank[x_root] > rank[y_root]:
            parent[y_root] = x_root
        else:
            parent[y_root] = x_root
            rank[x_root] += 1

def insert_avl(root, key):
    if not root:
        return Node(key)

    if key < root.data:
        root.left = insert_avl(root.left, key)
    else:
        root.right = insert_avl(root.right, key)

    return root

def search_avl(root, key):
    if not root or root.data == key:
        return root
    if key < root.data:
        return search_avl(root.left, key)
    return search_avl(root.right, key)

def insert_red_black_tree(root, key):
    if not root:
        return Node(key)
    if key < root.data:
        root.left = insert_red_black_tree(root.left, key)
    else:
        root.right = insert_red_black_tree(root.right, key)
    return root

def search_red_black_tree(root, key):
    if not root or root.data == key:
        return root
    if key < root.data:
        return search_red_black_tree(root.left, key)
    return search_red_black_tree(root.right, key)

def insert_b_tree(root, key):
    if not root:
        return Node(key)
    if key < root.data:
        root.left = insert_b_tree(root.left, key)
    else:
        root.right = insert_b_tree(root.right, key)
    return root

def search_b_tree(root, key):
     if not root or root.data == key:
        return root
     if key < root.data:
        return search_b_tree(root.left, key)
     return search_b_tree(root.right, key)

# --- Streamlit App ---
st.title("Interactive Data Structure Explorer")

data_structures = [
    "Linked List", "Doubly Linked List", "Queue", "Stack", "Binary Tree",
    "Binary Search Tree (BST)", "Hash Table", "Bloom Filter", "Skip List",
    "LRU Cache", "Disjoint Sets", "AVL Tree (Conceptual)",
    "Red-Black Tree (Conceptual)", "B-Tree (Conceptual)"
]
structure_choice = st.selectbox("Select a Data Structure", data_structures)

st.write("---")

if structure_choice == "Linked List":
    st.header("Linked List")
    list_data = st.text_input("Enter comma-separated values for the Linked List (e.g., 1,2,3)", "1,2,3")
    list_values = [int(x.strip()) for x in list_data.split(',') if x.strip().isdigit()]
    linked_list_head = create_linked_list(list_values)
    st.write("Linked List Representation:")
    st.code(display_linked_list(linked_list_head), language="text")

elif structure_choice == "Doubly Linked List":
    st.header("Doubly Linked List")
    list_data = st.text_input("Enter comma-separated values for the Doubly Linked List (e.g., 1,2,3)", "1,2,3")
    list_values = [int(x.strip()) for x in list_data.split(',') if x.strip().isdigit()]
    doubly_linked_list_head = create_doubly_linked_list(list_values)
    st.write("Doubly Linked List Representation:")
    st.code(display_doubly_linked_list(doubly_linked_list_head), language="text")

elif structure_choice == "Queue":
    st.header("Queue")
    queue_data = st.text_input("Enter comma-separated values to enqueue (e.g., 1,2,3)", "1,2,3")
    queue_values = [int(x.strip()) for x in queue_data.split(',') if x.strip().isdigit()]
    queue = []
    for val in queue_values:
        enqueue(queue, val)
        st.write(f"Enqueued: {val}, Queue: {queue}")
        time.sleep(0.5)
    while queue:
        dequeued_item = dequeue(queue)
        st.write(f"Dequeued: {dequeued_item}, Queue: {queue}")
        time.sleep(0.5)

elif structure_choice == "Stack":
    st.header("Stack")
    stack_data = st.text_input("Enter comma-separated values to push onto the stack (e.g., 1,2,3)", "1,2,3")
    stack_values = [int(x.strip()) for x in stack_data.split(',') if x.strip().isdigit()]
    stack = []
    for val in stack_values:
        push(stack, val)
        st.write(f"Pushed: {val}, Stack: {stack}")
        time.sleep(0.5)
    while stack:
        popped_item = pop(stack)
        st.write(f"Popped: {popped_item}, Stack: {stack}")
        time.sleep(0.5)

elif structure_choice == "Binary Tree":
    st.header("Binary Tree")
    tree_data = st.text_input("Enter comma-separated values for the Binary Tree (use 'None' for empty nodes, e.g., 1,2,3,None,None,4,5)", "1,2,3,None,None,4,5")
    tree_values = [int(x.strip()) if x.strip().isdigit() else None for x in tree_data.split(',')]
    binary_tree_root = create_binary_tree(tree_values)
    st.write("Binary Tree Representation:")
    st.code(display_tree(binary_tree_root), language="text")

elif structure_choice == "Binary Search Tree (BST)":
    st.header("Binary Search Tree (BST)")
    bst_data = st.text_input("Enter comma-separated values to insert into the BST (e.g., 5,3,7,2,4,6,8)", "5,3,7,2,4,6,8")
    bst_values = [int(x.strip()) for x in bst_data.split(',') if x.strip().isdigit()]
    bst_root = None
    for val in bst_values:
        bst_root = insert_bst(bst_root, val)
        st.write(f"Inserted: {val}")
        st.code(display_tree(bst_root), language="text")
        time.sleep(0.5)

    search_val = st.number_input("Enter a value to search for in the BST:", value=5, step=1)
    if st.button("Search"):
        found_node = search_bst(bst_root, search_val)
        if found_node:
            st.write(f"Value {search_val} found in the BST.")
        else:
            st.write(f"Value {search_val} not found in the BST.")

elif structure_choice == "Hash Table":
    st.header("Hash Table")
    hash_table_size = st.number_input("Enter the size of the Hash Table:", min_value=1, value=10, step=1)
    hash_table = [[] for _ in range(hash_table_size)]
    hash_data = st.text_input("Enter comma-separated key-value pairs (e.g., 1:a,2:b,3:c)", "1:a,2:b,3:c")
    hash_pairs = [pair.strip().split(':') for pair in hash_data.split(',') if ':' in pair]
    for key, value in hash_pairs:
        insert_hash_table(hash_table, int(key), value)
        st.write(f"Inserted: Key={key}, Value={value}, Hash Table: {hash_table}")
        time.sleep(0.5)

    search_key = st.number_input("Enter a key to search for in the Hash Table:", value=1, step=1)
    if st.button("Search"):
        found_value = search_hash_table(hash_table, search_key)
        if found_value:
            st.write(f"Value for key {search_key}: {found_value}")
        else:
            st.write(f"Key {search_key} not found in the Hash Table.")

elif structure_choice == "Bloom Filter":
    st.header("Bloom Filter")
    bloom_filter_size = st.number_input("Enter the size of the Bloom Filter:", min_value=1, value=20, step=1)
    bloom_filter = [False] * bloom_filter_size
    hash_functions = [
        lambda item, size: hash(item) % size,
        lambda item, size: (hash(item) >> 8) % size,
        lambda item, size: (hash(item) >> 16) % size
    ]
    bloom_data = st.text_input("Enter comma-separated values to add to the Bloom Filter (e.g., apple,banana,cherry)", "apple,banana,cherry")
    bloom_values = [x.strip() for x in bloom_data.split(',')]
    for val in bloom_values:
        add_to_bloom_filter(bloom_filter, val, hash_functions)
        st.write(f"Added: {val}, Bloom Filter: {bloom_filter}")
        time.sleep(0.5)

    check_val = st.text_input("Enter a value to check in the Bloom Filter:", "apple")
    if st.button("Check"):
        if check_bloom_filter(bloom_filter, check_val, hash_functions):
            st.write(f"'{check_val}' might be in the Bloom Filter.")
        else:
            st.write(f"'{check_val}' is definitely not in the Bloom Filter.")

elif structure_choice == "Skip List":
    st.header("Skip List")
    max_level = st.number_input("Enter the maximum level for the Skip List:", min_value=1, value=3, step=1)
    skip_list_data = st.text_input("Enter comma-separated key-value pairs (e.g., 1:a,2:b,3:c)", "1:a,2:b,3:c")
    skip_list_pairs = [pair.strip().split(':') for pair in skip_list_data.split(',') if ':' in pair]
    skip_list = []
    for key, value in skip_list_pairs:
        insert_skip_list(skip_list, int(key), value, max_level)
        st.write(f"Inserted: Key={key}, Value={value}, Skip List: {skip_list}")
        time.sleep(1)

    search_key = st.number_input("Enter a key to search for in the Skip List:", value=1, step=1)
    if st.button("Search"):
        found_value = search_skip_list(skip_list, search_key)
        if found_value:
            st.write(f"Value for key {search_key}: {found_value}")
        else:
            st.write(f"Key {search_key} not found in the Skip List.")

elif structure_choice == "LRU Cache":
    st.header("LRU Cache")
    cache_capacity = st.number_input("Enter the capacity of the LRU Cache:", min_value=1, value=5, step=1)
    lru_cache = LRUCache(cache_capacity)
    lru_data = st.text_input("Enter comma-separated key-value pairs (e.g., 1:a,2:b,3:c)", "1:a,2:b,3:c")
    lru_pairs = [pair.strip().split(':') for pair in lru_data.split(',') if ':' in pair]
    for key, value in lru_pairs:
        lru_cache.put(int(key), value)
        st.write(f"Put: Key={key}, Value={value}, Cache: {lru_cache.cache}, Queue: {lru_cache.queue}")
        time.sleep(1)

    get_key = st.number_input("Enter a key to get from the LRU Cache:", value=1, step=1)
    if st.button("Get"):
        retrieved_value = lru_cache.get(get_key)
        if retrieved_value:
            st.write(f"Value for key {get_key}: {retrieved_value}, Cache: {lru_cache.cache}, Queue: {lru_cache.queue}")
        else:
            st.write(f"Key {get_key} not found in the Cache.")

elif structure_choice == "Disjoint Sets":
     st.header("Disjoint Sets (Union-Find)")
     num_elements = st.number_input("Enter the number of elements:", min_value=1, value=5, step=1)
     parent = list(range(num_elements))
     rank = [0] * num_elements
     st.write(f"Initial Sets: {[ {i} ] for i in range(num_elements)]")

     union_pairs_str = st.text_input("Enter comma-separated pairs to union (e.g., 0,1;2,3)", "0,1;2,3")
     union_pairs = [pair.split(',') for pair in union_pairs_str.split(';') if ',' in pair]
     for pair in union_pairs:
         if len(pair) == 2 and pair[0].isdigit() and pair[1].isdigit():
             x, y = int(pair[0]), int(pair[1])
             union_sets(parent, rank, x, y)
             st.write(f"Union ({x}, {y}): Sets = {[find_set(parent, i) for i in range(num_elements)]}")
             time.sleep(1)

     check_pair_str = st.text_input("Enter a pair to check if they are in the same set (e.g., 0,1)", "0,1")
     check_pair = check_pair_str.split(',')
     if len(check_pair) == 2 and check_pair[0].isdigit() and check_pair[1].isdigit():
         a, b = int(check_pair[0]), int(check_pair[1])
         if st.button("Check Set"):
             if find_set(parent, a) == find_set(parent, b):
                 st.write(f"{a} and {b} are in the same set.")
             else:
                 st.write(f"{a} and {b} are in different sets.")

elif structure_choice == "AVL Tree (Conceptual)":
    st.header("AVL Tree (Conceptual Representation)")
    st.write("AVL trees are self-balancing BSTs.  Visualizing the rotations and balancing in a simple text-based interface is challenging.  This section will show the insertion sequence and a conceptual representation of the tree structure. For a full graphical representation, a dedicated visualization tool is recommended.")
    avl_data = st.text_input("Enter comma-separated values to insert into the AVL Tree (e.g., 10,20,30,40,50,25)", "10,20,30,40,50,25")
    avl_values = [int(x.strip()) for x in avl_data.split(',') if x.strip().isdigit()]
    avl_root = None
    for val in avl_values:
        avl_root = insert_avl(avl_root, val)
        st.write(f"Inserted: {val}")
        st.code(display_tree(avl_root), language="text")
        time.sleep(1)

    search_val = st.number_input("Enter a value to search for in the AVL Tree:", value=10, step=1)
    if st.button("Search AVL"):
        found_node = search_avl(avl_root, search_val)
        if found_node:
            st.write(f"Value {search_val} found in the AVL Tree.")
        else:
            st.write(f"Value {search_val} not found.")

elif structure_choice == "Red-Black Tree (Conceptual)":
    st.header("Red-Black Tree (Conceptual Representation)")
    st.write("Red-Black trees are self-balancing BSTs with specific coloring rules. Visualizing the coloring and balancing operations in detail is complex for a text-based interface. This section shows the insertion sequence and a conceptual tree structure. For a full graphical representation, a dedicated visualization tool is recommended.")
    rb_data = st.text_input("Enter comma-separated values to insert (e.g., 10,18,5,31,12,2)", "10,18,5,31,12,2")
    rb_values = [int(x.strip()) for x in rb_data.split(',') if x.strip().isdigit()]
    rb_root = None
    for val in rb_values:
        rb_root = insert_red_black_tree(rb_root, val)
        st.write(f"Inserted: {val}")
        st.code(display_tree(rb_root), language="text")
        time.sleep(1)

    search_val = st.number_input("Search for a value in the Red-Black Tree:", value=10, step=1)
    if st.button("Search Red-Black"):
         found_node = search_red_black_tree(rb_root, search_val)
         if found_node:
            st.write(f"Value {search_val} found in the Red-Black Tree.")
         else:
            st.write(f"Value {search_val} not found.")

elif structure_choice == "B-Tree (Conceptual)":
    st.header("B-Tree (Conceptual Representation)")
    st.write("B-Trees are self-balancing tree structures, often used for disk-based storage.  Visualizing their structure, especially with multiple children per node, is not easily done in a text-based interface.  This section provides a simplified conceptual view. A dedicated visualization tool is recommended for a full graphical representation.")
    b_data = st.text_input("Enter comma-separated values (e.g., 10,20,30,15,25,5)", "10,20,30,15,25,5")
    b_values = [int(x.strip()) for x in b_data.split(',') if x.strip().isdigit()]
    b_root = None
    for val in b_values:
        b_root = insert_b_tree(b_root, val)
        st.write(f"Inserted: {val}")
        st.code(display_tree(b_root), language="text")
        time.sleep(1)

    search_val = st.number_input("Search for a value in the B-Tree:", value=10, step=1)
    if st.button("Search B-Tree"):
        found_node = search_b_tree(b_root, search_val)
        if found_node:
            st.write(f"Value {search_val} found in the B-Tree.")
        else:
            st.write(f"Value {search_val} not found.")
