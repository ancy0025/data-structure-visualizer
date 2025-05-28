import streamlit as st
import random
import collections # For deque, ideal for queues

st.set_page_config(page_title="Data Structure Visualizer", layout="wide")
st.title("📚 Interactive Data Structure Visualizer")
st.markdown("Explore how basic data structures work by adding and removing elements.")

# --- Session State Initialization for Data Structures ---
if "my_list" not in st.session_state:
    st.session_state.my_list = []
if "my_bst" not in st.session_state:
    st.session_state.my_bst = None # Represents the root of the BST
if "my_stack" not in st.session_state:
    st.session_state.my_stack = [] # Using a list for stack
if "my_queue" not in st.session_state:
    st.session_state.my_queue = collections.deque() # Using collections.deque for efficient queue

# --- Helper Functions for Binary Search Tree (BST) ---
# Function to insert a node into the BST
def insert_bst_node(root, value):
    if root is None:
        return {"value": value, "left": None, "right": None}
    if value < root["value"]:
        root["left"] = insert_bst_node(root["left"], value)
    else: # Allow duplicates to go to the right
        root["right"] = insert_bst_node(root["right"], value)
    return root

# Function to display the BST in a hierarchical text format
def display_bst_node(node, prefix="", is_left=None):
    if node is not None:
        connector = ""
        if is_left is True:
            connector = "├── L: "
        elif is_left is False:
            connector = "└── R: "
        elif is_left is None: # Root node
            connector = "Root: "

        st.write(f"{prefix}{connector}**{node['value']}**")

        # Prepare for children
        new_prefix = prefix + ("│   " if is_left is not False else "    ") # 'is not False' covers True and None (for root)

        # Recursively display left child
        if node["left"]:
            display_bst_node(node["left"], new_prefix, True)
        else: # Show null left if it exists
            st.write(f"{new_prefix}├── L: (null)")

        # Recursively display right child
        if node["right"]:
            display_bst_node(node["right"], new_prefix, False)
        else: # Show null right if it exists
            st.write(f"{new_prefix}└── R: (null)")


# --- List Visualization Section ---
st.header("1. Python List (Array-like)")
st.markdown("A dynamic array that can grow and shrink. Elements are ordered.")

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
        st.json(st.session_state.my_list) # Shows the list in JSON format
        st.write(f"**Current Size:** `{len(st.session_state.my_list)}`")
    else:
        st.info("The list is currently empty.")

st.write("---")

# --- Stack (LIFO) Visualization Section ---
st.header("2. Stack (LIFO - Last In, First Out)")
st.markdown("A collection where elements are added and removed from the same end (the 'top'). Think of a stack of plates.")

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
            st.info(f"Top element: {st.session_state.my_stack[-1]}")
        else:
            st.warning("Stack is empty!")
    if st.button("Clear Stack", key="stack_clear_btn"):
        st.session_state.my_stack = []
        st.error("Stack cleared.")

with stack_col2:
    st.markdown("**Current Stack State (Top is at the end):**")
    if st.session_state.my_stack:
        # Displaying vertically to better represent a stack
        st.markdown("```")
        for item in reversed(st.session_state.my_stack):
            st.markdown(f"| {item:<10} |") # Adjust spacing for visualization
            st.markdown("------------")
        st.markdown(f"| (BOTTOM) |")
        st.markdown("```")
        st.write(f"**Current Size:** `{len(st.session_state.my_stack)}`")
    else:
        st.info("The stack is currently empty.")

st.write("---")

# --- Queue (FIFO) Visualization Section ---
st.header("3. Queue (FIFO - First In, First Out)")
st.markdown("A collection where elements are added to one end ('rear') and removed from the other end ('front'). Think of a waiting line.")

queue_col1, queue_col2 = st.columns([1, 3])

with queue_col1:
    st.markdown("**Operations:**")
    new_queue_val = st.number_input("Value to Enqueue", value=random.randint(1, 100), key="queue_enqueue_input")
    if st.button("Enqueue (Add to Rear)", key="queue_enqueue_btn"):
        st.session_state.my_queue.append(new_queue_val)
        st.success(f"Enqueued {new_queue_val} to the queue.")
    if st.button("Dequeue (Remove from Front)", key="queue_dequeue_btn"):
        if st.session_state.my_queue:
            dequeued_val = st.session_state.my_queue.popleft() # Efficient pop from left for deque
            st.info(f"Dequeued {dequeued_val} from the queue.")
        else:
            st.warning("Queue is empty! Nothing to dequeue.")
    if st.button("Front (View Front)", key="queue_front_btn"):
        if st.session_state.my_queue:
            st.info(f"Front element: {st.session_state.my_queue[0]}")
        else:
            st.warning("Queue is empty!")
    if st.button("Clear Queue", key="queue_clear_btn"):
        st.session_state.my_queue.clear()
        st.error("Queue cleared.")

with queue_col2:
    st.markdown("**Current Queue State (Front is on the left):**")
    if st.session_state.my_queue:
        # Displaying horizontally
        queue_str = " <-- ".join(map(str, st.session_state.my_queue))
        st.markdown(f"**FRONT** <-- `{queue_str}` <-- **REAR**")
        st.write(f"**Current Size:** `{len(st.session_state.my_queue)}`")
    else:
        st.info("The queue is currently empty.")

st.write("---")

# --- Binary Search Tree (BST) Visualization Section ---
st.header("4. Binary Search Tree (BST) (Conceptual)")
st.markdown("A tree-based data structure where each node has at most two children. For every node, values in the left subtree are smaller, and values in the right subtree are larger.")

bst_col1, bst_col2 = st.columns([1, 3])

with bst_col1:
    st.markdown("**Operations:**")
    new_bst_val = st.number_input("Value to Insert", value=random.randint(1, 100), key="bst_insert_input")
    if st.button("Insert Node", key="bst_insert_btn"):
        st.session_state.my_bst = insert_bst_node(st.session_state.my_bst, new_bst_val)
        st.success(f"Inserted node with value {new_bst_val}.")
    if st.button("Clear BST", key="bst_clear_btn"):
        st.session_state.my_bst = None
        st.error("Binary Search Tree cleared.")

with bst_col2:
    st.markdown("**Current BST Structure (JSON representation):**")
    if st.session_state.my_bst:
        st.json(st.session_state.my_bst) # Shows the raw dictionary/JSON structure
        st.subheader("Visualized Tree Structure (Simplified Text View):")
        display_bst_node(st.session_state.my_bst) # Shows the hierarchical text view
    else:
        st.info("The Binary Search Tree is currently empty. Insert a node to start building it!")

st.write("---")
st.markdown("Created with Streamlit for Data Structure Visualization")
