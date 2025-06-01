import streamlit as st
import pandas as pd
import graphviz

# --- Helper Functions for Binary Search Tree (BST) ---
# This class defines a node in our BST.
class BSTNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# This function inserts a new value into the BST.
def insert_bst_node(root, value):
    if root is None:
        return BSTNode(value)
    if value < root.value:
        root.left = insert_bst_node(root.left, value)
    elif value > root.value: # No duplicates for simplicity
        root.right = insert_bst_node(root.right, value)
    return root

# This function generates the 'dot' language string for Graphviz to draw the BST.
def generate_bst_dot(node, dot_str=""):
    if node is None:
        return dot_str
    # Define the node in Graphviz
    dot_str += f'    node{node.value} [label="{node.value}"];\n'
    # Add edges to children
    if node.left:
        dot_str += f'    node{node.value} -> node{node.left.value} [label="L", color="darkgreen"];\n'
        dot_str = generate_bst_dot(node.left, dot_str)
    if node.right:
        dot_str += f'    node{node.value} -> node{node.right.value} [label="R", color="darkred"];\n'
        dot_str = generate_bst_dot(node.right, dot_str)
    return dot_str

# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="Data Structure Visualizer")

st.title("📊 Interactive Data Structure Visualizer")
st.write("""
This application helps you understand fundamental **data structures** by letting you interact with them and see their visualizations.
Use the **sidebar** to select a data structure and its operations.
""")

# --- Initialize Session State for Data Structures and Input Keys ---
# Session state ensures that the data persists across user interactions.
if 'array_data' not in st.session_state:
    st.session_state.array_data = []
if 'linked_list_nodes' not in st.session_state:
    st.session_state.linked_list_nodes = [] # Stores (value, unique_id) for Graphviz
if 'stack_data' not in st.session_state:
    st.session_state.stack_data = []
if 'queue_data' not in st.session_state:
    st.session_state.queue_data = []
if 'bst_root' not in st.session_state:
    st.session_state.bst_root = None

# Input field clear counters - crucial for clearing input fields after submission
if 'array_input_key' not in st.session_state:
    st.session_state.array_input_key = 0
if 'll_input_key' not in st.session_state:
    st.session_state.ll_input_key = 0
if 'stack_input_key' not in st.session_state:
    st.session_state.stack_input_key = 0
if 'queue_input_key' not in st.session_state:
    st.session_state.queue_input_key = 0
if 'bst_input_key' not in st.session_state:
    st.session_state.bst_input_key = 0 # For number_input too

# --- Sidebar for Navigation ---
st.sidebar.header("Select Data Structure")
selected_ds = st.sidebar.radio(
    "Choose a structure to visualize:",
    ("Array/List", "Linked List", "Stack", "Queue", "Binary Search Tree")
)

st.sidebar.markdown("---")
st.sidebar.info("Use the controls below to interact with the selected data structure.")

# --- Main Content Area based on Selection ---

# --- Array/List Visualization ---
if selected_ds == "Array/List":
    st.header("1. Array / List")
    st.write("""
    An **Array** (or **List** in Python) is a collection of items stored at **contiguous memory locations**.
    It's one of the simplest and most fundamental data structures, providing **direct access** to elements using their index.
    """)

    st.subheader("Key Properties:")
    st.markdown("""
    * **Contiguous Memory:** Elements are stored in a single, unbroken block of memory.
    * **Direct Access (by Index):** You can access any element very quickly ($O(1)$) by knowing its position (index).
    * **Dynamic Sizing (Python Lists)::warning: Python lists can grow or shrink. When they grow beyond their current capacity, a new, larger block of memory is allocated, and elements are copied over.
    * **Insertion/Deletion (Middle):** Inserting or deleting elements in the middle can be slow ($O(N)$) because subsequent elements might need to be shifted.
    """)

    st.subheader("Interactive Example:")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        new_element = st.text_input("Enter element to add:", key=f"array_input_{st.session_state.array_input_key}")
        if st.button("Add Element to End", key="add_array_btn"):
            if new_element:
                st.session_state.array_data.append(new_element)
                st.session_state.array_input_key += 1 # Increment key to clear input
                st.rerun() # Rerun to apply new key and clear input
            else:
                st.warning("Please enter an element to add.")
        if st.button("Clear Array", key="clear_array_btn"):
            st.session_state.array_data = []
            st.info("Array cleared!")
            st.rerun() # Rerun to update visualization

    with col2:
        st.subheader("Visualization")
        if st.session_state.array_data:
            # Display using columns for a clear indexed view
            cols_viz = st.columns(len(st.session_state.array_data))
            st.markdown("<p style='text-align: center; font-weight: bold;'>Array Elements:</p>", unsafe_allow_html=True)
            for i, val in enumerate(st.session_state.array_data):
                with cols_viz[i]:
                    st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; margin: 5px; text-align: center; background-color: #e0f7fa; border-radius: 5px;'><b>[{i}]</b><br>{val}</div>", unsafe_allow_html=True)
            st.write(f"**Current Size:** {len(st.session_state.array_data)} elements")
        else:
            st.info("Array is empty. Add some elements to visualize!")

# --- Linked List Visualization ---
elif selected_ds == "Linked List":
    st.header("2. Linked List (Singly)")
    st.write("""
    A **Linked List** is a linear collection of data elements, called **nodes**, where each node
    contains data and a **pointer** (or reference) to the **next node** in the sequence.
    Elements are **not necessarily contiguous** in memory.
    """)

    st.subheader("Key Properties:")
    st.markdown("""
    * **Non-Contiguous Memory:** Nodes can be scattered throughout memory, connected by pointers.
    * **Dynamic Size:** Easily grows or shrinks as needed without requiring large contiguous blocks.
    * **Efficient Insertion/Deletion (Known Node):** Adding or removing elements is fast ($O(1)$) if you have a reference to the node before the insertion/deletion point.
    * **Inefficient Random Access:** To find an element by its position, you must traverse the list from the beginning ($O(N)$).
    """)

    st.subheader("Interactive Example:")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        ll_element = st.text_input("Enter element to add (at end):", key=f"ll_input_{st.session_state.ll_input_key}")
        if st.button("Add Element to End", key="add_ll_btn"):
            if ll_element:
                node_id = len(st.session_state.linked_list_nodes) # Unique ID for Graphviz nodes
                st.session_state.linked_list_nodes.append((ll_element, node_id))
                st.session_state.ll_input_key += 1 # Increment key to clear input
                st.rerun() # Rerun to apply new key and clear input
            else:
                st.warning("Please enter an element.")
        if st.button("Remove from Front", key="remove_ll_btn"):
            if st.session_state.linked_list_nodes:
                removed_val, _ = st.session_state.linked_list_nodes.pop(0) # Remove the first element
                st.success(f"Removed: **{removed_val}**")
                st.rerun() # Rerun to update visualization
            else:
                st.warning("Linked List is empty. Cannot remove.")
        if st.button("Clear Linked List", key="clear_ll_btn"):
            st.session_state.linked_list_nodes = []
            st.info("Linked List cleared!")
            st.rerun() # Rerun to update visualization

    with col2:
        st.subheader("Visualization")
        if st.session_state.linked_list_nodes:
            # Generate Graphviz DOT language code
            dot_code = 'digraph G {\n rankdir="LR"; node [shape=record, style=filled, fillcolor="#FFECB3", fontname="Arial"];\n edge [arrowhead=normal, color="#FF9800", penwidth=1.5];\n'
            for i, (val, node_id) in enumerate(st.session_state.linked_list_nodes):
                next_ptr_target = ""
                if i + 1 < len(st.session_state.linked_list_nodes):
                    next_ptr_target = f"node{st.session_state.linked_list_nodes[i+1][1]}:f0"
                else:
                    next_ptr_target = "null_node" # Point to a visual NULL

                dot_code += f'    node{node_id} [label="<f0> {val} | <f1> "];\n'
                if next_ptr_target: # Only draw edge if there's a next node or a null
                    dot_code += f'    node{node_id}:f1 -> {next_ptr_target};\n'

            dot_code += '    null_node [shape=plaintext, label="NULL", fontcolor="red"];\n' # Define NULL visually
            dot_code += '}\n'
            st.graphviz_chart(dot_code)
            st.write(f"**Current Size:** {len(st.session_state.linked_list_nodes)} nodes")
        else:
            st.info("Linked List is empty. Add some elements to visualize!")

# --- Stack Visualization ---
elif selected_ds == "Stack":
    st.header("3. Stack")
    st.write("""
    A **Stack** is a linear data structure that follows the **LIFO (Last In, First Out)** principle.
    Imagine a stack of plates: the last plate you put on is always the first one you take off.
    """)

    st.subheader("Key Properties:")
    st.markdown("""
    * **LIFO (Last In, First Out):** The most recently added element is the first one to be removed.
    * **Primary Operations:**
        * **`push()`:** Adds an element to the top of the stack.
        * **`pop()`:** Removes and returns the top element from the stack.
        * **`peek()` / `top()`:** Returns the top element without removing it.
    * **Applications:** Function call stacks in programming, undo/redo features, expression evaluation.
    """)

    st.subheader("Interactive Example:")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        stack_element = st.text_input("Enter element to push:", key=f"stack_input_{st.session_state.stack_input_key}")
        if st.button("Push (Add to Top)", key="push_stack_btn"):
            if stack_element:
                st.session_state.stack_data.append(stack_element)
                st.session_state.stack_input_key += 1 # Increment key
                st.rerun() # Rerun to apply new key
            else:
                st.warning("Please enter an element to push.")
        if st.button("Pop (Remove from Top)", key="pop_stack_btn"):
            if st.session_state.stack_data:
                popped_val = st.session_state.stack_data.pop()
                st.success(f"Popped: **{popped_val}**")
                st.rerun() # Rerun to update visualization
            else:
                st.warning("Stack is empty. Cannot pop.")
        if st.button("Clear Stack", key="clear_stack_btn"):
            st.session_state.stack_data = []
            st.info("Stack cleared!")
            st.rerun() # Rerun to update visualization

    with col2:
        st.subheader("Visualization")
        if st.session_state.stack_data:
            st.markdown("<p style='text-align: center; font-weight: bold;'>Stack (TOP is highest):</p>", unsafe_allow_html=True)
            # Display from top to bottom
            for i, val in enumerate(reversed(st.session_state.stack_data)):
                if i == 0: # This is the top element
                    st.markdown(f"<div style='border: 2px solid #4CAF50; padding: 10px; margin: 2px auto; text-align: center; background-color: #e8f5e9; border-radius: 5px; width: 50%;'><b>{val}</b> <span style='font-size: 0.8em; color: green;'>(TOP)</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px; margin: 2px auto; text-align: center; background-color: #f0f0f0; border-radius: 5px; width: 50%;'>{val}</div>", unsafe_allow_html=True)
            st.markdown("<div style='border-top: 3px double #333; margin: 5px auto; width: 50%;'></div>", unsafe_allow_html=True) # Bottom of the stack
            st.write(f"**Current Size:** {len(st.session_state.stack_data)} elements")
        else:
            st.info("Stack is empty. Push some elements to visualize!")

# --- Queue Visualization ---
elif selected_ds == "Queue":
    st.header("4. Queue")
    st.write("""
    A **Queue** is a linear data structure that follows the **FIFO (First In, First Out)** principle.
    Think of a line of people waiting for a service: the first person to join the line is the first one served.
    """)

    st.subheader("Key Properties:")
    st.markdown("""
    * **FIFO (First In, First Out):** The first element added is the first one to be removed.
    * **Primary Operations:**
        * **`enqueue()`:** Adds an element to the **rear** (back) of the queue.
        * **`dequeue()`:** Removes and returns the element from the **front** of the queue.
        * **`front()` / `peek()`:** Returns the front element without removing it.
    * **Applications:** Printer job scheduling, CPU scheduling, breadth-first search algorithms.
    """)

    st.subheader("Interactive Example:")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        queue_element = st.text_input("Enter element to enqueue:", key=f"queue_input_{st.session_state.queue_input_key}")
        if st.button("Enqueue (Add to Rear)", key="enqueue_queue_btn"):
            if queue_element:
                st.session_state.queue_data.append(queue_element)
                st.session_state.queue_input_key += 1 # Increment key
                st.rerun() # Rerun to apply new key
            else:
                st.warning("Please enter an element to enqueue.")
        if st.button("Dequeue (Remove from Front)", key="dequeue_queue_btn"):
            if st.session_state.queue_data:
                dequeued_val = st.session_state.queue_data.pop(0) # Python list pop(0) removes from front
                st.success(f"Dequeued: **{dequeued_val}**")
                st.rerun() # Rerun to update visualization
            else:
                st.warning("Queue is empty. Cannot dequeue.")
        if st.button("Clear Queue", key="clear_queue_btn"):
            st.session_state.queue_data = []
            st.info("Queue cleared!")
            st.rerun() # Rerun to update visualization

    with col2:
        st.subheader("Visualization")
        if st.session_state.queue_data:
            st.markdown("<p style='text-align: center; font-weight: bold;'>Queue (FRONT is left, REAR is right):</p>", unsafe_allow_html=True)
            # Custom CSS for better visualization of queue elements
            st.markdown("""
            <style>
            .queue-container {
                display: flex;
                align-items: center;
                justify-content: center;
                flex-wrap: wrap;
                margin-top: 10px;
            }
            .queue-element {
                border: 1px solid #64B5F6; /* Light Blue */
                background-color: #E3F2FD; /* Lighter Blue */
                padding: 10px 15px;
                margin: 5px;
                border-radius: 5px;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 50px;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
            }
            .queue-arrow {
                font-size: 1.5em;
                margin: 0 5px;
                color: #424242;
            }
            </style>
            """, unsafe_allow_html=True)

            html_display = "<div class='queue-container'>"
            if st.session_state.queue_data:
                html_display += "<span class='queue-arrow'>FRONT &larr;</span>"
                for i, val in enumerate(st.session_state.queue_data):
                    html_display += f"<div class='queue-element'>{val}</div>"
                    if i < len(st.session_state.queue_data) - 1:
                        html_display += "<span class='queue-arrow'>&rarr;</span>"
                html_display += "<span class='queue-arrow'>&rarr; REAR</span>"
            html_display += "</div>"
            st.markdown(html_display, unsafe_allow_html=True)
            st.write(f"**Current Size:** {len(st.session_state.queue_data)} elements")
        else:
            st.info("Queue is empty. Enqueue some elements to visualize!")

# --- Binary Search Tree Visualization ---
elif selected_ds == "Binary Search Tree":
    st.header("5. Binary Search Tree (BST)")
    st.write("""
    A **Binary Search Tree (BST)** is a tree-based data structure where each node has at most two children
    (a left child and a right child). It maintains a specific ordering property:
    * All values in the **left subtree** of a node are **smaller** than the node's value.
    * All values in the **right subtree** of a node are **larger** than the node's value.
    """)

    st.subheader("Key Properties:")
    st.markdown("""
    * **Ordered Structure:** This property allows for efficient searching.
    * **Efficient Search, Insertion, Deletion:** In a balanced BST, these operations take $O(\log N)$ time on average. In the worst-case (e.g., inserting sorted numbers), it can degrade to $O(N)$, resembling a linked list.
    * **No Duplicates:warning: Typically, BSTs do not allow duplicate values.
    * **Applications:** Implementing dictionaries/maps, priority queues, efficient sorting.
    """)

    st.subheader("Interactive Example:")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        bst_value = st.number_input("Enter integer value to insert:", step=1, value=0, key=f"bst_input_{st.session_state.bst_input_key}")
        if st.button("Insert Value", key="insert_bst_btn"):
            if bst_value is not None:
                st.session_state.bst_root = insert_bst_node(st.session_state.bst_root, int(bst_value))
                st.session_state.bst_input_key += 1 # Increment key
                st.rerun() # Rerun to apply new key
                st.success(f"Inserted: **{int(bst_value)}**") # Success message after rerun
            else:
                st.warning("Please enter an integer value.")
        if st.button("Clear BST", key="clear_bst_btn"):
            st.session_state.bst_root = None
            st.info("BST cleared!")
            st.rerun() # Rerun to update visualization

    with col2:
        st.subheader("Visualization")
        if st.session_state.bst_root:
            # Generate Graphviz DOT code for the current BST
            dot_code = 'digraph G {\n rankdir="TB"; node [shape=circle, style=filled, fillcolor="#C8E6C9", fontname="Arial"];\n edge [color="#388E3C"];\n'
            dot_code = generate_bst_dot(st.session_state.bst_root, dot_code)
            dot_code += '}\n'
            st.graphviz_chart(dot_code)
        else:
            st.info("BST is empty. Insert some integer values to visualize!")

st.sidebar.markdown("---")
st.sidebar.caption("Data Structure Visualizer v1.0")

st.markdown("---")
st.info("This application is designed for educational purposes to demonstrate basic data structure concepts and operations. For robust, production-ready data structure implementations, consider using Python's built-in types (like `list`, `dict`) or specialized libraries.")
