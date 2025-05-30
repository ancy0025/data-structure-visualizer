import streamlit as st
import graphviz

# --- Data Structure Definition: Linked List ---
class Node:
    """Represents a single node in the linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    """Represents a singly linked list."""
    def __init__(self):
        self.head = None

    def append(self, data):
        """Appends a new node with the given data to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def to_graphviz(self):
        """
        Generates a Graphviz Digraph object representing the linked list.
        This allows Streamlit to render the visualization.
        """
        # Create a directed graph with left-to-right rank direction
        dot = graphviz.Digraph(comment='Linked List', graph_attr={'rankdir': 'LR'})

        # Add a 'HEAD' node to indicate the start of the list
        dot.node('head', 'HEAD', shape='plaintext', fontname='Inter')

        current = self.head
        prev_node_id = 'head' # Start linking from the HEAD

        node_count = 0 # Unique identifier for each node

        # Handle empty list case
        if not current:
            dot.node('null_end', 'NULL', shape='plaintext', fontname='Inter')
            dot.edge('head', 'null_end')

        # Traverse the linked list and add nodes and edges to the graph
        while current:
            node_id = f'node_{node_count}'
            # Add a node for the current data element
            dot.node(node_id, str(current.data), shape='box', style='filled', fillcolor='#F0F8FF', fontname='Inter', fontsize='14')

            # Create an edge from the previous element (or HEAD) to the current node
            dot.edge(prev_node_id, node_id, arrowhead='normal', color='#4CAF50')

            prev_node_id = node_id # Update previous node for the next iteration
            current = current.next
            node_count += 1

        # After the loop, if there were nodes, link the last node to 'NULL'
        if prev_node_id != 'head': # Ensures this only runs if list is not empty
            dot.node('null_end', 'NULL', shape='plaintext', fontname='Inter')
            dot.edge(prev_node_id, 'null_end', arrowhead='normal', color='#FF6347')

        return dot

# --- Streamlit Application ---

def main():
    st.set_page_config(
        page_title="Data Structure Visualizer",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("📊 Linked List Visualizer")
    st.markdown(
        """
        This application visualizes a singly linked list.
        Add elements to see how the list grows and changes.
        """
    )

    # Initialize linked list in session state if not already present
    if 'linked_list' not in st.session_state:
        st.session_state.linked_list = LinkedList()

    # Input for adding new elements
    col1, col2 = st.columns([3, 1])
    with col1:
        new_element = st.text_input("Enter element to add:", key="element_input")
    with col2:
        st.write("") # For vertical alignment
        st.write("")
        if st.button("Add Element", use_container_width=True):
            if new_element:
                st.session_state.linked_list.append(new_element)
                st.success(f"Added '{new_element}' to the linked list.")
                st.session_state.element_input = "" # Clear the input field
            else:
                st.warning("Please enter an element to add.")

    st.markdown("---")

    # Display the current state of the linked list
    st.subheader("Current Linked List Visualization")
    if st.session_state.linked_list.head:
        # Get the Graphviz object and render it
        graph = st.session_state.linked_list.to_graphviz()
        st.graphviz_chart(graph)
    else:
        st.info("The linked list is currently empty. Add an element to see the visualization!")
        # Show an empty graph for clarity
        empty_graph = graphviz.Digraph(comment='Empty Linked List', graph_attr={'rankdir': 'LR'})
        empty_graph.node('head', 'HEAD', shape='plaintext', fontname='Inter')
        empty_graph.node('null_end', 'NULL', shape='plaintext', fontname='Inter')
        empty_graph.edge('head', 'null_end')
        st.graphviz_chart(empty_graph)


    st.markdown("---")

    # Clear button
    if st.button("Clear Linked List", type="secondary"):
        st.session_state.linked_list = LinkedList()
        st.success("Linked list cleared!")

    st.markdown(
        """
        <style>
        .stButton>button {
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
