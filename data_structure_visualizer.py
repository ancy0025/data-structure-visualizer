import streamlit as st
import pandas as pd

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Memory Hierarchy Visualizer")

st.title("🗃️ Computer Memory Hierarchy Visualizer")

st.write("""
This application illustrates the hierarchical structure of computer memory,
highlighting the fundamental trade-offs between **speed**, **size**, and **cost**.
Understanding this hierarchy is essential for optimizing software performance
and comprehending how modern computer systems operate.
""")

# --- Data Definition for Memory Levels ---
# Using a dictionary for structured data
memory_data = {
    "Level": [
        "Registers",
        "L1 Cache",
        "L2 Cache",
        "L3 Cache",
        "Main Memory (RAM)",
        "Secondary Storage (SSD/HDD)"
    ],
    "Typical Size": [
        "Few Bytes (e.g., 32-256 bytes)",
        "Tens to Hundreds of KB (e.g., 32KB - 512KB per core)",
        "Hundreds of KB to MBs (e.g., 256KB - 8MB per core/shared)",
        "Several MBs (e.g., 4MB - 64MB, shared)",
        "Gigabytes (e.g., 8GB - 128GB)",
        "Terabytes (e.g., 500GB - 16TB+)"
    ],
    "Access Speed (Approx. Latency)": [
        "0.1 - 1 ns (CPU Cycle)",
        "0.5 - 5 ns",
        "5 - 20 ns",
        "20 - 60 ns",
        "50 - 100 ns",
        "100 µs - 1 ms (SSD) / 1 - 10 ms (HDD)" # µs = microseconds, ms = milliseconds
    ],
    "Cost per bit (Relative)": [
        "Highest",
        "Very High",
        "High",
        "Medium-High",
        "Medium",
        "Lowest"
    ],
    "Volatility": [
        "Volatile (data lost on power off)",
        "Volatile",
        "Volatile",
        "Volatile",
        "Volatile",
        "Non-Volatile (data persists on power off)"
    ],
    "Distance from CPU": [
        "Inside CPU die",
        "Inside CPU die (closest)",
        "Inside CPU die (close)",
        "On CPU die (further)",
        "External chip (motherboard)",
        "External drive (separate device)"
    ]
}

df = pd.DataFrame(memory_data)

st.markdown("---")
st.header("1. Memory Hierarchy Diagram")

st.write("This diagram illustrates the flow of data requests from the CPU down to the slower, larger storage levels.")
st.graphviz_chart('''
    digraph memory_hierarchy {
        rankdir=TB; # Top-to-Bottom ranking
        node [shape=box, style=filled, fillcolor="#ADD8E6", fontname="Arial", fontsize=12]; # Light Blue nodes
        edge [arrowhead=normal, style=bold, color="#4682B4", penwidth=1.5]; # Steel Blue edges

        # Define Nodes
        CPU [label="CPU\n(Processor)", shape=parallelogram, fillcolor="#FFD700"]; # Gold for CPU
        Registers [label="Registers\n(Fastest, Smallest)", fillcolor="#90EE90"]; # Light Green
        L1_Cache [label="L1 Cache\n(SRAM)"];
        L2_Cache [label="L2 Cache\n(SRAM)"];
        L3_Cache [label="L3 Cache\n(SRAM)"];
        Main_Memory [label="Main Memory\n(DRAM)", fillcolor="#F08080"]; # Light Coral
        Secondary_Storage [label="Secondary Storage\n(SSD/HDD, Slowest, Largest)", fillcolor="#D3D3D3"]; # Light Grey

        # Define Connections (Hierarchy Flow)
        CPU -> Registers [label="Direct Access"];
        Registers -> L1_Cache [label="Data Path/Next Level"];
        L1_Cache -> L2_Cache [label="Hierarchical Access"];
        L2_Cache -> L3_Cache [label="Hierarchical Access"];
        L3_Cache -> Main_Memory [label="Hierarchical Access"];
        Main_Memory -> Secondary_Storage [label="Page Swapping/File I/O"];

        # Grouping (optional, but helps with layout)
        {rank=same; CPU; Registers;}
        {rank=same; L1_Cache; L2_Cache; L3_Cache;}
        {rank=same; Main_Memory;}
        {rank=same; Secondary_Storage;}

        # Legend/Principles
        subgraph cluster_principles {
            label="Key Principles";
            style=dashed;
            color="#696969"; # Dark Grey
            node [shape=note, fillcolor="#FFFACD", fontname="Arial", fontsize=10]; # Lemon Chiffon
            principle1 [label="Closer to CPU:\nFaster, Smaller, More Expensive"];
            principle2 [label="Farther from CPU:\nSlower, Larger, Cheaper"];
            principle3 [label="Locality of Reference\n(Temporal & Spatial)"];
            principle4 [label="Caching:\nMaintains copies of data from lower levels"];
        }
    }
''')

st.markdown("---")
st.header("2. Detailed Characteristics of Each Memory Level")
st.write("This table provides a comprehensive overview of each memory level's properties.")
st.dataframe(df.set_index("Level"), use_container_width=True)

st.markdown("---")
st.header("3. Understanding the Hierarchy: Key Principles")
st.markdown("""
The memory hierarchy is a fundamental concept in computer architecture designed to bridge the immense speed gap
between the lightning-fast CPU and relatively slower, but much larger and cheaper, storage devices.
It leverages several key principles:

* **Locality of Reference:** This is the cornerstone. Programs tend to access data and instructions that are:
    * **Temporal Locality:** Recently accessed items are likely to be accessed again soon.
    * **Spatial Locality:** Items whose addresses are close to a recently accessed item are likely to be accessed soon.
    Caches exploit this by bringing blocks of data (not just single bytes) into faster memory.
* **Caching:** Faster, smaller memory levels (like L1, L2, L3 caches) store copies of data from slower,
    larger memory levels. When the CPU needs data, it first checks the fastest cache.
    * **Cache Hit:** If the data is found, it's retrieved very quickly.
    * **Cache Miss:** If not found, the data is fetched from the next slower level, brought into the current cache,
        and then provided to the CPU. This process continues down the hierarchy.
* **Trade-offs:** Every level in the hierarchy represents a balance:
    * **Speed vs. Capacity:** Faster memory is generally smaller in capacity.
    * **Speed vs. Cost:** Faster memory is significantly more expensive per bit.
    The hierarchy optimizes overall system performance and cost by providing multiple levels of memory with varying characteristics.
""")

st.markdown("---")
st.header("4. Real-World Examples & Impact")
st.markdown("""
To better understand the memory hierarchy, let's look at how it affects your everyday computer usage:

* **Analogy: The Chef's Kitchen 👨‍🍳**
    * **CPU:** The Chef (the worker).
    * **Registers:** The ingredients currently in the Chef's hands or on the cutting board (immediate use, tiny amount, super fast).
    * **L1/L2 Cache:** The small spice rack (L1) right next to the cutting board, and the pantry (L2) a few steps away in the kitchen (frequently used ingredients, fast access).
    * **L3 Cache:** The walk-in refrigerator (L3) just outside the kitchen door (larger, takes a short walk, shared by multiple chefs).
    * **Main Memory (RAM):** The main grocery store in town (much larger, but requires leaving the kitchen and traveling). All the ingredients you've bought for today's dishes are here.
    * **Secondary Storage (SSD/HDD):** The vast warehouse on the outskirts of the city where all raw ingredients are stored (enormous capacity, but takes a long time to get anything). This is where you store ingredients you might need next month.

* **Opening an Application (e.g., a Web Browser):**
    * When you click the browser icon, its program code and initial data are loaded from **Secondary Storage (SSD/HDD)** into **Main Memory (RAM)**. This is why apps take time to launch.
    * As you interact with the browser, frequently used parts of its code and data (e.g., your Browse history, currently open tabs) are moved from RAM into **L3, L2, and L1 Caches** for quick access by the **CPU**.

* **Browse a Website (Scrolling):**
    * When you scroll down a long web page, the browser predicts you'll need the next section of content. This "pre-fetching" leverages **spatial locality**, bringing data from **RAM** into **cache** before the **CPU** even explicitly requests it. This makes scrolling feel smooth.

* **Playing a Game:**
    * The game's executable and large assets (textures, models) reside on **Secondary Storage**.
    * The actively loaded level, characters, and currently used game data are in **Main Memory (RAM)**.
    * The very specific instructions and small data sets the **CPU** needs *right now* (e.g., calculating character movement, rendering a specific frame) are brought into **cache**. A "cache miss" (CPU needing data not in cache) can cause micro-stutters.

* **Running Out of RAM ("System Slows Down"):**
    * If you open too many programs or large files, your **Main Memory (RAM)** might become full.
    * The operating system then has to temporarily move (swap) data from RAM back to **Secondary Storage (SSD/HDD)** to free up space. This is called **virtual memory** or "paging."
    * Accessing data from disk is thousands to millions of times slower than RAM, leading to a noticeable slowdown, often indicated by disk activity lights flickering constantly.

Understanding this hierarchy helps you appreciate why more RAM often improves multitasking, why SSDs feel much faster than HDDs, and why a CPU with a larger cache can perform better in certain tasks.
""")

st.markdown("---")
st.info("Developed by [Your Name/Organization Name] for educational purposes. Feel free to explore and learn!")
