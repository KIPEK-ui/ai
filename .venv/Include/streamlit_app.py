import streamlit as st
import os



# --- PAGE SETUP ---
webscrapper_page = st.Page(
    "views/webscrapper.py",
    title="Web Scrapper App",
    icon=":material/terrain:",
    default=True,
)


# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "AI APPs": [webscrapper_page],
    }
)

# --- SHARED ON ALL PAGES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'assets', 'logoi.jpg')
image_path_2 = os.path.join(current_dir, 'assets', 'logo.png')
st.logo(image_path_2)  
#st.image(image_path_2, width=120)# Adjust the width as needed
st.sidebar.markdown("Made with Prescision")

pg.run()
    