import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import py3Dmol
from datetime import datetime
import os
import csv
import time
import traceback
import logging
import json
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom utilities
from utils.molecular_utils import (
    validate_smiles, 
    generate_molecule_from_smiles, 
    get_molecule_svg, 
    get_molecule_properties,
    get_molecule_3d_data,
    process_batch_smiles
)
from utils.prediction_utils import log_prediction, get_prediction_history
from utils.auth_utils import (
    init_auth, 
    register_user, 
    login_user, 
    validate_session, 
    logout_user
)
from models.toxicity import predict_toxicity
from models.druglikeness import calculate_druglikeness

# Initialize auth system
init_auth()

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Drug Discovery Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
    
# Load existing predictions from logs.csv
try:
    if os.path.exists('data/logs.csv'):
        st.session_state.prediction_history = get_prediction_history()
except Exception as e:
    logger.error(f"Error loading prediction history: {str(e)}")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5 !important;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.8rem !important;
        color: #0D47A1 !important;
        margin-top: 10px;
    }
    .description {
        font-size: 1.1rem !important;
        color: #555555;
        margin-bottom: 30px;
        text-align: center;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .result-container {
        border: 1px solid #BBDEFB;
        border-radius: 5px;
        padding: 20px;
        margin-top: 30px;
        background-color: #F7FBFF;
    }
    .property-title {
        font-weight: bold;
        color: #0D47A1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F8FF;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #BBDEFB;
    }
</style>
""", unsafe_allow_html=True)

# Header with improved styling
st.markdown('<h1 class="main-header">AI-Powered Drug Discovery Platform</h1>', unsafe_allow_html=True)
st.markdown("""
<p class="description">
    Accelerate drug discovery with AI-powered predictions for molecular properties,
    toxicity assessment, and drug-likeness evaluation from chemical structures.
</p>
""", unsafe_allow_html=True)

# Define example molecules
example_smiles = {
    "Select an example": "",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Penicillin G": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
    "Sildenafil": "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    "Fluoxetine": "CNCCC(OC1=CC=C(C=C1)C(F)(F)F)C2=CC=CC=C2"
}

# Check if user is authenticated
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.session_id = None

# Authentication sidebar
with st.sidebar:
    st.markdown("## 👤 User Authentication")
    
    if not st.session_state.authenticated:
        auth_option = st.radio("", ["Login", "Register", "Reset Password"], horizontal=True)
        
        if auth_option == "Login":
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_button"):
                if username and password:
                    success, message, session_id = login_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.session_id = session_id
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter both username and password")
        
        elif auth_option == "Register":
            username = st.text_input("Username", key="register_username")
            password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.button("Register", key="register_button"):
                if username and password:
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        success, message = register_user(username, password)
                        if success:
                            st.success(message)
                            # Automatically log in after registration
                            success, _, session_id = login_user(username, password)
                            if success:
                                st.session_state.authenticated = True
                                st.session_state.username = username
                                st.session_state.session_id = session_id
                                st.rerun()
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill out all fields")
        
        else:  # Reset Password
            username = st.text_input("Username", key="reset_username")
            new_password = st.text_input("New Password", type="password", key="reset_password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Reset Password", key="reset_button"):
                if username and new_password:
                    if new_password != confirm_new_password:
                        st.error("Passwords do not match")
                    else:
                        success, message = reset_password(username, new_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill out all fields")
    else:
        st.markdown(f"**Welcome, {st.session_state.username}!**")
        if st.button("Logout"):
            logout_user(st.session_state.session_id)
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.session_id = None
            st.rerun()
    
    st.markdown("---")
    
    # Additional sidebar features
    st.markdown("## 🛠️ Tools")
    st.markdown("- **Settings**: Configure API connections")
    st.markdown("- **Documentation**: View complete documentation")
    st.markdown("- **Export**: Download results as CSV or JSON")
    
    st.markdown("## 📱 Contact")
    st.markdown("For support, contact: support@drugdiscoveryplatform.ai")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💊 Molecular Prediction", 
    "🔬 2D Visualization", 
    "🧪 3D Structure", 
    "🧮 Batch Processing",
    "📊 History", 
    "ℹ️ About"
])

with tab1:
    st.markdown('<h2 class="sub-header">Molecular Property Prediction</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Enter a SMILES string or select from examples to predict molecular properties, 
        toxicity, and drug-likeness metrics. The prediction uses AI models to evaluate 
        your compound.
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Input field for SMILES string
        smiles_input = st.text_input("Enter SMILES string:", 
                                     placeholder="E.g., CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)",
                                     help="SMILES (Simplified Molecular Input Line Entry System) is a notation that represents chemical structures")
    
    with col2:
        # Example SMILES dropdown
        selected_example = st.selectbox("Or select from examples:", 
                                        options=list(example_smiles.keys()),
                                        help="Choose a pre-defined molecule to analyze")
        
        if selected_example != "Select an example":
            smiles_input = example_smiles[selected_example]
    
    # Create columns for the prediction button and spinner
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button("🔍 Predict Properties", 
                                  use_container_width=True,
                                  help="Click to analyze the molecular properties")
    
    # Display status
    prediction_status = st.empty()
    
    # Process the prediction when the button is clicked
    if predict_button and smiles_input:
        with st.spinner("Analyzing molecule structure and properties..."):
            prediction_status.info("Validating molecular structure...")
            
            # Validate SMILES string
            is_valid, message = validate_smiles(smiles_input)
            
            if not is_valid:
                prediction_status.error(f"Invalid SMILES string: {message}")
            else:
                try:
                    prediction_status.info("Generating molecular model...")
                    # Generate molecule
                    molecule = generate_molecule_from_smiles(smiles_input)
                    
                    prediction_status.info("Calculating basic properties...")
                    # Get molecular properties
                    properties = get_molecule_properties(molecule)
                    
                    prediction_status.info("Predicting toxicity profile...")
                    # Predict toxicity
                    toxicity_results = predict_toxicity(molecule)
                    
                    prediction_status.info("Assessing drug-likeness...")
                    # Calculate drug-likeness
                    druglikeness_results = calculate_druglikeness(molecule)
                    
                    # Clear status message
                    prediction_status.empty()
                    
                    # Create combined results
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    results = {
                        "timestamp": timestamp,
                        "smiles": smiles_input,
                        "molecular_weight": properties["molecular_weight"],
                        "logP": properties["logP"],
                        "num_hydrogen_donors": properties["num_hydrogen_donors"],
                        "num_hydrogen_acceptors": properties["num_hydrogen_acceptors"],
                        "molar_refractivity": properties["molar_refractivity"],
                        "topological_polar_surface_area": properties["topological_polar_surface_area"],
                        "toxicity_probability": toxicity_results["toxicity_probability"],
                        "toxicity_class": toxicity_results["toxicity_class"],
                        "lipinski_violations": druglikeness_results["lipinski_violations"],
                        "druglikeness_score": druglikeness_results["druglikeness_score"]
                    }
                    
                    # Add to history
                    st.session_state.prediction_history.append(results)
                    
                    # Log prediction
                    log_prediction(results)
                    
                    # Display results container
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    
                    # Display molecule name if from examples
                    if selected_example != "Select an example":
                        st.markdown(f"<h3 style='text-align:center; color:#1E88E5;'>{selected_example}</h3>", unsafe_allow_html=True)
                    
                    # Display molecular structure
                    try:
                        molecule_svg = get_molecule_svg(molecule, width=400, height=250)
                        st.markdown(molecule_svg, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not generate molecule visualization: {str(e)}")
                    
                    # Display results in tables
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown('<p class="property-title">Basic Properties</p>', unsafe_allow_html=True)
                        prop_df = pd.DataFrame({
                            "Property": ["Molecular Weight", "LogP", "H-Bond Donors", "H-Bond Acceptors", 
                                        "Molar Refractivity", "Topological Polar Surface Area"],
                            "Value": [
                                f"{properties['molecular_weight']:.2f} g/mol",
                                f"{properties['logP']:.2f}",
                                properties['num_hydrogen_donors'],
                                properties['num_hydrogen_acceptors'],
                                f"{properties['molar_refractivity']:.2f}",
                                f"{properties['topological_polar_surface_area']:.2f} Å²"
                            ]
                        })
                        st.table(prop_df)
                    
                    with col2:
                        st.markdown('<p class="property-title">Toxicity & Drug-likeness</p>', unsafe_allow_html=True)
                        combined_df = pd.DataFrame({
                            "Metric": ["Toxicity Probability", "Toxicity Class", "Lipinski Violations", "Drug-likeness Score"],
                            "Value": [
                                f"{toxicity_results['toxicity_probability']:.2f}",
                                toxicity_results['toxicity_class'],
                                druglikeness_results['lipinski_violations'],
                                f"{druglikeness_results['druglikeness_score']:.2f}"
                            ]
                        })
                        st.table(combined_df)
                    
                    # Create radar chart for drug discovery profile
                    st.markdown('<p class="property-title">Drug Discovery Profile</p>', unsafe_allow_html=True)
                    
                    radar_categories = [
                        'MW (÷500)', 'LogP (Scaled)', 'H-Donors (÷5)', 
                        'H-Acceptors (÷10)', 'Drug-likeness'
                    ]
                    
                    # Normalize values for radar chart (0-1 scale)
                    radar_values = [
                        min(properties['molecular_weight'] / 500, 1),
                        max(0, min((properties['logP'] + 5) / 10, 1)),  # Scaling logP from -5 to 5
                        min(properties['num_hydrogen_donors'] / 5, 1),
                        min(properties['num_hydrogen_acceptors'] / 10, 1),
                        max(0, min(druglikeness_results['druglikeness_score'], 1))
                    ]
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=radar_values,
                        theta=radar_categories,
                        fill='toself',
                        name='Compound Profile',
                        line=dict(color='#1E88E5')
                    ))
                    
                    # Add reference for optimal values
                    fig.add_trace(go.Scatterpolar(
                        r=[0.8, 0.5, 0.6, 0.6, 0.8],
                        theta=radar_categories,
                        fill='toself',
                        name='Optimal Drug Profile',
                        line=dict(color='#4CAF50'),
                        opacity=0.2
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=True,
                        height=450
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed toxicity information
                    if toxicity_results.get("toxicity_details"):
                        with st.expander("Detailed Toxicity Analysis"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if toxicity_results["toxicity_details"].get("reactive_groups"):
                                    st.markdown("**Reactive Groups:**")
                                    for group in toxicity_results["toxicity_details"]["reactive_groups"]:
                                        st.markdown(f"- {group}")
                                else:
                                    st.markdown("**Reactive Groups:** None detected")
                            
                            with col2:
                                if toxicity_results["toxicity_details"].get("structural_alerts"):
                                    st.markdown("**Structural Alerts:**")
                                    for alert in toxicity_results["toxicity_details"]["structural_alerts"]:
                                        st.markdown(f"- {alert}")
                                else:
                                    st.markdown("**Structural Alerts:** None detected")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    logger.error(f"Error during prediction: {str(e)}")
                    logger.error(traceback.format_exc())
                    prediction_status.error(f"Error during prediction: {str(e)}")
    elif predict_button:
        prediction_status.warning("Please enter a SMILES string or select an example molecule.")

with tab2:
    st.markdown('<h2 class="sub-header">2D Molecular Visualization</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Visualize molecular structures from SMILES strings. The visualization shows a 2D representation 
        of the molecule with standard atom colors and bond types.
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for input in visualization tab
    vis_col1, vis_col2 = st.columns([1, 1])
    
    with vis_col1:
        # Input field for visualization
        vis_smiles = st.text_input("Enter SMILES string for visualization:", 
                                  value=smiles_input if 'smiles_input' in locals() and smiles_input else "",
                                  help="Enter a SMILES string to visualize its structure")
    
    with vis_col2:
        # Example selector for visualization
        vis_example = st.selectbox("Or select from examples for visualization:", 
                                 options=list(example_smiles.keys()), 
                                 key="vis_example",
                                 help="Choose a pre-defined molecule to visualize")
        
        if vis_example != "Select an example":
            vis_smiles = example_smiles[vis_example]
    
    # Create visualization button
    visualize_status = st.empty()
    vis_button = st.button("🔍 Visualize Molecule", use_container_width=True)
    
    if (vis_button or vis_smiles) and vis_smiles:
        try:
            visualize_status.info("Validating SMILES string...")
            is_valid, message = validate_smiles(vis_smiles)
            
            if not is_valid:
                visualize_status.error(f"Invalid SMILES string for visualization: {message}")
            else:
                visualize_status.info("Generating molecule and visualization...")
                # Generate molecule
                molecule = generate_molecule_from_smiles(vis_smiles)
                
                # Get molecule SVG
                svg_string = get_molecule_svg(molecule, width=800, height=500)
                
                # Clear status
                visualize_status.empty()
                
                # Create a container for visualization
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                # Display name if from examples
                if vis_example != "Select an example":
                    st.markdown(f"<h3 style='text-align:center; color:#1E88E5;'>{vis_example}</h3>", unsafe_allow_html=True)
                
                # Display the molecule
                st.markdown('<p class="property-title">Molecular Structure</p>', unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center;'><b>SMILES:</b> {vis_smiles}</p>", unsafe_allow_html=True)
                st.markdown(svg_string, unsafe_allow_html=True)
                
                # Display basic properties
                try:
                    properties = get_molecule_properties(molecule)
                    st.markdown('<p class="property-title">Basic Properties</p>', unsafe_allow_html=True)
                    
                    # Create columns for properties
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    
                    with pc1:
                        st.metric("Molecular Weight", f"{properties['molecular_weight']:.1f}")
                    with pc2:
                        st.metric("LogP", f"{properties['logP']:.2f}")
                    with pc3:
                        st.metric("H-Donors", properties['num_hydrogen_donors'])
                    with pc4:
                        st.metric("H-Acceptors", properties['num_hydrogen_acceptors'])
                        
                except Exception as e:
                    st.warning(f"Could not calculate molecular properties: {str(e)}")
                
                # Show 3D options
                with st.expander("Advanced Visualization Options"):
                    st.markdown("""
                    For 3D visualization, we recommend using external tools like PyMol, Chimera, or online viewers.
                    
                    You can:
                    1. Export the molecule as a SDF or PDB file
                    2. Use online visualization tools like [MolView](https://molview.org/)
                    3. Import the SMILES string into molecular modeling software
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}")
            logger.error(traceback.format_exc())
            visualize_status.error(f"Error during visualization: {str(e)}")

with tab3:
    st.markdown('<h2 class="sub-header">3D Molecular Structure</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Visualize molecular structures in 3D using an interactive viewer. You can rotate, zoom, and explore
        the three-dimensional structure of molecules to better understand their spatial arrangement.
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for input in 3D visualization tab
    vis3d_col1, vis3d_col2 = st.columns([1, 1])
    
    with vis3d_col1:
        # Input field for 3D visualization
        vis3d_smiles = st.text_input("Enter SMILES string for 3D visualization:", 
                                  key="vis3d_smiles",
                                  placeholder="E.g., CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)",
                                  help="Enter a SMILES string to visualize its 3D structure")
    
    with vis3d_col2:
        # Example selector for 3D visualization
        vis3d_example = st.selectbox("Or select from examples for 3D visualization:", 
                                 options=list(example_smiles.keys()), 
                                 key="vis3d_example",
                                 help="Choose a pre-defined molecule to visualize in 3D")
        
        if vis3d_example != "Select an example":
            vis3d_smiles = example_smiles[vis3d_example]
    
    # Create visualization button and display options
    vis3d_status = st.empty()
    vis3d_col1, vis3d_col2 = st.columns([1, 1])
    
    with vis3d_col1:
        vis3d_button = st.button("🔍 Generate 3D Structure", key="vis3d_button", use_container_width=True)
    
    with vis3d_col2:
        style_options = st.selectbox(
            "Visualization style:",
            options=["Stick", "Ball and Stick", "Sphere", "Cartoon", "Wireframe"],
            index=1,
            key="vis3d_style"
        )
    
    # Process the 3D visualization
    if (vis3d_button or vis3d_smiles) and vis3d_smiles:
        try:
            vis3d_status.info("Validating SMILES string...")
            is_valid, message = validate_smiles(vis3d_smiles)
            
            if not is_valid:
                vis3d_status.error(f"Invalid SMILES string for 3D visualization: {message}")
            else:
                vis3d_status.info("Generating 3D molecule structure...")
                
                # Generate molecule with 3D coordinates
                molecule = generate_molecule_from_smiles(vis3d_smiles)
                
                # Get 3D visualization data
                mol_data = get_molecule_3d_data(molecule)
                
                if mol_data is None:
                    vis3d_status.error("Could not generate 3D coordinates for this molecule")
                else:
                    # Clear status
                    vis3d_status.empty()
                    
                    # Display molecule information
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    
                    # Display name if from examples
                    if vis3d_example != "Select an example":
                        st.markdown(f"<h3 style='text-align:center; color:#1E88E5;'>{vis3d_example}</h3>", unsafe_allow_html=True)
                    
                    st.markdown(f"<p style='text-align:center;'><b>SMILES:</b> {vis3d_smiles}</p>", unsafe_allow_html=True)
                    
                    # Create py3Dmol viewer
                    viewer_height = 500
                    viewer = py3Dmol.view(width=None, height=viewer_height)
                    
                    # Map style selection to py3Dmol styles
                    style_mapping = {
                        "Stick": {"stick": {}},
                        "Ball and Stick": {"stick": {}, "sphere": {"scale": 0.3}},
                        "Sphere": {"sphere": {}},
                        "Cartoon": {"cartoon": {}},
                        "Wireframe": {"line": {}}
                    }
                    
                    selected_style = style_mapping.get(style_options, {"stick": {}, "sphere": {"scale": 0.3}})
                    
                    # Add atoms to viewer
                    for i, atom in enumerate(mol_data['atoms']):
                        pos = mol_data['positions'][i]
                        viewer.addSphere({
                            'center': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
                            'radius': atom['size'],
                            'color': atom['color']
                        })
                    
                    # Add bonds to viewer
                    for bond in mol_data['bonds']:
                        begin_idx = bond['begin_atom_idx']
                        end_idx = bond['end_atom_idx']
                        begin_pos = mol_data['positions'][begin_idx]
                        end_pos = mol_data['positions'][end_idx]
                        
                        # Create a cylinder for each bond
                        viewer.addCylinder({
                            'start': {'x': begin_pos[0], 'y': begin_pos[1], 'z': begin_pos[2]},
                            'end': {'x': end_pos[0], 'y': end_pos[1], 'z': end_pos[2]},
                            'radius': 0.1,
                            'color': '#909090',
                            'fromCap': True,
                            'toCap': True
                        })
                    
                    # Set camera and render
                    viewer.zoomTo()
                    viewer.setBackgroundColor('#f8f9fa')
                    viewer.spin(True)
                    
                    # Render the viewer
                    viewer_html = viewer.render()
                    st.components.v1.html(viewer_html, height=viewer_height)
                    
                    # Display basic properties
                    try:
                        properties = get_molecule_properties(molecule)
                        st.markdown('<p class="property-title">Basic Properties</p>', unsafe_allow_html=True)
                        
                        # Create columns for properties
                        pc1, pc2, pc3, pc4 = st.columns(4)
                        
                        with pc1:
                            st.metric("Molecular Weight", f"{properties['molecular_weight']:.1f}")
                        with pc2:
                            st.metric("LogP", f"{properties['logP']:.2f}")
                        with pc3:
                            st.metric("H-Donors", properties['num_hydrogen_donors'])
                        with pc4:
                            st.metric("H-Acceptors", properties['num_hydrogen_acceptors'])
                            
                    except Exception as e:
                        st.warning(f"Could not calculate molecular properties: {str(e)}")
                    
                    # Add viewer controls explanation
                    with st.expander("3D Viewer Controls"):
                        st.markdown("""
                        - **Rotate**: Click and drag with the mouse
                        - **Zoom**: Use the scroll wheel or pinch gesture
                        - **Reset view**: Double-click in the viewer
                        - **The molecule is rotating automatically**: Click to pause rotation
                        """)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
        except Exception as e:
            logger.error(f"Error during 3D visualization: {str(e)}")
            logger.error(traceback.format_exc())
            vis3d_status.error(f"Error during 3D visualization: {str(e)}")
            
with tab4:
    st.markdown('<h2 class="sub-header">Batch Processing</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Process multiple molecules at once by entering multiple SMILES strings. 
        Batch processing allows you to analyze many compounds simultaneously and compare their properties.
    </div>
    """, unsafe_allow_html=True)
    
    # Text area for batch input
    batch_smiles = st.text_area(
        "Enter multiple SMILES strings (one per line):",
        height=150,
        help="Enter one SMILES string per line to process multiple molecules at once"
    )
    
    # Example set selection
    st.markdown("### Or select from example sets:")
    
    example_sets = {
        "Select an example set": [],
        "Common Pain Relievers": [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC(=O)NC1=CC=C(C=C1)O"  # Acetaminophen (Paracetamol)
        ],
        "Antibiotics": [
            "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O",  # Penicillin G
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Amoxicillin (simplified)
            "COC1=C(OC)C=C2C(=O)C(C3CCN(C)CC3)C(O)=C(O2)C2=C1OC(C)=CC2=O"  # Tetracycline (simplified)
        ],
        "Anti-diabetic Drugs": [
            "CC1=CN(C2OC(CO)C(O)C2O)C(=O)NC1=O",  # Metformin (simplified)
            "CN(C)C(=N)NC(=N)N",  # Glyburide (simplified)
            "CC1=CC=C(C=C1)C1CC(=NN1)C(=O)NCCC1=CC=CC=C1"  # Sitagliptin (simplified)
        ]
    }
    
    selected_set = st.selectbox(
        "Example compound sets:",
        options=list(example_sets.keys()),
        key="batch_example_set"
    )
    
    if selected_set != "Select an example set":
        batch_smiles = "\n".join(example_sets[selected_set])
        st.code(batch_smiles, language="plaintext")
    
    # Process button
    batch_status = st.empty()
    process_batch_button = st.button("🔍 Process Batch", use_container_width=True)
    
    if process_batch_button and batch_smiles:
        try:
            # Split input into lines and remove empty lines
            smiles_list = [line.strip() for line in batch_smiles.split("\n") if line.strip()]
            
            if not smiles_list:
                batch_status.warning("Please enter at least one valid SMILES string")
            else:
                batch_status.info(f"Processing {len(smiles_list)} molecules...")
                
                # Process batch
                batch_results = process_batch_smiles(smiles_list)
                
                # Display results
                valid_results = [r for r in batch_results if r['valid']]
                invalid_results = [r for r in batch_results if not r['valid']]
                
                batch_status.success(f"Processed {len(valid_results)} valid molecules ({len(invalid_results)} invalid)")
                
                # Display statistics
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                if valid_results:
                    st.markdown('<p class="property-title">Batch Results</p>', unsafe_allow_html=True)
                    
                    # Create DataFrame for valid results
                    data = []
                    for result in valid_results:
                        data.append({
                            'SMILES': result['smiles'],
                            'Molecular Weight': result['properties']['molecular_weight'],
                            'LogP': result['properties']['logP'],
                            'H-Donors': result['properties']['num_hydrogen_donors'],
                            'H-Acceptors': result['properties']['num_hydrogen_acceptors'],
                            'Toxicity Score': result['toxicity']['toxicity_probability'],
                            'Druglikeness Score': result['druglikeness']['druglikeness_score']
                        })
                    
                    results_df = pd.DataFrame(data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Add the batch results to history
                    for result in valid_results:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        history_entry = {
                            "timestamp": timestamp,
                            "smiles": result['smiles'],
                            "molecular_weight": result['properties']['molecular_weight'],
                            "logP": result['properties']['logP'],
                            "num_hydrogen_donors": result['properties']['num_hydrogen_donors'],
                            "num_hydrogen_acceptors": result['properties']['num_hydrogen_acceptors'],
                            "molar_refractivity": result['properties']['molar_refractivity'],
                            "topological_polar_surface_area": result['properties']['topological_polar_surface_area'],
                            "toxicity_probability": result['toxicity']['toxicity_probability'],
                            "toxicity_class": result['toxicity']['toxicity_class'],
                            "lipinski_violations": result['druglikeness']['lipinski_violations'],
                            "druglikeness_score": result['druglikeness']['druglikeness_score']
                        }
                        
                        st.session_state.prediction_history.append(history_entry)
                        log_prediction(history_entry)
                    
                    # Visualize property distributions
                    st.markdown('<p class="property-title">Property Distributions</p>', unsafe_allow_html=True)
                    
                    # Create tabs for different visualizations
                    viz_tabs = st.tabs(["Molecular Weight", "LogP", "Toxicity", "Druglikeness"])
                    
                    with viz_tabs[0]:
                        fig = px.histogram(
                            results_df, 
                            x="Molecular Weight",
                            nbins=10,
                            title="Distribution of Molecular Weights",
                            color_discrete_sequence=['#1E88E5']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with viz_tabs[1]:
                        fig = px.histogram(
                            results_df, 
                            x="LogP",
                            nbins=10,
                            title="Distribution of LogP Values",
                            color_discrete_sequence=['#1E88E5']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with viz_tabs[2]:
                        fig = px.histogram(
                            results_df, 
                            x="Toxicity Score",
                            nbins=10,
                            title="Distribution of Toxicity Scores",
                            color_discrete_sequence=['#1E88E5']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with viz_tabs[3]:
                        fig = px.histogram(
                            results_df, 
                            x="Druglikeness Score",
                            nbins=10,
                            title="Distribution of Druglikeness Scores",
                            color_discrete_sequence=['#1E88E5']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add download options
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_results.csv",
                        mime="text/csv"
                    )
                    
                if invalid_results:
                    with st.expander(f"View {len(invalid_results)} Invalid SMILES"):
                        for result in invalid_results:
                            st.code(f"{result['smiles']} - Error: {result['error']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            logger.error(f"Error during batch processing: {str(e)}")
            logger.error(traceback.format_exc())
            batch_status.error(f"Error during batch processing: {str(e)}")
                
    elif process_batch_button:
        batch_status.warning("Please enter at least one SMILES string for batch processing")

with tab5:
    st.markdown('<h2 class="sub-header">Prediction History</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        View the history of all molecular predictions made in this session and analyze trends 
        in the data. Download your prediction history for further analysis.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.prediction_history:
        # Create dataframe from history
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Add a search box to filter by SMILES
        search_query = st.text_input("Search by SMILES substring:", 
                                    placeholder="Enter part of a SMILES string to filter",
                                    help="Filter the history table by typing part of a SMILES string")
        
        # Filter the dataframe if search is provided
        if search_query:
            filtered_df = history_df[history_df['smiles'].str.contains(search_query, case=False)]
            if len(filtered_df) == 0:
                st.warning(f"No matches found for '{search_query}'")
                display_df = history_df
            else:
                display_df = filtered_df
        else:
            display_df = history_df
        
        # Display history table with improved formatting
        st.markdown('<p class="property-title">Prediction Records</p>', unsafe_allow_html=True)
        
        # Format the dataframe for display
        display_columns = [
            'timestamp', 'smiles', 'molecular_weight', 'logP', 
            'toxicity_probability', 'toxicity_class', 'druglikeness_score'
        ]
        
        # Add column for visual indicator of druglikeness
        display_df_styled = display_df[display_columns].copy()
        
        # Display the table
        st.dataframe(display_df_styled, use_container_width=True)
        
        # Show number of records
        st.info(f"Displaying {len(display_df)} of {len(history_df)} prediction records")
        
        # Option to download history as CSV
        col1, col2 = st.columns(2)
        
        with col1:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full History as CSV",
                data=csv,
                file_name="drug_discovery_predictions.csv",
                mime="text/csv",
                help="Download all prediction records as a CSV file"
            )
        
        with col2:
            # Option to clear history
            if st.button("🗑️ Clear Session History", 
                        help="Remove all prediction records from the current session"):
                st.session_state.prediction_history = []
                st.rerun()
        
        # Visualize property distributions
        st.markdown('<p class="property-title">Property Distributions</p>', unsafe_allow_html=True)
        
        # Create columns for property selection and chart type
        prop_col1, prop_col2 = st.columns(2)
        
        with prop_col1:
            # Create selection for property to visualize
            property_to_plot = st.selectbox(
                "Select property to analyze:",
                ['molecular_weight', 'logP', 'num_hydrogen_donors', 'num_hydrogen_acceptors', 
                'toxicity_probability', 'druglikeness_score'],
                help="Choose which molecular property to visualize"
            )
        
        with prop_col2:
            chart_type = st.selectbox(
                "Select chart type:",
                ["Histogram", "Box Plot", "Scatter Plot"],
                help="Choose the visualization type for the selected property"
            )
        
        # Create the selected chart type
        if chart_type == "Histogram":
            fig = px.histogram(
                display_df, 
                x=property_to_plot,
                nbins=20,
                title=f"Distribution of {property_to_plot}",
                color_discrete_sequence=['#1E88E5']
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Box Plot":
            fig = px.box(
                display_df,
                y=property_to_plot,
                title=f"Box Plot of {property_to_plot}",
                color_discrete_sequence=['#1E88E5']
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Scatter Plot":
            # If scatter plot, allow selection of x and y
            second_property = st.selectbox(
                "Select second property for scatter plot:",
                [p for p in ['molecular_weight', 'logP', 'num_hydrogen_donors', 'num_hydrogen_acceptors', 
                           'toxicity_probability', 'druglikeness_score'] if p != property_to_plot],
                help="Choose a second property for the scatter plot"
            )
            
            fig = px.scatter(
                display_df,
                x=property_to_plot,
                y=second_property,
                title=f"{property_to_plot} vs {second_property}",
                color="toxicity_class" if "toxicity_class" in display_df.columns else None,
                hover_data=["smiles"]
            )
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No predictions have been made yet. Make predictions in the 'Molecular Prediction' tab.")
        
        # Add a sample prediction button with a default molecule
        if st.button("Generate Sample Prediction", help="Create a sample prediction to see how the history works"):
            try:
                # Use Aspirin as a default
                sample_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
                
                # Generate molecule
                molecule = generate_molecule_from_smiles(sample_smiles)
                
                # Get molecular properties
                properties = get_molecule_properties(molecule)
                
                # Predict toxicity
                toxicity_results = predict_toxicity(molecule)
                
                # Calculate drug-likeness
                druglikeness_results = calculate_druglikeness(molecule)
                
                # Create combined results
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                results = {
                    "timestamp": timestamp,
                    "smiles": sample_smiles,
                    "molecular_weight": properties["molecular_weight"],
                    "logP": properties["logP"],
                    "num_hydrogen_donors": properties["num_hydrogen_donors"],
                    "num_hydrogen_acceptors": properties["num_hydrogen_acceptors"],
                    "molar_refractivity": properties["molar_refractivity"],
                    "topological_polar_surface_area": properties["topological_polar_surface_area"],
                    "toxicity_probability": toxicity_results["toxicity_probability"],
                    "toxicity_class": toxicity_results["toxicity_class"],
                    "lipinski_violations": druglikeness_results["lipinski_violations"],
                    "druglikeness_score": druglikeness_results["druglikeness_score"]
                }
                
                # Add to history
                st.session_state.prediction_history.append(results)
                
                # Log prediction
                log_prediction(results)
                
                st.success("Sample prediction created! Check the history tab.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error creating sample prediction: {str(e)}")

with tab4:
    st.markdown('<h2 class="sub-header">About This Platform</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        This AI-powered Drug Discovery Platform helps researchers analyze chemical compounds for their 
        potential as drug candidates. The platform combines molecular visualization, property prediction, 
        and drug-likeness assessment to accelerate the drug discovery process.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### How It Works")
    
    # Create columns for the how it works section
    hw_col1, hw_col2 = st.columns([1, 2])
    
    with hw_col1:
        st.markdown("""
        **Input:**
        - SMILES string representation of molecules
        - Selection from example compounds
        
        **Processing:**
        - Molecular structure validation
        - Generation of 3D coordinates
        - Property calculation using cheminformatics
        - AI-based toxicity prediction
        - Drug-likeness assessment
        
        **Output:**
        - Molecular visualization
        - Comprehensive property analysis
        - Toxicity assessment
        - Drug-likeness score
        - Historical data analysis
        """)
    
    with hw_col2:
        st.markdown("#### The SMILES Format")
        st.markdown("""
        SMILES (Simplified Molecular Input Line Entry System) is a notation that represents the structure of chemical compounds using short ASCII strings. For example:
        
        - `CC(=O)OC1=CC=CC=C1C(=O)O` represents Aspirin
        - `C1=CC=C(C=C1)C(=O)O` represents Benzoic acid
        - `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` represents Caffeine
        
        The notation uses letters to represent atoms (C for carbon, O for oxygen, etc.), numbers to indicate rings, and symbols for bonds and branching.
        """)
    
    st.markdown("### Key Features")
    
    # Create three columns for key features
    kf_col1, kf_col2, kf_col3 = st.columns(3)
    
    with kf_col1:
        st.markdown("#### Molecular Prediction")
        st.markdown("""
        - Basic molecular properties
        - Lipinski's Rule of Five analysis
        - Property visualization
        - AI-powered toxicity prediction
        """)
    
    with kf_col2:
        st.markdown("#### Visualization")
        st.markdown("""
        - 2D structure rendering
        - Interactive molecule display
        - Property highlighting
        - Export options
        """)
    
    with kf_col3:
        st.markdown("#### Analysis")
        st.markdown("""
        - Historical data tracking
        - Statistical analysis of properties
        - Trend visualization
        - Data export functionality
        """)
    
    st.markdown("### Technologies Used")
    
    st.markdown("""
    - **Streamlit**: Frontend web application framework
    - **RDKit**: Cheminformatics and molecular visualization
    - **Plotly**: Interactive data visualization
    - **Pandas**: Data manipulation and analysis
    - **NumPy**: Numerical computations
    - **py3Dmol**: 3D molecular visualization
    """)
    
    st.markdown("### Development Team")
    
    team_col1, team_col2 = st.columns(2)
    
    with team_col1:
        st.markdown("""
        **Team Members:**
        - Pravalika
        - Sumanth
        - Ramya
        - Yamuna
        """)
    
    with team_col2:
        st.markdown("""
        **Institution:**  
        Developed at Vignan Institute of Information Technology, 2025
        """)
        
    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <p>© 2025 AI-Powered Drug Discovery Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Disclaimer")
    
    st.markdown("""
    This platform is for research and educational purposes only. Predictions should be validated experimentally 
    before being used for any real-world applications in drug discovery. The toxicity predictions are based on 
    simplified models and should not replace proper laboratory testing.
    """)
    
    st.markdown("### About the Developers")
    
    st.markdown("""
    This AI-powered Drug Discovery Platform was developed to accelerate the early stages of drug discovery 
    by providing rapid assessment of molecular properties and potential drug-likeness of compounds.
    
    The platform is constantly evolving with new features and improved predictive capabilities.
    """)

# Sidebar content
with st.sidebar:
    st.markdown("## Quick Navigation")
    
    st.markdown("""
    - [Molecular Prediction](#molecular-prediction)
    - [Visualization](#visualization)
    - [Prediction History](#prediction-history)
    - [About](#about-this-platform)
    """)
    
    st.markdown("## Tips for Use")
    
    st.markdown("""
    - Enter SMILES strings or select from examples
    - View molecular properties and toxicity predictions
    - Visualize molecular structures in 2D
    - Track prediction history and download results
    """)
    
    # Add a feature for quick access to examples
    st.markdown("## Quick Examples")
    
    quick_example = st.selectbox(
        "Select a molecule to analyze:",
        list(example_smiles.keys())[1:],  # Skip the "Select an example" option
        key="sidebar_examples"
    )
    
    if st.button("Analyze Selected Molecule"):
        # Set the main tab to Molecular Prediction
        st.experimental_set_query_params(tab="Molecular-Prediction")
        
        # This is just to allow the user to click this - the actual tab switch
        # isn't possible directly, but we can modify session state
        if "selected_molecule" not in st.session_state:
            st.session_state.selected_molecule = quick_example
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("""
    **Version:** 1.0  
    **Last Updated:** March 2025
    """)

# Additional buttons for new users
if len(st.session_state.prediction_history) == 0:
    st.markdown("---")
    st.markdown("### 🚀 New to the Platform?")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Try with Aspirin"):
            # Set up state to analyze aspirin
            st.session_state.quick_analysis = "Aspirin"
            st.rerun()
            
    with col2:
        if st.button("Tour Features"):
            st.session_state.show_tour = True
            st.info("Check out the different tabs to explore all features of the platform!")
            
    with col3:
        if st.button("Read Documentation"):
            # Switch to the About tab
            st.session_state.active_tab = 3
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
**AI-Powered Drug Discovery Platform** | Built with Streamlit, RDKit, and DeepChem  
This platform is for research purposes only. Predictions should be validated experimentally.
""")
