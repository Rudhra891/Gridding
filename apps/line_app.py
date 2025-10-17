import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Function to determine direction
def get_direction(row, center):
    dlat = row["Latitude"] - center["Latitude"]
    dlon = row["Longitude"] - center["Longitude"]
    if dlat > 0 and dlon > 0: return "NE"
    if dlat > 0 and dlon < 0: return "NW"
    if dlat < 0 and dlon > 0: return "SE"
    if dlat < 0 and dlon < 0: return "SW"
    if dlat == 0 and dlon > 0: return "E"
    if dlat == 0 and dlon < 0: return "W"
    if dlon == 0 and dlat > 0: return "N"
    if dlon == 0 and dlat < 0: return "S"
    return "C"

def run():
    #st.title("📈 Geophysical Data – Traverses (Line View)")
    st.markdown("<h1 style='text-align: center;'>📈 Geophysical Data – Traverses<br><span style='font-size: 0.6em;'>(Line View)</span></h1>", unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"],help="upload your data file, if need input template format: download below")
    # Path to template excel file
    template_path = "template_Rudra Geophysicist.xlsx"

     # Download button
    with open(template_path, "rb") as file:
        st.download_button(
            label="📥 Download Input Template",
            data=file,
            help ="Input template",
            file_name="template_Rudra Geophysicist.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    if not uploaded_file:
        #st.info("👆 Please upload an Excel file to proceed.")
        return
    df = pd.read_excel(uploaded_file)
    
    
    
    # X-axis selection
    axis_choice = st.radio("Plot Resistivity (Res) vs:", ("Longitude", "Latitude", "Ref"),help="Click the parameter to see resistivity variations along the line respective to that parameter")

    selected_line = st.selectbox("Select LineNumber", sorted(df["LineNumber"].unique()),help="select survey line number from the dropdown")
    filtered = df[df["LineNumber"] == selected_line]

    # Selected Refs with priorities
    selected_refs = st.multiselect("Highlight Data Points", sorted(filtered["Ref"].unique()),help="Select the referece(s) to highlight the priority point(s) on the graph")
    prio_text = st.text_input("Priority labels (comma separated to match Refs):", "",help="Give the priority point(s) names **e.g P1,P2** as per '**Highlight Data Points**' order")
    priorities = [p.strip() for p in prio_text.split(",")] if prio_text else []

    # Ref range bars with colors and labels
     
        
    ref_range_help="""
                Enter reference value range(small to big) to 
                denote the geologica features/type of geology
                with color bar. (e.g. 5:10,15:20):
               """
    ranges_input = st.text_input("Reference ranges ", "",help=ref_range_help)
    ref_ranges = [r.strip() for r in ranges_input.split(",") if ":" in r]

    bar_colors, bar_labels = [], []
    for rr in ref_ranges:
        col = st.color_picker(f"Color for {rr}", "#DDDDDD",help="pick the color for bar")
        bar_colors.append(col)
        help_range_text = """
                         write the geology/geological features/anything scientific
                         related to this particular range (e.g. **weathered rock/ Fractured granite**)
                         """
        lbl = st.text_input(f"Legend label for {rr}", f"Range {rr}",help= help_range_text )
        bar_labels.append(lbl)

    # Compute bounding center for direction calculation
    center = {
        "Longitude": (filtered["Longitude"].max() + filtered["Longitude"].min()) / 2,
        "Latitude": (filtered["Latitude"].max() + filtered["Latitude"].min()) / 2
    }

    filtered["Direction"] = filtered.apply(lambda r: get_direction(r, center), axis=1)
    start, end = filtered.iloc[0], filtered.iloc[-1]
    line_label = f"Line{selected_line} ({start['Direction']}-{end['Direction']})"

    fig = go.Figure()

    # Determine axis baseline (for bar placement)
    y_axis_zero = min(filtered["Resistivity"].min(), 0)
    bar_thickness = (filtered["Resistivity"].max() - filtered["Resistivity"].min()) * 0.09

    # Add colored ranges
    shapes = []
    for idx, rr in enumerate(ref_ranges):
        try:
            a, b = map(int, rr.split(":"))
            sub = filtered[filtered["Ref"].between(a, b)]
            if sub.empty:
                continue

            if axis_choice == "Longitude":
                x0, x1 = sub["Longitude"].min(), sub["Longitude"].max()
            elif axis_choice == "Latitude":
                x0, x1 = sub["Latitude"].min(), sub["Latitude"].max()
            elif axis_choice == "Ref":
                x0, x1 = sub["Ref"].min(), sub["Ref"].max()

            shapes.append(dict(
                type="rect",
                xref="x", yref="y",
                x0=x0, x1=x1,
                y0=y_axis_zero,
                y1=y_axis_zero + bar_thickness,
                fillcolor=bar_colors[idx],
                opacity=1,
                layer="above",
                line=dict(width=0)
            ))
            # Legend dummy trace
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=20, color=bar_colors[idx], symbol='square', line=dict(width=0)),
                name=bar_labels[idx],
                showlegend=True
            ))              

        except:
            continue

    fig.update_layout(shapes=shapes)

    # Main line trace
    fig.add_trace(go.Scatter(
        x=filtered[axis_choice], y=filtered["Resistivity"],
        mode="lines+markers", name=line_label,
        line=dict(color="#1f77b4", width=2),
        marker=dict(color="#1f77b4", size=6)
    ))

    # Highlight selected Refs with labels
    for i, ref in enumerate(selected_refs):
        pt = filtered[filtered["Ref"] == ref].iloc[0]
        prio = priorities[i] if i < len(priorities) else ""
        legend_name = f"Ref {ref} – {prio}" if prio else f"Ref {ref}"
        fig.add_trace(go.Scatter(
            x=[pt[axis_choice]], y=[pt["Resistivity"]],
            mode="markers", name=legend_name,
            marker=dict(size=12, color="red", symbol="circle"),
            hovertemplate=legend_name + f"<br>Lat: {pt['Latitude']:.4f}<br>Lon: {pt['Longitude']:.4f}<extra></extra>"
        ))
        fig.add_annotation(
            x=pt[axis_choice], y=pt["Resistivity"], text=prio,
            showarrow=False, xshift=10, yshift=10,
            font=dict(color="red", size=10), bgcolor="white", opacity=0.7
        )

    # Direction annotations below x-axis
    fig.add_annotation(x=start[axis_choice], yref="paper", y=0, text=start["Direction"], showarrow=False,
                       font=dict(size=12), xanchor="center", yshift=-30)
    fig.add_annotation(x=end[axis_choice], yref="paper", y=0, text=end["Direction"], showarrow=False,
                       font=dict(size=12), xanchor="center", yshift=-30)

    # Axis labels
    fig.update_xaxes(
        title=axis_choice,
        showline=True, linecolor="black", mirror=True,
        ticks="outside", ticklen=8, tickcolor="black",
        minor=dict(ticks="inside", ticklen=4, gridcolor="lightgray"),
        showgrid=True, gridcolor="gray"
    )
    fig.update_yaxes(
        title="Resistivity",
        showline=True, linecolor="black", mirror=True,
        ticks="outside", ticklen=8, tickcolor="black",
        minor=dict(ticks="inside", ticklen=4, gridcolor="lightgray"),
        showgrid=True, gridcolor="gray"
    )

    # Final layout
    fig.update_layout(
        title=dict(text=f"Line {selected_line} – Resistivity vs {axis_choice}", x=0.5),
        template="plotly_white",
        margin=dict(t=60, b=120, l=60, r=40),
        legend=dict(title="")
    )

    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
